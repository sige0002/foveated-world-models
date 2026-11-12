from pathlib import Path
import json
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.train_cached_decoders import evaluate, file_hash, main


def test_code_ablation_uses_global_roll_in_last_batch():
    class KnownDynamics(torch.nn.Module):
        def forward(self, features, actions):
            return features + actions.unsqueeze(-1)
    h = torch.zeros(3, 1, 4, 8)
    z = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1)
    data = dict(current=h, future=h+z.unsqueeze(-1), codes=z,
                rows=[dict(episode=str(i), dt=1.2) for i in range(3)])
    result = evaluate(KnownDynamics(), data, batch_size=2)
    assert result['metrics']['mse'] == 0
    assert result['metrics']['shuffled_mse'] == 2
    assert result['rows'][-1]['shuffled_mse'] == 4


def test_cached_training_and_artifacts(tmp_path, monkeypatch):
    cache = tmp_path/'cache'
    cache.mkdir()
    for split in ('train','validation'):
        h = torch.randn(4, 1, 4, 8)
        z = torch.randn(4, 1, 4)
        torch.save(dict(current=h, future=h+0.1, codes=z,
                        rows=[dict(episode=f'{split}-{i}', dt=1.2) for i in range(4)]), cache/f'{split}.pt')
    (cache/'cache_manifest.json').write_text(json.dumps(dict(cache_sha256={
        split:file_hash(cache/f'{split}.pt') for split in ('train','validation')})))
    output = tmp_path/'experiment'
    monkeypatch.setattr(sys, 'argv', ['train_cached_decoders', '--cache',str(cache),
                         '--output',str(output),'--steps','2','--dim','16','--layers','1','--batch-size','3'])
    main()
    report = json.loads((output/'comparison.json').read_text())
    assert len(report['results']) == 5
    for result in report['results']:
        assert result['history'][-1]['step'] == 2
        assert len(result['validation']['rows']) == 4
        saved = torch.load(output/f"{result['variant']}-seed42.pt", weights_only=True)
        assert all(torch.isfinite(t).all() for t in saved['state_dict'].values())
