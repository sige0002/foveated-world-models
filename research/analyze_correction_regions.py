"""Fixed feature-change strata; descriptive evaluation, not physical motion labels."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from research.train_cached_decoders import file_hash, make_decoder


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('New output required')
    torch.set_num_threads(1)
    manifest = json.loads((a.cache/'cache_manifest.json').read_text())
    assert file_hash(a.cache/'validation.pt') == manifest['cache_sha256']['validation']
    data = torch.load(a.cache/'validation.pt', weights_only=True)
    h, y = data['current'], data['future']
    change = (h-y).square().mean(-1).squeeze(1)
    order = change.argsort(dim=-1, stable=True)
    n = change.shape[-1]
    strata = {'all': order, 'low_change_half': order[:, :n//2],
              'high_change_decile': order[:, -math.ceil(n*.1):]}
    def per_region(prediction):
        error = (prediction-y).square().mean(-1).squeeze(1)
        return {name: error.gather(1, idx).mean(1).tolist() for name, idx in strata.items()}
    prior = per_region(data['baseline'])
    results = []
    for path in a.checkpoints:
        saved = torch.load(path, weights_only=True)
        assert saved['variant'] == 'prior_residual' and saved['steps'] == 400
        model = make_decoder(saved['variant'], saved['dim'], saved['layers'],
                             h.shape[-1], data['codes'].shape[-1], math.isqrt(n)).eval()
        model.load_state_dict(saved['state_dict'], strict=True)
        predictions = []
        for start in range(0, len(h), 8):
            sl = slice(start, start+8)
            predictions.append(model(h[sl], data['codes'][sl], baseline=data['baseline'][sl]))
        values = per_region(torch.cat(predictions))
        results.append(dict(seed=saved['seed'], checkpoint=str(path),
                            checkpoint_sha256=file_hash(path), per_episode=values))
    summary = {}
    for name in strata:
        errors = torch.tensor([r['per_episode'][name] for r in results], dtype=torch.float64)
        baseline = torch.tensor(prior[name], dtype=torch.float64)
        summary[name] = dict(direct_mse=float(errors.mean()), prior_mse=float(baseline.mean()),
                             relative_reduction=float(1-errors.mean()/baseline.mean()),
                             per_seed_mse=errors.mean(1).tolist())
    report = dict(cache_sha256=manifest['cache_sha256'], script_sha256=file_hash(Path(__file__)),
                  strata='Per-example ranking by true current-to-future DINO squared change; low50% and high10% (ceil); ties stable by patch index.',
                  scope='Descriptive, already-observed research validation; feature change includes camera/lighting and is not an action label. No region labels enter prediction. True future still enters posterior codes. Baseline is zero-code prior, not original posterior decoder.',
                  episodes=[r['episode'] for r in data['rows']], prior_per_episode=prior,
                  summary=summary, results=results)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
