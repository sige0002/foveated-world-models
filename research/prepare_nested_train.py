"""Select 384 additional training episodes using metadata only, retaining train128."""
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor/lawam')]
import numpy as np
import torch
import yaml
from research.cache_stage1_features import sha256
from vendor.dataset.robomind import RoboMINDDataset, check_disjoint


def main():
    output = ROOT / 'research/results/nested-train512-selection-20261113.json'
    if output.exists():
        raise FileExistsError(output)
    parent = ROOT / 'runs/reference-idm42-with-prior10000'
    manifest = json.loads((parent / 'cache_manifest.json').read_text())
    cached = {}
    for split in ('train', 'validation'):
        assert sha256(parent / f'{split}.pt') == manifest['cache_sha256'][split]
        cached[split] = torch.load(parent / f'{split}.pt', weights_only=True)
    cfg = yaml.safe_load((ROOT / 'runs/frozen-idm-step2900/resolved_config.yaml').read_text())
    options = dict(image_hw=cfg['model']['image_hw'], max_state_dim=cfg['model']['max_state_dim'],
                   num_embodiments=cfg['model']['num_embodiments'],
                   horizon_seconds=cfg['data']['horizon_seconds'],
                   tolerance_seconds=cfg['data']['tolerance_seconds'], require_observed_endpoints=True)
    train = RoboMINDDataset(cfg['data']['train_manifest'], **options)
    validation = RoboMINDDataset(cfg['data']['val_manifest'], **options)
    check_disjoint(train, validation)
    retained = cached['train']['rows']
    old_episodes = {r['episode'] for r in retained}
    assert len(retained) == len(old_episodes) == 128
    groups = {}
    for index, (i, first, last, _) in enumerate(train.pairs):
        episode = str(train.rows[i]['episode_id'])
        if episode not in old_episodes:
            groups.setdefault(episode, []).append(index)
    rng = np.random.default_rng(20261113)
    episodes = sorted(groups)
    chosen = rng.choice(episodes, 384, replace=False).tolist()
    added = []
    for episode in chosen:
        index = int(rng.choice(groups[episode]))
        i, first, last, _ = train.pairs[index]
        arrays = train._arrays(i)
        added.append(dict(index=index, episode=episode, endpoint_indices=[first, last],
                          dt=float(arrays['timestamps'][last]-arrays['timestamps'][first])))
    for row in retained + added:
        i, first, last, _ = train.pairs[row['index']]
        arrays = train._arrays(i)
        assert str(train.rows[i]['episode_id']) == row['episode']
        assert abs(float(arrays['timestamps'][last]-arrays['timestamps'][first])-row['dt']) < 1e-9
        assert arrays['original_valid'][[first,last]].all()
    assert len({r['episode'] for r in retained+added}) == 512
    assert not {r['episode'] for r in retained+added} & {str(e) for e in validation.episodes}
    report = dict(seed=20261113, parent=str(parent), parent_cache_sha256=manifest['cache_sha256'],
                  manifest=str(train.path), manifest_sha256=sha256(train.path), options=options,
                  validation_manifest=str(validation.path), validation_manifest_sha256=sha256(validation.path),
                  retained=retained, added=added, candidate_new_episodes=len(episodes),
                  rule='Preserve old128 exactly; choose384 distinct remaining train episodes uniformly, then one observed-endpoint pair uniformly per episode. Metadata only; no images/errors inspected.',
                  scope='Nested training expansion only. All160 main validation episodes have already been research-observed; reuse64/96 transparently, no fresh-test claim.',
                  extraction='Frozen encoder2900, fixed reference head42 and its original train128 calibration; final10000 current-only baseline. No head refit in first scaling comparison.',
                  comparison='Same correction architecture/seed42. Fixed400 updates for equal step budget and fixed1600 for equal expected exposures; report both, do not select evaluation-best.',
                  script_sha256=sha256(Path(__file__)))
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(output, '128 retained +384 additional training episodes; no image decoding')


if __name__ == '__main__':
    main()
