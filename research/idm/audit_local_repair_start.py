"""Read-only, independent starting-point audit for local head/decoder repair."""
import hashlib
import json
from pathlib import Path
import sys
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.idm.train_reference_heads import representation
from research.train_cached_decoders import make_decoder


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    cache = ROOT / 'runs/reference-idm42-nested512-prior10000'
    obs = ROOT / 'runs/reference-idm42-observed96-prior10000'
    paths = dict(train=cache/'train.pt', dev64=cache/'validation.pt', observed96=obs/'validation.pt')
    data = {k: torch.load(p, weights_only=True, mmap=True) for k, p in paths.items()}
    sources = {str(p.relative_to(ROOT)): sha(p) for p in paths.values()}
    manifests = {k: json.loads((p.parent/'cache_manifest.json').read_text()) for k, p in paths.items()}
    for k, p in paths.items():
        assert sources[str(p.relative_to(ROOT))] == manifests[k]['cache_sha256'][p.stem]
        for name in ('prior_checkpoint_sha256', 'prior_weights_sha256'):
            assert manifests[k][name] == manifests['train'][name]
    episodes = [{r['episode'] for r in d['rows']} for d in data.values()]
    assert all(episodes[i].isdisjoint(episodes[j]) for i in range(3) for j in range(i+1, 3))
    qp = dict(old=ROOT/'research/idm/reference-query-full/train.pt',
              added=cache/'additional_queries.pt',
              dev64=ROOT/'research/idm/reference-query-full/validation.pt',
              observed96=ROOT/'research/idm/observed96-reference-queries2900/validation.pt')
    q = {k: torch.load(p, weights_only=True) for k, p in qp.items()}
    q['train'] = {k: torch.cat([q['old'][k], q['added'][k]]) for k in ('query', 'reference')}
    q['train']['rows'] = q['old']['rows'] + q['added']['rows']
    hp = ROOT/'research/idm/reference-comparison2000/reference-42.pt'
    cp = ROOT/'research/idm/reference-query-full/train_calibration.pt'
    dp = ROOT/'runs/reference-idm42-nested512-comparison400-20261113/prior_residual-seed42.pt'
    for p in [*qp.values(), hp, cp, dp]:
        sources[str(p.relative_to(ROOT))] = sha(p)
    assert sha(hp) == manifests['train']['export']['head_checkpoint_sha256']
    assert sha(cp) == manifests['train']['export']['calibration_sha256']
    hs = torch.load(hp, weights_only=True)
    ds = torch.load(dp, weights_only=True)
    assert hs['steps'] == 2000 and hs['variant'] == 'reference' and hs['seed'] == 42
    assert ds['steps'] == 400 and ds['variant'] == 'prior_residual' and ds['seed'] == 42
    head = nn.Sequential(nn.Linear(1024,256), nn.GELU(), nn.Linear(256,64)).eval()
    head.load_state_dict({k[5:]: v for k,v in hs['state_dict'].items() if k.startswith('head.')}, strict=True)
    decoder = make_decoder('prior_residual', ds['dim'], ds['layers'], 768,32,16).eval()
    decoder.load_state_dict(ds['state_dict'], strict=True)
    cal = torch.load(cp, weights_only=True)
    codes = {}; variances = {}; checks = {}
    identity = lambda rs: [(r['episode'],r['index'],r['dt'],r.get('endpoint_indices')) for r in rs]
    for name, d in data.items():
        assert identity(q[name]['rows']) == identity(d['rows'])
        x = representation(q[name], cal, 'reference')
        post = torch.cat([head(v) for v in x.split(8)])
        mu, lv = post.chunk(2, -1); lv = lv.clamp(-10, 10)
        assert torch.isfinite(post).all()
        codes[name] = mu; variances[name] = lv.exp()
        checks[name] = {'rows': len(d['rows']), 'cached_mu_max_difference': float((mu-d['codes']).abs().max()),
                        'reference_variance_min': float(lv.exp().min()), 'reference_variance_max': float(lv.exp().max())}
        for key in ('current','future','baseline'):
            assert d[key].device.type == 'cpu' and torch.isfinite(d[key]).all()
    constant = codes['train'].mean(0, keepdim=True)
    for name, d in data.items():
        rows = []
        for start in range(0, len(d['rows']), 8):
            h,b,y = [d[k][start:start+8] for k in ('current','baseline','future')]
            inputs = {'F': codes[name][start:start+8], 'K': constant.expand(len(h), *constant.shape[1:]),
                      'cached': d['codes'][start:start+8]}
            errors = {k: (decoder(h,z,baseline=b)-y).square().flatten(1).mean(1) for k,z in inputs.items()}
            rows += [dict(episode=d['rows'][start+i]['episode'], index=d['rows'][start+i]['index'],
                          **{k: float(v[i]) for k,v in errors.items()}) for i in range(len(h))]
        checks[name]['initial_mse'] = {k: sum(r[k] for r in rows)/len(rows) for k in errors}
        checks[name]['rows_detail'] = rows
        print(name, checks[name]['initial_mse'], flush=True)
    report = {'precision': 'CPU2 float32 highest, B8', 'source_sha256':sources,
              'checks':checks, 'calibration':'Unchanged original train128 scalar; no evaluation fitting',
              'all_episode_sets_disjoint':True, 'script_sha256':sha(Path(__file__)),
              'scope':'Read-only preflight; no optimization, no selection by initial risks. Posterior reconstruction.'}
    out = ROOT/'research/idm/local-repair-start-audit-20261113.json'
    with out.open('x') as f:
        json.dump(report, f, indent=2); f.write('\n')


if __name__ == '__main__':
    main()
