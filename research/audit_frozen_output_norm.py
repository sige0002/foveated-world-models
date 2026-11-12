"""Certified convex-relaxation error floor for a fixed affine output LayerNorm.

LayerNorm outputs gamma*u+beta with sum(u)=0 and ||u||^2 < D.
Relax to the closed ball. For v=y-beta, the Lagrangian uses lambda>=0
and 2*eta*sum(u). Its dual is ||v||^2-lambda*D-
sum((gamma*v-eta)^2/(gamma^2+lambda)). Every dual value is a lower bound.
Targets enter this oracle diagnostic only, never a deployed predictor.
"""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(2**20), b''):
            h.update(b)
    return h.hexdigest()


def project(y, gamma, beta):
    v = y.double() - beta
    d = v.shape[-1]
    g2 = gamma.square()
    gv = gamma * v

    def solve(lam):
        inv = (g2 + lam).reciprocal()
        eta = (gv * inv).sum(-1, keepdim=True) / inv.sum(-1, keepdim=True)
        u = (gv - eta) * inv
        return u, eta

    low = torch.zeros_like(v[..., :1])
    u0, _ = solve(low)
    active = u0.square().sum(-1, keepdim=True) > d
    high = torch.ones_like(low)
    for _ in range(60):
        u, _ = solve(high)
        outside = u.square().sum(-1, keepdim=True) > d
        if not outside.any():
            break
        high = torch.where(outside, high * 2, high)
    else:
        raise RuntimeError('Failed to bracket norm multiplier')
    for _ in range(45):
        mid = (low + high) * .5
        u, _ = solve(mid)
        outside = u.square().sum(-1, keepdim=True) > d
        low = torch.where(outside, mid, low)
        high = torch.where(outside, high, mid)
    lam = torch.where(active, high, torch.zeros_like(high))
    u, eta = solve(lam)
    primal = (gamma * u - v).square().mean(-1)
    dual = (v.square().sum(-1) - lam.squeeze(-1) * d
            - ((gv - eta).square() / (g2 + lam)).sum(-1)) / d
    # Explicit primal feasibility and primal-dual gap verify the numerical solver.
    assert u.sum(-1).abs().max() < 1e-8
    assert u.square().sum(-1).max() <= d + 1e-8
    assert (primal - dual).abs().max() < 1e-8
    hyperplane = (v / gamma).sum(-1).square() / ((1 / gamma).square().sum() * d)
    assert (dual - hyperplane).min() >= -1e-8
    return {'dual_lower_bound': dual, 'primal_relaxation_error': primal,
            'hyperplane_lower_bound': hyperplane, 'norm_constraint_active': active.squeeze(-1).double(),
            'duality_gap': primal - dual}


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    out = ROOT / 'research/results/frozen-output-norm-audit-20261113.json'
    if out.exists():
        raise FileExistsError(out)
    weights = ROOT / 'runs/frozen-model-step10000/reference_weights.pt'
    saved = torch.load(weights, weights_only=True, mmap=True, map_location='cpu')
    assert saved['global_step'] == 10000
    state = saved['state_dict']
    gamma = state['lam.decoder.last_ln.weight'].double()
    beta = state['lam.decoder.last_ln.bias'].double()
    assert gamma.abs().min() > 1e-6
    sources = {str(weights.relative_to(ROOT)): sha(weights)}
    report = {'source_sha256': sources, 'script_sha256': sha(Path(__file__)),
              'gamma_min_abs': float(gamma.abs().min()), 'splits': {},
              'scope': 'Oracle geometric diagnostic for frozen final LayerNorm. Convex closed-ball relaxation gives an error lower bound, not an achievable predictor or forecasting metric. True targets used only for analysis. Reused evaluation data.'}
    for split, rel, disk in [
        ('train', 'runs/reference-idm42-nested512-prior10000', 'train'),
        ('validation', 'runs/reference-idm42-nested512-prior10000', 'validation'),
        ('observed96', 'runs/reference-idm42-observed96-prior10000', 'validation')]:
        root = ROOT / rel
        path = root / (disk + '.pt')
        manifest = json.loads((root / 'cache_manifest.json').read_text())
        digest = sha(path)
        assert digest == manifest['cache_sha256'][disk]
        assert sources[str(weights.relative_to(ROOT))] == manifest['prior_weights_sha256']
        sources[str(path.relative_to(ROOT))] = digest
        data = torch.load(path, weights_only=True, map_location='cpu')
        rows = []
        maximum_gap = 0.
        for start in range(0, len(data['rows']), 8):
            y = data['future'][start:start+8].squeeze(1)
            result = project(y, gamma, beta)
            maximum_gap = max(maximum_gap, float(result['duality_gap'].abs().max()))
            metrics = {k: v.mean(-1) for k, v in result.items()}
            prior = (data['baseline'][start:start+len(y)].squeeze(1) - y).double().square().mean((1, 2))
            assert (metrics['dual_lower_bound'] - prior).max() < 1e-7
            for j in range(len(y)):
                rows.append({'source_row': data['rows'][start+j], 'prior_mse': float(prior[j]),
                             **{k: float(v[j]) for k, v in metrics.items()}})
        means = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0] if k != 'source_row'}
        report['splits'][split] = {'metrics': means, 'max_token_duality_gap': maximum_gap, 'rows': rows}
        print(split, json.dumps(means), flush=True)
    out.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
