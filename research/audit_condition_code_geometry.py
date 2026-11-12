"""Train-only geometry of the fixed reference code; no fitting or code changes."""
import hashlib
import json
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def spectrum(matrix):
    eigenvalues = torch.linalg.eigvalsh(matrix).flip(0)
    assert eigenvalues.min() > -1e-10
    eigenvalues = eigenvalues.clamp_min(0)
    p = eigenvalues / eigenvalues.sum()
    positive = eigenvalues[eigenvalues > eigenvalues.max() * 1e-10]
    return {'eigenvalues_descending': eigenvalues.tolist(),
            'numerical_rank_relative_1e_minus10': len(positive),
            'positive_subspace_condition': float(positive.max() / positive.min()),
            'entropy_effective_rank': float(torch.exp(-(p * p.clamp_min(1e-300).log()).sum())),
            'participation_ratio': float(eigenvalues.sum().square() / eigenvalues.square().sum()),
            'trace_fractions': {str(k): float(p[:k].sum()) for k in (1, 2, 4, 8, 16, 32)}}


def main():
    torch.set_num_threads(1)
    root = ROOT / 'runs/reference-idm42-nested512-prior10000'
    path = root / 'train.pt'
    manifest = json.loads((root / 'cache_manifest.json').read_text())
    source_hash = digest(path)
    assert source_hash == manifest['cache_sha256']['train']
    data = torch.load(path, weights_only=True, map_location='cpu')
    z = data['codes'].reshape(512, 32).double()
    assert torch.isfinite(z).all()
    mean = z.mean(0)
    centered = z - mean
    gram = z.T @ z / len(z)
    covariance = centered.T @ centered / len(z)
    result = {'train_cache_sha256': source_hash, 'script_sha256': digest(Path(__file__)),
              'count': len(z), 'mean_norm': float(mean.norm()),
              'code_rms_norm': float(z.square().sum(-1).mean().sqrt()),
              'mean_energy_fraction': float(mean.square().sum() / z.square().sum(-1).mean()),
              'uncentered_second_moment': spectrum(gram), 'centered_covariance': spectrum(covariance),
              'scope': 'Fixed reference42 codes on train512 only. Code anisotropy is not the decoder Hessian condition number, semantic dimensionality, or an information estimate. No whitening or experiment change.'}
    out = ROOT / 'research/results/condition-code-geometry-20261113.json'
    if out.exists():
        raise FileExistsError(out)
    out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('uncentered_second_moment', 'centered_covariance')}, indent=2))
    for key in ('uncentered_second_moment', 'centered_covariance'):
        print(key, json.dumps({k: v for k, v in result[key].items() if k != 'eigenvalues_descending'}))


if __name__ == '__main__':
    main()
