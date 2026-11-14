"""CPU-only spectra of frozen bottleneck weights; no model or optimizer runs.

Weight spectra do not measure sample information or identify training history.
They test whether a nearly rank-one linear map alone explains the mu geometry.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import torch

ROOT = Path(__file__).resolve().parents[1]
SSD = Path('/media/sadasue/SSD1/foveated-world-models')
RUNS = SSD / 'runs'
CHECKPOINTS = {
    'official': SSD / 'models/lawam-lam-bd993da/checkpoints/pytorch_model.pt',
    'combined3000': RUNS / 'robotwin-loss-ablation-b4-20261114/kl_mse/checkpoints/final.ckpt',
    'baseline3000': RUNS / 'robotwin-loss-ablation-b4-20261114/baseline/checkpoints/final.ckpt',
    'old10000': RUNS / 'robotwin-qfix-delta2-b4-20261114/checkpoints/last.ckpt',
}


def spectrum(matrix):
    singular_values = torch.linalg.svdvals(matrix.double())
    squared = singular_values.square()
    fractions = squared / squared.sum()
    return {
        'shape': list(matrix.shape),
        'singular_values': singular_values.tolist(),
        'top_squared_singular_fraction': float(fractions[0]),
        'participation_ratio': float(1 / fractions.square().sum()),
        'entropy_effective_rank': float(torch.exp(-(fractions * fractions.clamp_min(1e-300).log()).sum())),
        'frobenius_norm': float(squared.sum().sqrt()),
    }


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    reports = {}
    for name, path in CHECKPOINTS.items():
        before = path.stat()
        saved = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        state = saved['state_dict']
        selected = {key: state[key].detach().double().clone() for key in (
            'lam.encoder.out_proj.weight', 'lam.encoder.out_proj.bias',
            'lam.vq.pre_norm.weight', 'lam.vq.pre_norm.bias',
            'lam.vq.mu.weight', 'lam.vq.mu.bias',
            'lam.vq.logvar.weight', 'lam.vq.logvar.bias',
        )}
        digest = hashlib.sha256()
        for key in sorted(selected):
            digest.update(key.encode() + b'\0')
            digest.update(selected[key].contiguous().numpy().tobytes())
        out = selected['lam.encoder.out_proj.weight']
        mu = selected['lam.vq.mu.weight']
        lv = selected['lam.vq.logvar.weight']
        gain = selected['lam.vq.pre_norm.weight']
        # LayerNorm centers over coordinates before its learned affine map.
        # Its sample-dependent denominator is intentionally NOT modeled here.
        projector = torch.eye(len(gain), dtype=torch.float64) - torch.ones(len(gain), len(gain), dtype=torch.float64) / len(gain)
        maps = {
            'encoder_out_proj': out,
            'encoder_out_proj_coordinate_centered': projector @ out,
            'mu_weight': mu,
            'logvar_weight': lv,
            'mu_after_layernorm_affine_and_centering': mu @ torch.diag(gain) @ projector,
        }
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
        reports[name] = {
            'checkpoint': str(path), 'checkpoint_size': before.st_size,
            'checkpoint_mtime_ns': before.st_mtime_ns,
            'selected_float64_tensors_sha256': digest.hexdigest(),
            'spectra': {key: spectrum(value) for key, value in maps.items()},
            'layernorm_gain_min': float(gain.min()), 'layernorm_gain_max': float(gain.max()),
            'mu_bias_norm': float(selected['lam.vq.mu.bias'].norm()),
        }
        del saved, state, selected
    result = {
        'complete': True, 'cuda_initialized': torch.cuda.is_initialized(),
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'models': reports,
        'limits': 'Spectra of fixed linear weights are not sample covariance spectra. Full-rank weights can receive collapsed inputs; a dominant singular direction need not dominate outputs on the actual data. LayerNorm is nonlinear; its input-dependent denominator is excluded from the composed linear map. No training-history causality is established.',
    }
    output = ROOT / 'research/results/robotwin-bottleneck-weight-spectra-20261114.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'output': str(output), 'cuda_initialized': result['cuda_initialized'],
        'summary': {name: {key: {'top': value['top_squared_singular_fraction'], 'PR': value['participation_ratio']} for key, value in report['spectra'].items()} for name, report in reports.items()}}))


if __name__ == '__main__':
    main()
