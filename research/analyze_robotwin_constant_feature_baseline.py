"""CPU oracle lower bound for a single constant future-feature prediction."""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from research.analyze_robotwin_posterior_geometry import SOURCE, digest


def main():
    torch.set_num_threads(1)
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    manifest = json.loads((SOURCE / 'feature-cache-manifest.json').read_text())
    path = Path(protocol['cache'])
    cache_hash = digest(path)
    assert cache_hash == manifest['sha256']
    cache = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    targets = cache['y']
    assert targets.shape[0] == 512 and targets.dtype == torch.float32
    chunk_size = 8
    mean = torch.zeros_like(targets[0], dtype=torch.float64)
    for start in range(0, len(targets), chunk_size):
        mean += targets[start:start + chunk_size].double().sum(0)
    mean /= len(targets)
    errors = []
    for start in range(0, len(targets), chunk_size):
        block = targets[start:start + chunk_size].double() - mean
        errors.extend(block.square().flatten(1).mean(1).tolist())
    constant_mse = float(np.mean(errors))
    comparisons = {}
    hashes = {'cache': cache_hash, 'script': digest(__file__)}
    identity = [(row['episode'], row['base']) for row in cache['rows']]
    for name in ['official', 'kl_mse']:
        report_path = SOURCE / name / 'results.json'
        report = json.loads(report_path.read_text())
        assert [(row['episode'], row['base']) for row in report['rows']] == identity
        mse = float(np.mean([row['metrics']['real_visual_mse'] for row in report['rows']]))
        comparisons[name] = {'real_visual_mse': mse,
                             'oracle_constant_minus_real_mse': constant_mse - mse,
                             'relative_reduction_from_oracle_constant': (constant_mse - mse) / constant_mse,
                             'strictly_better_than_any_single_constant_on_these_rows': mse < constant_mse}
        hashes[name + '_results'] = digest(report_path)
    assert not torch.cuda.is_initialized()
    result = {'cache': str(path), 'input_sha256': hashes,
              'row_identity_sha256': cache['reference_identity_sha256'],
              'samples': len(targets), 'target_shape': list(targets.shape), 'chunk_size': chunk_size,
              'oracle_constant_output_mse': constant_mse, 'comparisons': comparisons,
              'method': 'Min_c mean_i mean_coordinates((y_i-c)^2); c=mean_i(y_i), retaining the full spatial/channel shape. Two-pass chunked CPU float64; source tensors float32 via mmap.',
              'interpretation': 'The constant is fitted to all evaluation targets, so it is an optimistic oracle lower bound for any single constant output on these exact rows. It is not a held-out trained baseline.',
              'limits': 'Outperforming this lower bound excludes a fully identical future-feature output across these rows, but does not prove adequate motion/action prediction or exclude latent representation collapse. The decoder can use current-image features independently of z.',
              'cuda_initialized': False}
    output = ROOT / 'research/results/robotwin-constant-feature-baseline-20261114.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
