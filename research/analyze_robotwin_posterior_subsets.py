"""CPU-only matched subset check for latent and state-delta concentration."""
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
from research.analyze_robotwin_posterior_geometry import SOURCE, digest, spectrum


def covariance(values):
    centered = values - values.mean(0)
    return centered.T @ centered / len(values)


def main():
    torch.set_num_threads(1)
    protocol = json.loads((SOURCE / 'protocol.json').read_text())
    cache_manifest = json.loads((SOURCE / 'feature-cache-manifest.json').read_text())
    cache_path = Path(protocol['cache'])
    cache_sha = digest(cache_path)
    assert cache_sha == cache_manifest['sha256']
    cache = torch.load(cache_path, map_location='cpu', weights_only=True, mmap=True)
    rows = cache['rows']
    assert len(rows) == 512
    seen = set()
    distinct = []
    for i, row in enumerate(rows):
        if row['episode'] not in seen:
            distinct.append(i)
            seen.add(row['episode'])
    assert len(distinct) == 415
    subsets = {'all512': np.arange(len(rows)),
               'first_row_per_episode415': np.asarray(distinct),
               'full_horizon_moving_at_least5cm': np.array([
                   i for i, row in enumerate(rows) if row['full_horizon'] and row['moving_5cm']])}
    assert all(rows[i]['future'] - rows[i]['base'] == 48 and rows[i]['max_arm_displacement_cm'] >= 5
               for i in subsets['full_horizon_moving_at_least5cm'])
    delta = cache['delta'][:, :16].double().numpy()
    xyz_indices = [0, 1, 2, 8, 9, 10]
    codes = {}
    input_hashes = {'cache': cache_sha, 'feature_cache_manifest': digest(SOURCE / 'feature-cache-manifest.json'),
                    'protocol': digest(SOURCE / 'protocol.json')}
    for name in ['official', 'kl_mse']:
        path = SOURCE / name / 'posterior-codes.pt'
        data = torch.load(path, map_location='cpu', weights_only=True)
        assert data['row_identity_sha256'] == cache['reference_identity_sha256']
        codes[name] = {'mu': data['mu'].double().flatten(1).numpy(),
                       'noise': data['logvar'].double().exp().flatten(1).numpy()}
        input_hashes[name] = digest(path)
    reports = {}
    for subset, index in subsets.items():
        label_spectra = {}
        for label, values in [('normalized_delta_state16', delta[index]),
                              ('normalized_delta_xyz6', delta[index][:, xyz_indices])]:
            cov = covariance(values)
            label_spectra[label] = {'variance_trace': float(np.trace(cov)),
                                   **spectrum(np.linalg.eigvalsh(cov))}
        models = {}
        for name, values in codes.items():
            mu, noise = values['mu'][index], values['noise'][index]
            cov = covariance(mu)
            noise_diagonal = noise.mean(0)
            noise_normalized_cov = cov / np.sqrt(noise_diagonal[:, None] * noise_diagonal[None, :])
            eigenvalues = np.linalg.eigvalsh(noise_normalized_cov)
            models[name] = {
                'mu_variance_trace': float(np.trace(cov)),
                'noise_variance_trace': float(noise_diagonal.sum()),
                'mu_covariance': spectrum(np.linalg.eigvalsh(cov)),
                'noise_normalized_signal_covariance': {
                    **spectrum(eigenvalues),
                    'directions_signal_to_noise_above_one': int(np.sum(eigenvalues > 1)),
                    'directions_signal_to_noise_above_point_one': int(np.sum(eigenvalues > .1))}}
        reports[subset] = {'samples': len(index), 'episodes': len({rows[i]['episode'] for i in index}),
                           'row_indices': index.tolist(), 'models': models, 'label_covariance': label_spectra}
    assert not torch.cuda.is_initialized()
    result = {'source': str(SOURCE), 'script_sha256': digest(__file__), 'input_sha256': input_hashes,
              'row_identity_sha256': cache['reference_identity_sha256'], 'subsets': reports,
              'method': {
                  'first_row_per_episode': 'First row in the existing fixed evaluator order, before examining any codes or labels.',
                  'full_horizon_moving': '48-frame horizon and at least one end effector displaced by >=5cm, using the evaluator metadata.',
                  'covariance': 'Population covariance with divisor subset sample count; CPU float64 arithmetic.',
                  'noise_normalization': 'Eigenvalues of N^(-1/2) Cov(mu) N^(-1/2), N=diag(mean(exp(logvar))) within the same subset.',
                  'labels': 'Upstream-normalized future-minus-current 16D state, or its 6 XYZ components; no whitening of label coordinates.',
                  'scope': 'Effective variance concentration, not mathematical rank or mutual information. Different episodes are not proof of semantic diversity; state-label spectra provide an additional measured check.',
                  'input_limit': 'State deltas are measured; raw image/feature covariance is not computed here.',
                  'prediction_limit': 'Posterior geometry cannot alone establish collapse of all decoder future predictions or the training-history cause.'},
              'cuda_initialized': False}
    output = ROOT / 'research/results/robotwin-posterior-subset-audit-20261114.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name, subset in reports.items():
        concise = {'subset': name, 'samples': subset['samples'], 'episodes': subset['episodes'],
                   'models': {model: {'mu_top_fraction': m['mu_covariance']['top_fraction'],
                                      'mu_entropy_effective_rank': m['mu_covariance']['entropy_effective_rank'],
                                      'snr_directions_above_one': m['noise_normalized_signal_covariance']['directions_signal_to_noise_above_one']}
                              for model, m in subset['models'].items()},
                   'labels': {label: {'top_fraction': m['top_fraction'], 'entropy_effective_rank': m['entropy_effective_rank'],
                                      'dimensions_for_90percent_variance': m['dimensions_for_variance_fraction']['0.9']}
                              for label, m in subset['label_covariance'].items()}}
        print(json.dumps(concise), flush=True)
    print(output, flush=True)


if __name__ == '__main__':
    main()
