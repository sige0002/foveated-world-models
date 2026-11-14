"""CPU-only second-moment audit of the common FP32 posterior-code cache."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path('/media/sadasue/SSD1/foveated-world-models/runs/robotwin-loss-ablation-b4-20261114/evaluations/final3000-highest-fp32-v2')
OUTPUT = ROOT / 'research/results/robotwin-posterior-direction-audit-20261114.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    quantiles = [0, .01, .05, .25, .5, .75, .95, .99, 1]
    return {'mean': float(values.mean()), 'std': float(values.std()),
            'quantiles': {str(q): float(np.quantile(values, q)) for q in quantiles},
            'largest_share_of_total': {
                str(f): {'samples': int(np.ceil(len(values) * f)),
                         'share': float(np.sort(values)[-int(np.ceil(len(values) * f)):].sum() / values.sum())}
                for f in [.01, .05, .1]}}


def spectrum(values):
    values = np.maximum(np.asarray(values)[::-1], 0)
    proportions = values / values.sum()
    return {'eigenvalues_descending': values.tolist(),
            'top_fraction': float(proportions[0]),
            'entropy_effective_rank': float(np.exp(-np.sum(proportions * np.log(np.maximum(proportions, 1e-300))))),
            'participation_ratio': float(1 / np.sum(proportions ** 2)),
            'dimensions_for_variance_fraction': {str(q): int(np.searchsorted(proportions.cumsum(), q) + 1)
                                                for q in [.9, .95, .99, .999]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--models', nargs='+', default=['official', 'kl_mse', 'baseline', 'kl_only', 'mse_only'])
    parser.add_argument('--skip-label-analysis', action='store_true')
    args = parser.parse_args()
    source = args.source.resolve()
    torch.set_num_threads(1)
    reports, hashes, identities = {}, {}, set()
    label_analysis = None
    labels = None
    groups = {'xyz': [0, 1, 2, 8, 9, 10], 'quaternion': [3, 4, 5, 6, 11, 12, 13, 14],
              'grippers': [7, 15], 'left_gripper': [7], 'right_gripper': [15]}
    state_names = [side + '_' + component for side in ['left', 'right']
                   for component in ['x', 'y', 'z', 'quaternion_0', 'quaternion_1', 'quaternion_2', 'quaternion_3', 'gripper']]
    if not args.skip_label_analysis:
        protocol = json.loads((source / 'protocol.json').read_text())
        cache_manifest = json.loads((source / 'feature-cache-manifest.json').read_text())
        cache_path = Path(protocol['cache'])
        cache_hash = digest(cache_path)
        assert cache_hash == cache_manifest['sha256']
        cache = torch.load(cache_path, map_location='cpu', weights_only=True, mmap=True)
        assert torch.equal(cache['delta'], cache['states'][:, 1] - cache['states'][:, 0])
        old_path = Path('/media/sadasue/SSD1/foveated-world-models/runs/robotwin-qfix-delta2-b4-20261114/evaluations/idm-state-probe-20261114/codes.pt')
        old = torch.load(old_path, map_location='cpu', weights_only=True, mmap=True)
        expected_rows = [{'episode': row['episode'], 'base': row['base']} for row in cache['rows']]
        assert old['val']['rows'] == expected_rows
        assert torch.equal(old['val']['state'], cache['states'][:, 0, :16])
        assert torch.equal(old['val']['delta'], cache['delta'][:, :16])
        labels = {'current_state': cache['states'][:, 0, :16].double().numpy(),
                  'future_state': cache['states'][:, 1, :16].double().numpy(),
                  'delta': cache['delta'][:, :16].double().numpy()}
        target_scales = {}
        for part, tensor in [('val512', cache['delta'][:, :16]), ('train2048', old['train']['delta'])]:
            array = tensor.double().numpy()
            variance = np.var(array, axis=0)
            second_moment = np.mean(array ** 2, axis=0)
            target_scales[part] = {
                'samples': len(array), 'per_coordinate_variance': variance.tolist(),
                'per_coordinate_zero_predictor_mse': second_moment.tolist(),
                'group_variance_share': {key: float(variance[idx].sum() / variance.sum()) for key, idx in groups.items()},
                'group_zero_predictor_mse_share': {key: float(second_moment[idx].sum() / second_moment.sum()) for key, idx in groups.items()},
                'mean_predictor_total_mse16': float(variance.mean()),
                'zero_predictor_total_mse16': float(second_moment.mean())}
        label_analysis = {'cache_path': str(cache_path), 'cache_sha256': cache_hash,
                          'old_label_source': str(old_path), 'old_label_source_sha256': digest(old_path),
                          'old_codes_used': 'Only exact matching state/delta labels and row identities; no old latent codes or predictions.',
                          'row_identity_sha256': cache['reference_identity_sha256'],
                          'coordinate_names': state_names, 'group_indices': groups,
                          'target_delta_scale': target_scales,
                          'scope': 'Descriptive same-sample correlations, not a held-out probe. Label variance shares are not actual model-loss or gradient shares.'}
        del cache, old

    def correlation_report(score, target):
        x = score - score.mean()
        y = target - target.mean(0)
        cov = np.mean(x[:, None] * y, axis=0)
        xvar, yvar = np.mean(x ** 2), np.mean(y ** 2, axis=0)
        correlation = cov / np.sqrt(np.maximum(xvar * yvar, 1e-300))
        return {'pearson_per_coordinate': correlation.tolist(), 'r_squared_per_coordinate': (correlation ** 2).tolist(),
                'group_variance_weighted_r_squared': {
                    key: float(np.sum(cov[idx] ** 2 / xvar) / np.sum(yvar[idx])) for key, idx in groups.items()}}

    for name in args.models:
        path = source / name / 'posterior-codes.pt'
        data = torch.load(path, map_location='cpu', weights_only=True)
        identities.add(data['row_identity_sha256'])
        mu = data['mu'].double().flatten(1).numpy()
        current = data['current_only_mu'].double().flatten(1).numpy()
        noise = data['logvar'].double().exp().flatten(1).numpy()
        assert mu.shape == current.shape == noise.shape == (512, 32)
        centered = mu - mu.mean(0)
        delta = mu - current
        covariance = centered.T @ centered / len(mu)
        eigenvalues, directions = np.linalg.eigh(covariance)
        trace_signal = np.trace(covariance)
        noise_per_sample = noise.sum(1)
        noise_diagonal = noise.mean(0)
        whitened_mu = centered / np.sqrt(noise_diagonal)
        sn_covariance = whitened_mu.T @ whitened_mu / len(mu)
        sn_eigenvalues = np.linalg.eigvalsh(sn_covariance)
        displacement_coefficients = delta @ directions
        delta_squared = np.sum(delta ** 2, axis=1)
        # Full covariance has an almost-null direction from LayerNorm's affine
        # constraint. Show threshold sensitivity instead of inverting it blindly.
        covariance_normalizations = {}
        for relative_cutoff in [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]:
            keep = eigenvalues > eigenvalues[-1] * relative_cutoff
            retained = int(keep.sum())
            maha_squared = np.sum(displacement_coefficients[:, keep] ** 2 / eigenvalues[keep], axis=1)
            covariance_normalizations[str(relative_cutoff)] = {
                'retained_dimensions': retained,
                'displacement_rms_in_covariance_units_per_retained_dimension': float(np.sqrt(maha_squared.mean() / retained)),
                'displacement_energy_fraction_retained': float(np.sum(displacement_coefficients[:, keep] ** 2) / np.sum(delta_squared)),
                'mahalanobis_norm_over_sqrt_retained_dimensions': distribution(np.sqrt(maha_squared / retained))}
        pc_projections = {}
        for fraction in [.9, .95, .99, .999]:
            k = int(np.searchsorted(np.maximum(eigenvalues[::-1], 0).cumsum() / trace_signal, fraction) + 1)
            basis = directions[:, -k:]
            projected_noise = noise @ np.sum(basis ** 2, axis=1)
            pc_projections[str(fraction)] = {
                'dimensions': k,
                'signal_variance': float(eigenvalues[-k:].sum()),
                'noise_variance_mean': float(projected_noise.mean()),
                'fraction_of_total_noise_in_subspace': float(projected_noise.mean() / noise_per_sample.mean()),
                'signal_to_noise_in_subspace': float(eigenvalues[-k:].sum() / projected_noise.mean())}
        result_path = source / name / 'results.json'
        decoder = {}
        if result_path.is_file():
            result = json.loads(result_path.read_text())
            row_metrics = [row['metrics'] for row in result['rows']]
            for control in ['mean', 'swapped', 'zero', 'current_only']:
                errors = np.array([r[control + '_visual_mse'] - r['real_visual_mse'] for r in row_metrics])
                decoder[control + '_minus_real_visual_mse'] = float(errors.mean())
        reports[name] = {
            'mu_variance_trace': float(trace_signal),
            'posterior_noise_trace_mean': float(noise_per_sample.mean()),
            'global_trace_noise_fraction': float(noise_per_sample.mean() / (noise_per_sample.mean() + trace_signal)),
            'noise_trace_per_sample': distribution(noise_per_sample),
            'centered_mu_norm': distribution(np.linalg.norm(centered, axis=1)),
            'mu_covariance': spectrum(eigenvalues),
            'noise_normalized_signal_covariance': {
                **spectrum(sn_eigenvalues),
                'directions_signal_to_noise_above_one': int(np.sum(sn_eigenvalues > 1)),
                'directions_signal_to_noise_above_point_one': int(np.sum(sn_eigenvalues > .1)),
                'definition': 'Eigenvalues of N^(-1/2) C N^(-1/2), C=Cov_population(mu), N=diag(mean(exp(logvar))).'},
            'mu_principal_subspace_noise': pc_projections,
            'current_only_intervention': {
                'displacement_norm': distribution(np.linalg.norm(delta, axis=1)),
                'displacement_rms_over_centered_mu_rms': float(np.sqrt(delta_squared.mean() / trace_signal)),
                'displacement_mean_shift_squared_over_mu_variance': float(np.sum(delta.mean(0) ** 2) / trace_signal),
                'displacement_centered_variance_over_mu_variance': float(np.var(delta, axis=0).sum() / trace_signal),
                'displacement_noise_normalized_squared_norm_mean': float(np.sum(delta ** 2 / noise_diagonal, axis=1).mean()),
                'covariance_normalizations_by_relative_eigenvalue_cutoff': covariance_normalizations},
            'per_coordinate': {'mu_variance': np.diag(covariance).tolist(), 'noise_variance': noise_diagonal.tolist()},
            'deterministic_decoder_control_gaps': decoder}
        if labels is not None:
            assert data['row_identity_sha256'] == label_analysis['row_identity_sha256']
            correlations = {}
            for pc in range(4):
                direction = directions[:, -(pc + 1)]
                # Fix the arbitrary eigenvector sign using its largest loading.
                direction = direction * np.sign(direction[np.argmax(np.abs(direction))])
                scores = centered @ direction
                correlations[f'pc{pc + 1}'] = {key: correlation_report(scores, target) for key, target in labels.items()}
                if pc == 0:
                    correlations['pc1_current_only_displacement'] = {
                        key: correlation_report(delta @ direction, target) for key, target in labels.items()}
            reports[name]['descriptive_pc_label_correlations'] = correlations
        hashes[name] = {'posterior_codes': digest(path), 'results': digest(result_path) if result_path.is_file() else None}
    assert len(identities) == 1
    assert not torch.cuda.is_initialized()
    report = {'source': str(source), 'samples': 512, 'row_identity_sha256': identities.pop(),
              'precision': 'Saved posterior codes are float32; original FP32 evaluator provenance is in the source protocol, not inferred for overridden inputs.',
              'source_file_sha256': hashes, 'script_sha256': digest(__file__), 'models': reports,
              'label_analysis': label_analysis,
              'definitions_and_limits': [
                  'All covariance moments use population denominator N=512; float64 CPU arithmetic on saved float32 codes.',
                  'Expected posterior noise is exp(logvar); no sampled epsilon is needed for this audit.',
                  'Trace ratios are invariant to common scalar rescaling but not arbitrary changes of latent coordinates.',
                  'Generalized eigenvalues of (C,N) are invariant under consistent invertible linear transformations of both covariance matrices.',
                  'The posterior is input-dependent; N is its sample average, not a claim of homoscedastic noise.',
                  'Covariance spectra and signal/noise counts measure second moments, not semantic dimensions, information bits, or mutual information.',
                  'Covariance-whitened displacement excludes small eigenvalues and reports sensitivity to the relative cutoff.',
                  'Current-only is an input replacement, not a parameter-gradient or training-history measurement.',
                  'No old10k data with unmatched encoding precision is included.',
                  'No GPU operations or training performed.'
              ], 'cuda_initialized': False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(args.output)
    for name in args.models:
        m = reports[name]
        print(json.dumps({'model': name, 'mu_cov_top_fraction': m['mu_covariance']['top_fraction'],
                          'sn_directions_above_one': m['noise_normalized_signal_covariance']['directions_signal_to_noise_above_one'],
                          'mu_99percent_subspace': m['mu_principal_subspace_noise']['0.99'],
                          'control_gaps': m['deterministic_decoder_control_gaps']}))


if __name__ == '__main__':
    main()
