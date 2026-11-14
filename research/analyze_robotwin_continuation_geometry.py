"""CPU audit of the saved matched512 posterior codes at continuation step3500."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OPENBLAS_NUM_THREADS'] = '1'
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path('/media/sadasue/SSD1/foveated-world-models/runs')
GROUP = RUNS / 'robotwin-data-step-continuation-b4-20261114'
INITIAL = RUNS / 'robotwin-loss-ablation-b4-20261114/evaluations/final3000-highest-fp32-v2'
IDENTITY = ('episode', 'base', 'donor_index', 'donor_episode', 'donor_base')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(1)
    comparison_path = GROUP / 'paired-analysis/comparison-step003500.json'
    comparison = json.loads(comparison_path.read_text())
    arms = ['seed43_replay', 'seed43_new', 'seed44_replay', 'seed44_new']
    locations = {'initial3000': INITIAL / 'kl_mse', 'official': INITIAL / 'official',
                 **{arm: GROUP / arm / 'evaluations/step003500' for arm in arms}}
    base_contract = json.loads((RUNS / 'robotwin-loss-ablation-b4-20261114/kl_mse/data_contract.json').read_text())
    models, hashes = {}, {}
    reference_rows = None
    reference_code_sha = None
    for name, path in locations.items():
        result_path, code_path = path / 'results.json', path / 'posterior-codes.pt'
        result = json.loads(result_path.read_text())
        code = torch.load(code_path, map_location='cpu', weights_only=True)
        if name in arms:
            assert result['absolute_step'] == 3500 and result['samples'] == 512
            assert 'highest FP32' in result['precision']
            assert digest(result_path) == comparison['source_hashes']['validation_sha256'][name]
            assert json.loads((GROUP / name / 'data_contract.json').read_text()) == base_contract
        else:
            assert digest(result_path) == comparison['source_hashes'][name + '_sha256']
        identities = [[row[key] for key in IDENTITY] for row in result['rows']]
        if reference_rows is None:
            reference_rows = identities
            reference_code_sha = code['row_identity_sha256']
        assert identities == reference_rows
        assert code['row_identity_sha256'] == reference_code_sha
        assert hashlib.sha256(json.dumps(identities, separators=(',', ':')).encode()).hexdigest() == comparison['identity_sha256']
        assert len(identities) == 512 and len({row[0] for row in identities}) == 415
        assert all(row[0] != row[3] and identities[row[2]][:2] == [row[3], row[4]] for row in identities)
        assert code['mu'].shape == code['logvar'].shape == code['nodes'].shape == (512, 1, 32)
        assert all(code[key].dtype == torch.float32 and torch.isfinite(code[key]).all()
                   for key in ['mu', 'logvar', 'nodes', 'current_only_mu'])
        mu = code['mu'].double().flatten(1).numpy()
        centered = mu - mu.mean(0)
        covariance = centered.T @ centered / len(mu)
        noise = code['logvar'].double().exp().flatten(1).numpy().mean(0)
        eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)[::-1]
        proportions = eigenvalues / eigenvalues.sum()
        snr = np.linalg.eigvalsh(covariance / np.sqrt(noise[:, None] * noise[None, :]))[::-1]
        metrics = {'real_visual_mse': float(np.mean([row['metrics']['real_visual_mse'] for row in result['rows']]))}
        for control in ['zero', 'mean', 'swapped', 'current_only']:
            metrics[control + '_minus_real_visual_mse'] = float(np.mean([
                row['metrics'][control + '_visual_mse'] - row['metrics']['real_visual_mse'] for row in result['rows']]))
        for key, value in metrics.items():
            assert abs(value - comparison['arm_and_official_estimates'][name][key]) < 1e-12
        assert abs(float(eigenvalues.sum()) - result['geometry']['mu_variance_sum']) < 1e-12
        assert abs(float(noise.sum()) - result['geometry']['noise_variance_sum']) < 1e-12
        delta = mu - code['current_only_mu'].double().flatten(1).numpy()
        models[name] = {
            'source': str(path), 'mu_pc1_fraction': float(proportions[0]),
            'mu_participation_ratio': float(1 / np.sum(proportions ** 2)),
            'mu_entropy_effective_rank': float(np.exp(-np.sum(proportions * np.log(np.maximum(proportions, 1e-300))))),
            'mu_variance_trace': float(eigenvalues.sum()), 'posterior_noise_trace': float(noise.sum()),
            'mu_covariance_eigenvalues_descending': eigenvalues.tolist(),
            'noise_normalized_signal_eigenvalues_descending': snr.tolist(),
            'snr_directions_above_one': int(np.sum(snr > 1)),
            'current_only_delta_rms_over_mu_centered_rms': float(np.sqrt(np.sum(delta ** 2, axis=1).mean() / eigenvalues.sum())),
            'decoder_intervention_metrics': metrics}
        hashes[name] = {'results': digest(result_path), 'posterior_codes': digest(code_path)}
    for name, model in models.items():
        model['pc1_change_vs_initial3000_percentage_points'] = 100 * (model['mu_pc1_fraction'] - models['initial3000']['mu_pc1_fraction'])
    assert not torch.cuda.is_initialized()
    report = {'samples': 512, 'episodes': 415, 'absolute_step': 3500, 'additional_updates': 500,
              'watcher_comparison_path': str(comparison_path), 'watcher_comparison_sha256': digest(comparison_path),
              'watcher_identity_sha256': comparison['identity_sha256'], 'posterior_row_identity_sha256': reference_code_sha,
              'all_rows_donors_metrics_and_source_hashes_match_watcher': True,
              'source_sha256': hashes, 'script_sha256': digest(__file__), 'models': models,
              'definition': 'Population covariance in CPUfloat64 on saved common FP32 means. Directional SNR eigenvalues=eig(N^(-1/2) Cov(mu) N^(-1/2)); N=diag(mean(exp(logvar))).',
              'limits': [
                  'All512 validation identities and donor assignments match the original3000 and watcher comparison; no64-row or different-precision inputs are mixed.',
                  'High PC1 fraction means concentrated variance, not exactly rank1, identical outputs, or absence of learning.',
                  'Every model has its own PCA direction; this comparison does not establish alignment or identical semantics of those directions.',
                  'These are two continuation seeds from one common trained checkpoint; not independent initial-training replications.',
                  'Step3500 is the first full continuation evaluation, before the faulty precision restoration and subsequent evaluation failure.',
                  'No GPU/model training or modifications to active source files.'
              ], 'cuda_initialized': False}
    output = ROOT / 'research/results/robotwin-continuation3500-geometry-20261114.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    for name, model in models.items():
        print(json.dumps({'model': name, 'pc1_percent': 100 * model['mu_pc1_fraction'],
                          'snr_directions_gt1': model['snr_directions_above_one'],
                          'mu_variance': model['mu_variance_trace'], **model['decoder_intervention_metrics']}))
    print(output)


if __name__ == '__main__':
    main()
