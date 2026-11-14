"""CPU-only matched512 geometry and decoder intervention trajectory audit."""
from __future__ import annotations

import csv
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
ARMS = tuple(f'seed{seed}_{kind}' for seed in (43, 44) for kind in ('replay', 'new'))
IDENTITY = ('episode', 'base', 'donor_index', 'donor_episode', 'donor_base')
CONTROLS = ('zero', 'mean', 'swapped', 'current_only')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(path):
    report = json.loads((path / 'results.json').read_text())
    code = torch.load(path / 'posterior-codes.pt', map_location='cpu', weights_only=True)
    assert code['mu'].shape == code['logvar'].shape == (512, 1, 32)
    assert all(code[k].dtype == torch.float32 and torch.isfinite(code[k]).all()
               for k in ('mu', 'logvar', 'current_only_mu', 'nodes'))
    mu = code['mu'].double().flatten(1).numpy()
    centered = mu - mu.mean(0)
    covariance = centered.T @ centered / len(mu)
    noise = code['logvar'].double().exp().flatten(1).numpy().mean(0)
    eigenvalues = np.maximum(np.linalg.eigvalsh(covariance), 0)[::-1]
    proportions = eigenvalues / eigenvalues.sum()
    snr = np.linalg.eigvalsh(covariance / np.sqrt(noise[:, None] * noise[None, :]))[::-1]
    metrics = {key: float(np.mean([r['metrics'][key] for r in report['rows']]))
               for key in report['rows'][0]['metrics']}
    for control in CONTROLS:
        for metric in ('visual_mse', 'state_mse'):
            if 'real_' + metric in metrics:
                metrics[control + '_minus_real_' + metric] = metrics[control + '_' + metric] - metrics['real_' + metric]
        metrics[control + '_visual_increase_percent'] = 100 * metrics[control + '_minus_real_visual_mse'] / metrics['real_visual_mse']
    assert abs(eigenvalues.sum() - report['geometry']['mu_variance_sum']) < 1e-12
    assert abs(noise.sum() - report['geometry']['noise_variance_sum']) < 1e-12
    geometry = {'pc1_fraction': float(proportions[0]), 'mu_variance_trace': float(eigenvalues.sum()),
                'non_pc1_variance': float(eigenvalues[1:].sum()), 'noise_variance_trace': float(noise.sum()),
                'participation_ratio': float(1 / np.sum(proportions ** 2)),
                'snr_directions_gt1': int((snr > 1).sum()), 'snr_largest': float(snr[0]),
                'snr_second': float(snr[1]), 'snr_eigenvalues': snr.tolist(),
                'mu_eigenvalues': eigenvalues.tolist(),
                'pre_norm_node_coordinate_std_mean': report['geometry']['pre_norm_node_coordinate_std_mean']}
    episodes, inverse = np.unique([r['episode'] for r in report['rows']], return_inverse=True)
    picks = np.random.default_rng(1942).integers(len(episodes), size=(4000, len(episodes)))
    denominator = np.bincount(inverse)[picks].sum(1)
    intervals = {}
    for control in CONTROLS:
        values = np.array([r['metrics'][control + '_visual_mse'] - r['metrics']['real_visual_mse'] for r in report['rows']])
        bootstrap = np.bincount(inverse, weights=values)[picks].sum(1) / denominator
        intervals[control + '_minus_real_visual_mse'] = {'estimate': float(values.mean()),
            'ci95': np.quantile(bootstrap, [.025, .975]).tolist()}
    return {'geometry': geometry, 'metrics': metrics, 'decoder_intervention_ci95': intervals,
            'result_sha256': digest(path / 'results.json'), 'codes_sha256': digest(path / 'posterior-codes.pt'),
            'source': str(path)}, report, code['row_identity_sha256']


def main():
    torch.set_num_threads(1)
    models, reports, identities, code_identities = {}, {}, {}, {}
    comparisons = {step: json.loads((GROUP / f'paired-analysis/comparison-step{step:06d}.json').read_text())
                   for step in (3500, 4000)}
    locations = {'initial3000': INITIAL / 'kl_mse', 'official': INITIAL / 'official'}
    locations.update({f'{arm}_{step}': GROUP / ('recovery-v2' if step > 3500 else '') / arm / 'evaluations' / f'step{step:06d}'
                      for step in (3500, 4000) for arm in ARMS})
    for name, path in locations.items():
        models[name], reports[name], code_identities[name] = analyze(path)
        identities[name] = [[r[k] for k in IDENTITY] for r in reports[name]['rows']]
        assert identities[name] == identities['initial3000']
        assert code_identities[name] == code_identities['initial3000']
        assert len(identities[name]) == 512 and len({r[0] for r in identities[name]}) == 415
        if name not in ('initial3000', 'official'):
            arm, step = name.rsplit('_', 1); comparison = comparisons[int(step)]
            assert reports[name]['absolute_step'] == int(step) and reports[name]['samples'] == 512
            assert reports[name]['precision'] == 'highest FP32 cached DINO/encoder/VAE/decoder; no autocast; TF32 disabled'
            assert models[name]['result_sha256'] == comparison['source_hashes']['validation_sha256'][arm]
            for key, expected in comparison['arm_and_official_estimates'][arm].items():
                assert abs(models[name]['metrics'][key] - expected) < 1e-12
        else:
            assert all(models[name]['result_sha256'] == c['source_hashes'][name + '_sha256'] for c in comparisons.values())
    geometry_keys = ('pc1_fraction', 'mu_variance_trace', 'non_pc1_variance', 'noise_variance_trace',
                     'participation_ratio', 'snr_directions_gt1', 'snr_largest', 'snr_second')
    contrasts = {}
    for step in (3500, 4000):
        paired = {}
        for seed in (43, 44):
            new, replay = (models[f'seed{seed}_{kind}_{step}']['geometry'] for kind in ('new', 'replay'))
            paired[str(seed)] = {k: new[k] - replay[k] for k in geometry_keys}
        contrasts[str(step)] = {'geometry_new_minus_replay': paired,
            'geometry_two_seed_mean_new_minus_replay': {k: sum(p[k] for p in paired.values()) / 2 for k in geometry_keys},
            'decoder_paired_new_minus_replay_ci95': comparisons[step]['paired_new_minus_replay'],
            'decoder_two_seed_mean_new_minus_replay_ci95': comparisons[step]['two_seed_average_new_minus_replay'],
            'decoder_per_arm_minus_initial3000_ci95': comparisons[step]['paired_arm_minus_initial3000']}
    assert not torch.cuda.is_initialized()
    result = {'samples': 512, 'episodes': 415, 'steps': [3000, 3500, 4000], 'models': models, 'contrasts': contrasts,
        'identity_sha256': hashlib.sha256(json.dumps(identities['initial3000'], separators=(',', ':')).encode()).hexdigest(),
        'posterior_row_identity_sha256': code_identities['initial3000'], 'script_sha256': digest(__file__),
        'comparison_sha256': {str(s): digest(GROUP / f'paired-analysis/comparison-step{s:06d}.json') for s in comparisons},
        'limits': ['Population covariance of posterior means; not sampled-z rank or information in bits.',
                   'SNR eigenvalues use diag(mean(exp(logvar))) as posterior noise covariance. Noise is heteroscedastic; this is a second-moment diagnostic.',
                   'Geometry contrasts are descriptive; no CI for eigenvalue/rank statistics. Decoder CIs use paired415-episode bootstrap4000.',
                   'Two continuation seeds from one trained checkpoint do not estimate general initial-training seed variance.',
                   '3500 is committed original output;4000 comes only from verified recovery after original precision failure.',
                   'Current-only and latent swaps may be out of distribution; variance amplitude alone does not establish useful latent semantics.',
                   'Task diversity and full-rank XYZ labels weaken simple monotony explanations but do not rule out limited training scale or conditional future ambiguity.'],
        'cuda_initialized': False}
    output = ROOT / 'research/results/robotwin-continuation4000-geometry-20261114.json'
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    table = []
    for name, model in models.items():
        g, m = model['geometry'], model['metrics']
        table.append({'model': name, 'pc1_percent': 100 * g['pc1_fraction'],
                      'mu_variance': g['mu_variance_trace'], 'non_pc1_variance': g['non_pc1_variance'],
                      'snr_gt1': g['snr_directions_gt1'], 'snr1': g['snr_largest'], 'snr2': g['snr_second'],
                      'real_visual_mse': m['real_visual_mse'],
                      **{c + '_increase_percent': m[c + '_visual_increase_percent'] for c in CONTROLS}})
    with output.with_suffix('.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    print(json.dumps(table, indent=2))
    print(output)


if __name__ == '__main__':
    main()
