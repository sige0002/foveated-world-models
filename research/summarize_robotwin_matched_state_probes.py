"""CPU-only independent checks and compact tables for completed matched probes."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

MODELS = ('baseline3000', 'old10000', 'combined3000', 'official')
LABELS = {'baseline3000': '基準3000', 'old10000': '旧10000', 'combined3000': '両修正3000', 'official': '公式'}
XYZ = [0, 1, 2, 8, 9, 10]
GRIP = [7, 15]
TARGET_METRICS = ('state_mse', 'xyz_mse', 'gripper_mse')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def geometry(x):
    x = x.numpy().astype(np.float64)
    s = np.linalg.svd(x - x.mean(0), compute_uv=False)
    eigenvalues = s * s / len(x)
    p = eigenvalues / eigenvalues.sum()
    std = x.std(0)
    return {'samples': len(x), 'dimensions': x.shape[1], 'raw_covariance_trace': float(eigenvalues.sum()),
        'participation_ratio': float(1 / (p * p).sum()), 'top_eigenvalue_fraction': float(p[0]),
        'dimensions_for_95pct_variance': int(np.searchsorted(p.cumsum(), .95) + 1),
        'axis_std_min_median_max': np.quantile(std, [0, .5, 1]).tolist(),
        'median_inverse_axis_std': float(np.median(1 / np.maximum(std, 1e-12))),
        'eigenvalues_descending': eigenvalues.tolist()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    source, output = args.source.resolve(), args.output.resolve()
    result = json.loads((source / 'results.json').read_text())
    assert result['complete'] and set(result['models']) == set(MODELS)
    protocol = result['protocol']
    assert protocol['train_samples'] == 2048 and protocol['val_samples'] == 512
    old_path = Path(protocol['old_codes_path'])
    assert digest(old_path) == protocol['old_codes_sha256']
    old = torch.load(old_path, weights_only=True, map_location='cpu')
    yt, yv = [old[split]['delta'].numpy().astype(np.float64) for split in ('train', 'val')]
    st, sv = [old[split]['state'].numpy().astype(np.float64) for split in ('train', 'val')]
    train_eps = {r['episode'] for r in old['train']['rows']}
    val_eps = np.array([r['episode'] for r in old['val']['rows']])
    assert not train_eps & set(val_eps)
    unique, inverse = np.unique(val_eps, return_inverse=True)
    assert len(unique) == 415
    picks = np.random.default_rng(142).integers(len(unique), size=(4000, len(unique)))
    counts = np.bincount(inverse)
    denominator = counts[picks].sum(1)
    def interval(values):
        draws = np.bincount(inverse, weights=values)[picks].sum(1) / denominator
        return {'estimate': float(values.mean()), 'ci95': np.quantile(draws, [.025, .975]).tolist()}
    compact, metric_arrays, hashes = {}, {}, {}
    for name in MODELS:
        report = result['models'][name]
        assert report['unchanged'] and report['g_never_called']
        assert report['module_hashes_before'] == report['module_hashes_after']
        code_path, readout_path = source / (name + '-posterior-means.pt'), source / (name + '-readouts.pt')
        hashes[name] = {'posterior_means': digest(code_path), 'readouts': digest(readout_path)}
        codes = torch.load(code_path, weights_only=True, map_location='cpu')
        readouts = torch.load(readout_path, weights_only=True, map_location='cpu')
        zt, zv = [codes[split]['z'].numpy().astype(np.float64) for split in ('train', 'val')]
        ct, cv = [codes[split]['current_only_z'].numpy().astype(np.float64) for split in ('train', 'val')]
        designs = {'z': (zt, zv), 'state': (st, sv), 'state_z': (np.c_[st, zt], np.c_[sv, zv]),
                   'current_only_z': (ct, cv), 'state_current_only_z': (np.c_[st, ct], np.c_[sv, cv])}
        donors = readouts['donors'].numpy()
        assert donors.tolist() == protocol['donors'] and (val_eps[donors] != val_eps).all()
        predictions, zero_geometry = {}, {}
        for design, (xt, xv) in designs.items():
            fit = {k: v.numpy() for k, v in readouts['fits'][design].items()}
            np.testing.assert_allclose(fit['xm'], xt.mean(0), rtol=0, atol=1e-12)
            np.testing.assert_allclose(fit['xs'], np.maximum(xt.std(0), 1e-12), rtol=0, atol=1e-12)
            np.testing.assert_allclose(fit['ym'], yt.mean(0), rtol=0, atol=1e-12)
            if design == 'z':
                zero_geometry = {'raw_zero_standardized_L2': float(np.linalg.norm(fit['xm'] / fit['xs'])),
                    'training_standardized_L2_median_p95_max': np.quantile(np.linalg.norm((xt - fit['xm']) / fit['xs'], axis=1), [.5, .95, 1]).tolist(),
                    'raw_train_mean_L2': float(np.linalg.norm(fit['xm'])),
                    'scope': 'Distance in the fitted axis-standardized z coordinates; diagnostic for extrapolation, not a decoder intervention.'}
            def predict(x):
                return (x - fit['xm']) / fit['xs'] @ fit['w'] + fit['ym']
            predictions[design] = predict(xv)
            if design in ('z', 'state_z'):
                predictions[design + '_zero'] = predict(np.zeros_like(zv) if design == 'z' else np.c_[sv, np.zeros_like(zv)])
                predictions[design + '_shuffled'] = predict(zv[donors] if design == 'z' else np.c_[sv, zv[donors]])
        predictions['constant'] = np.broadcast_to(yt.mean(0), yv.shape)
        predictions['no_change'] = np.zeros_like(yv)
        metrics = {}
        for design, prediction in predictions.items():
            np.testing.assert_allclose(prediction, readouts['predictions'][design].numpy(), rtol=0, atol=1e-12)
            error = (prediction - yv) ** 2
            metrics[design] = {'state_mse': error.mean(1), 'xyz_mse': error[:, XYZ].mean(1), 'gripper_mse': error[:, GRIP].mean(1)}
            for metric, values in metrics[design].items():
                np.testing.assert_allclose(values.mean(), report['metrics'][design][metric], rtol=0, atol=1e-12)
        contrasts = {}
        for contrast in report['contrasts']:
            treatment, control = contrast['treatment'], contrast['control']
            key = treatment + '__vs__' + control
            contrasts[key] = {}
            for metric in TARGET_METRICS:
                estimate = interval(metrics[control][metric] - metrics[treatment][metric])
                prior = contrast['metrics'][metric]
                np.testing.assert_allclose(estimate['estimate'], prior['control_minus_treatment'], rtol=0, atol=1e-12)
                np.testing.assert_allclose(estimate['ci95'], prior['ci95'], rtol=0, atol=1e-12)
                contrasts[key][metric] = estimate
        metric_arrays[name] = metrics
        compact[name] = {'geometry': {split: {kind: geometry(codes[split][kind]) for kind in ('z', 'current_only_z')} for split in ('train', 'val')},
            'raw_zero_geometry_in_z_head': zero_geometry,
            'metrics': {design: {key: float(values.mean()) for key, values in m.items()} for design, m in metrics.items()},
            'contrasts': contrasts}
    official_comparisons = {}
    for name in MODELS[:-1]:
        official_comparisons[name] = {}
        for design in ('z', 'state_z', 'current_only_z', 'state_current_only_z'):
            official_comparisons[name][design] = {}
            for metric in TARGET_METRICS:
                estimate = interval(metric_arrays[name][design][metric] - metric_arrays['official'][design][metric])
                prior = result['paired_comparisons_to_official'][name][design][metric]
                np.testing.assert_allclose(estimate['estimate'], prior['local_minus_official'], rtol=0, atol=1e-12)
                np.testing.assert_allclose(estimate['ci95'], prior['ci95'], rtol=0, atol=1e-12)
                official_comparisons[name][design][metric] = estimate
    analysis = {'complete': True, 'source': str(source), 'results_sha256': digest(source / 'results.json'),
        'analysis_script_sha256': digest(__file__), 'artifact_hashes': hashes, 'models': compact,
        'paired_local_minus_official': official_comparisons,
        'independent_checks': 'Recomputed every saved head prediction, raw-zero/shuffle intervention, train-only full-fit scaling, all requested MSEs and 4000-episode-bootstrap CIs from persisted codes/readouts/common labels. No fit, optimizer or GPU was used.',
        'limitations': result['limitations']}
    output.mkdir(parents=True, exist_ok=True)
    (output / 'summary.json').write_bytes((source / 'results.json').read_bytes())
    (output / 'analysis.json').write_text(json.dumps(analysis, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'complete': True, 'output': str(output / 'analysis.json'), 'models': {name: {'state_plus_z': compact[name]['metrics']['state_z']['state_mse'], 'state_only': compact[name]['metrics']['state']['state_mse'], 'z_only': compact[name]['metrics']['z']['state_mse'], 'current_only_z': compact[name]['metrics']['current_only_z']['state_mse']} for name in MODELS}}), flush=True)


if __name__ == '__main__':
    main()
