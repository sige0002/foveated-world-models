"""Audit and compare fixed-endpoint early condition experiments, including pending runs."""
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARMS = ('real', 'constant', 'rms_constant')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def identity(rows):
    return [(str(r['source_row']['episode']), r['source_row']['index'],
             r['source_row']['dt'], r['source_row'].get('endpoint_indices')) for r in rows]


def main():
    declared_path = ROOT / 'research/results/early-condition512-protocol-20261113.json'
    declared = read(declared_path)
    audit = read(ROOT / 'research/idm/early-condition-gradient-audit-20261113.json')
    expected = {'real': audit['real_U_gradient_norm'], 'constant': audit['constant_U_gradient_norm']}
    expected['rms_constant'] = expected['constant'] * audit['train_code_rms_norm'] / audit['train_mean_code_norm']
    models, pending, first_steps, protocols, sources = {}, [], {}, {}, {}
    sources[str(declared_path.relative_to(ROOT))] = sha(declared_path)
    for arm in ARMS:
        run = ROOT / f'runs/early-condition-{arm}-512-400-20261113'
        pp = run / 'protocol.json'
        if not pp.exists():
            pending.append(arm)
            continue
        p = read(pp)
        protocols[arm] = p
        assert p['arm'] == arm and p['execute_training'] is True
        for key in ('minibatch_seed', 'steps', 'batch_size', 'lr', 'weight_decay', 'gradient_clip'):
            assert p[key] == declared[key], key
        assert p['seed'] == declared['model_seed']
        for path, digest in p['source_hashes'].items():
            if path in declared['source_sha256']:
                assert digest == declared['source_sha256'][path]
        pre = read(run / 'preflight.json')
        assert pre['initial_exact'] and pre['initial_zero_exact']
        assert pre['trainable_parameters'] == declared['trainable_parameters']
        assert pre['source_split_counts'] == {'train': 512, 'validation': 64, 'observed96': 96}
        log = ROOT / f'research/results/early-condition-{arm}-20261113.log'
        entries = [json.loads(s) for s in log.read_text().splitlines() if s.startswith('{')]
        if entries:
            first = entries[0]
            assert first['step'] == 1
            assert np.isclose(first['preclip_gradient_norm'], expected[arm], rtol=1e-5, atol=1e-9)
            assert abs(first['mse'] - audit['mse']) < 1e-6
            first_steps[arm] = first
            backend = read(run / 'initial_backend.json')
            assert backend['train_vs_inference_max_abs'] <= p['initial_backend_tolerance']
            assert backend['train_vs_cached_baseline_max_abs'] <= p['initial_backend_tolerance']
        rp = run / 'results.json'
        if not rp.exists():
            pending.append(arm)
            continue
        r = read(rp)
        assert r['protocol_sha256'] == sha(pp)
        assert r['checkpoint_sha256'] == sha(run / 'adapter.pt')
        assert r['zero_exact'] and r['frozen_hash_unchanged']
        assert r['logs'][-1]['step'] == declared['steps']
        for split, count in pre['source_split_counts'].items():
            ev = r['evaluation'][split]
            assert len(ev['rows']) == count
            assert ev['zero_vs_cached_baseline_max_abs'] <= 1e-4
            assert abs(np.mean([row['mse'] for row in ev['rows']]) - ev['mse']) < 1e-12
        models[arm] = r
        sources[str(rp.relative_to(ROOT))] = sha(rp)
    if protocols:
        ref = next(iter(protocols.values()))
        for p in protocols.values():
            for key in ('source_hashes', 'initial_state_hash', 'frozen_hash', 'batch_hash', 'options'):
                assert p[key] == ref[key], key
    comparisons = {}
    if len(models) == len(ARMS):
        for split in ('train', 'validation', 'observed96'):
            real = models['real']['evaluation'][split]
            draws = np.random.default_rng(20261113).integers(0, len(real['rows']), (20000, len(real['rows'])))
            comparisons[split] = {}
            for control in ('zero_prior', 'constant', 'rms_constant'):
                if control == 'zero_prior':
                    control_errors = np.array([v['zero_mse'] for v in real['rows']])
                else:
                    other = models[control]['evaluation'][split]
                    assert identity(real['rows']) == identity(other['rows'])
                    assert np.allclose([v['zero_mse'] for v in real['rows']],
                                       [v['zero_mse'] for v in other['rows']], atol=1e-9, rtol=0)
                    control_errors = np.array([v['mse'] for v in other['rows']])
                diff = np.array([v['mse'] for v in real['rows']]) - control_errors
                comparisons[split][control] = {
                    'real_minus_control_mse': float(diff.mean()),
                    'relative_change_percent': float(100 * diff.mean() / control_errors.mean()),
                    'conditional_paired_ci95': np.quantile(diff[draws].mean(1), [.025, .975]).tolist()}
    result = {
        'complete': not pending, 'pending': pending, 'source_sha256': sources,
        'first_step_independent_gradient_audit': first_steps,
        'models': {a: {'evaluation': {s: {k: v for k, v in ev.items() if k != 'rows'}
                                    for s, ev in r['evaluation'].items()},
                       'condition_diagnostics': r['condition_diagnostics'], 'seconds': r['seconds']}
                   for a, r in models.items()},
        'real_code_comparisons': comparisons,
        'scope': 'Single seed fixed400, future-conditioned posterior reconstruction on reused evaluation data. Conditional paired intervals exclude training and adaptive selection uncertainty. Constants have different effective capacity; RMS matching does not match function-space regularization. Concurrent wall time is not an isolated speed benchmark.',
        'script_sha256': sha(Path(__file__))}
    path = ROOT / 'research/results/early-condition512-summary-20261113.json'
    path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'pending': pending, 'first_steps': first_steps, 'models': result['models']}, indent=2))


if __name__ == '__main__':
    main()
