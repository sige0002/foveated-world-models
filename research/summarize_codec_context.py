"""Paired, fixed-model comparisons for the prespecified codec context experiment."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    protocol_path = ROOT / 'research/results/codec-context-capacity512-protocol-20261113.json'
    protocol = json.loads(protocol_path.read_text())
    reports = {}
    hashes = {}
    for arm in protocol['arms']:
        path = ROOT / arm['output'] / 'results.json'
        reports[arm['name']] = json.loads(path.read_text())
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        report = reports[arm['name']]
        assert report['steps'] == protocol['steps'] and report['seed'] == protocol['seed']
        assert report['script_sha256'] == protocol['script_sha256']
        assert report['architecture'] == dict(dim=arm['dim'], layers=arm['layers'], baseline_context=True)
    for variant in ('direct_latent', 'flow'):
        path = ROOT / f'runs/codec32-{variant}-512-400-20261113-v2/results.json'
        reports['old-' + variant] = json.loads(path.read_text())
        hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    path = ROOT / 'research/results/nested512-direct-constant400-observed96-20261113.json'
    reference = json.loads(path.read_text())['results'][0]
    hashes[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    rows = {k: v['results']['observed96']['rows'] for k, v in reports.items()}
    rows['full-direct'] = reference['rows']
    identity = lambda rs: [(r['episode'], r['index'], r['dt'], r.get('endpoint_indices')) for r in rs]
    assert all(identity(rs) == identity(reference['rows']) for rs in rows.values())
    errors = {k: np.array([r['mean_mse'] for r in rs]) for k, rs in rows.items()}
    draws = np.random.default_rng(20261113).integers(0, 96, (20000, 96))
    pairs = [(a['name'], 'full-direct') for a in protocol['arms']]
    pairs += [('small-direct', 'old-direct_latent'), ('small-flow', 'old-flow'),
              ('large-direct', 'small-direct'), ('large-flow', 'small-flow'),
              ('small-flow', 'small-direct'), ('large-flow', 'large-direct'),
              ('large-feature', 'large-direct')]
    comparisons = {}
    for a, b in pairs:
        delta = errors[a] - errors[b]
        comparisons[a + '_minus_' + b] = {
            'mse_difference': float(delta.mean()),
            'relative_change_percent': float(100 * delta.mean() / errors[b].mean()),
            'paired_ci95': np.quantile(delta[draws].mean(1), [.025, .975]).tolist(),
        }
    result = {
        'protocol': str(protocol_path.relative_to(ROOT)), 'source_sha256': hashes,
        'models': {k: {'observed96_mse': float(errors[k].mean()),
                       'dev64': v['results']['dev64']['metrics'],
                       'observed96': v['results']['observed96']['metrics'],
                       'parameters_trainable': v['parameters_trainable']} for k, v in reports.items()},
        'full_direct_mse': float(errors['full-direct'].mean()), 'comparisons': comparisons,
        'scope': 'One seed, fixed400, reused96, future-conditioned IDM. Conditional bootstrap excludes training and adaptive selection uncertainty. Capacity differs across sizes and full-space baseline; fixed codec adds1000 steps.'}
    output = ROOT / 'research/results/codec-context-capacity512-summary-20261113.json'
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: float(v.mean()) for k, v in errors.items()}, indent=2))


if __name__ == '__main__':
    main()
