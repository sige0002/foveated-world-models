"""Summarize fixed final-prior controls on previously observed episodes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evaluation', type=Path, nargs='+', required=True)
    parser.add_argument('--reference-reports', type=Path, nargs=3,
                        help='Fixed400 reference reports in seed42/43/44 order; first result per report')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('Output must be new')
    root = Path(__file__).resolve().parents[1]
    sources = {}

    def read(path):
        path = root / path
        raw = path.read_bytes()
        sources[str(path.relative_to(root))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    controls = [result for path in args.evaluation for result in read(path)['results']]
    ref_paths = args.reference_reports or [Path(f'research/results/reference-idm{s}-final-prior-observed96-20261113.json') for s in (42,43,44)]
    refs = [read(path)['results'][0] for path in ref_paths]
    assert [r['seed'] for r in refs] == [42,43,44]
    assert all(r['steps'] == 400 and not r['constant_code_control'] for r in refs)
    identities = [(r['episode'], r['index']) for r in refs[0]['rows']]
    assert len(set(identities)) == len(identities) == 96

    def values(result, key='mse'):
        assert [(r['episode'], r['index']) for r in result['rows']] == identities
        return np.array([r[key] for r in result['rows']])

    ref = np.stack([values(r) for r in refs])
    constant = sorted([r for r in controls if r['constant_code_control']], key=lambda r: r['seed'])
    assert [r['seed'] for r in constant] == [42, 43, 44]
    con = np.stack([values(r) for r in constant])
    flow, = [r for r in controls if r['variant'] == 'prior_flow']
    assert flow['seed'] == 42 and not flow['constant_code_control']

    def compare(a, b):
        diff = a - b
        rng = np.random.default_rng(20261113)
        bootstrap = diff[rng.integers(0, len(diff), (20000, len(diff)))].mean(1)
        return dict(candidate_mse=float(a.mean()), baseline_mse=float(b.mean()),
                    difference=float(diff.mean()), relative_reduction=float(1-a.mean()/b.mean()),
                    paired_episode_ci95=np.quantile(bootstrap, [.025, .975]).tolist())

    report = dict(
        scope='Previously observed96; fixed400; true-future posterior reconstruction. '
              'Mean per-model risks, not a prediction ensemble. Bootstrap conditions on fitted models.',
        sources=sources,
        reference_per_seed=ref.mean(1).tolist(), constant_per_seed=con.mean(1).tolist(),
        reference_vs_constant=compare(ref.mean(0), con.mean(0)),
        flow_sample_vs_direct42=compare(values(flow), ref[0]),
        flow_mean4_vs_direct42=compare(values(flow, 'mean_mse'), ref[0]),
        limitations='Additional IDM-head pretraining cost is not matched; single flow seed; '
                    'constant code tests extra capacity, not semantic action identity. '
                    'Equal training steps are not equal training/inference FLOPs.')
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
