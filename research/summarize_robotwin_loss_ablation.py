"""Read live/final paired RoboTwin loss ablations without loading models.

Example:
  python3 research/summarize_robotwin_loss_ablation.py --runs-root <SSD-run-group>
  python3 research/summarize_robotwin_loss_ablation.py \
      --run baseline=<dir> --run kl_only=<dir> \
      --run mse_only=<dir> --run kl_mse=<dir> --output <summary.json>

The objective values differ between arms; common visual/state MSE and latent
interventions are compared instead. Bootstrap intervals condition on these
fixed trained weights and validation episodes, not on training-seed variance.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sys


ARMS = ('baseline', 'kl_only', 'mse_only', 'kl_mse')
ARM_FACTORS = {
    'baseline': {'image_loss': 'smooth_l1', 'kl_beta': 5e-5},
    'kl_only': {'image_loss': 'smooth_l1', 'kl_beta': 1e-5},
    'mse_only': {'image_loss': 'l2', 'kl_beta': 5e-5},
    'kl_mse': {'image_loss': 'l2', 'kl_beta': 1e-5},
}
CONTRASTS = {
    'kl_only_minus_baseline': {'kl_only': 1, 'baseline': -1},
    'mse_only_minus_baseline': {'mse_only': 1, 'baseline': -1},
    'both_minus_baseline': {'kl_mse': 1, 'baseline': -1},
    'kl_marginal_effect': {'kl_only': .5, 'baseline': -.5, 'kl_mse': .5, 'mse_only': -.5},
    'mse_marginal_effect': {'mse_only': .5, 'baseline': -.5, 'kl_mse': .5, 'kl_only': -.5},
    'interaction': {'kl_mse': 1, 'kl_only': -1, 'mse_only': -1, 'baseline': 1},
}
HASH_KEYS = ('cumulative_data_sha256', 'cumulative_aug_sha256', 'cumulative_epsilon_sha256')


def read_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def read_jsonl(path: Path, warnings: list[str]):
    if not path.is_file():
        return []
    lines = path.read_text().splitlines()
    records = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            # Atomic JSON files are preferred; a writer can be halfway through
            # appending the final JSONL line during a live snapshot.
            if index == len(lines)-1:
                warnings.append(f'Ignored incomplete final line: {path}')
            else:
                raise
    return records


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def percentile(sorted_values, p):
    location = (len(sorted_values)-1)*p
    lo = int(location)
    hi = min(lo+1, len(sorted_values)-1)
    return sorted_values[lo]*(hi-location)+sorted_values[hi]*(location-lo) if hi != lo else sorted_values[lo]


def episode_bootstrap(values, episodes, *, draws, seed):
    groups = defaultdict(lambda: [0., 0])
    for value, episode in zip(values, episodes):
        groups[str(episode)][0] += value
        groups[str(episode)][1] += 1
    clusters = list(groups.values())
    result = {'estimate': sum(values)/len(values), 'samples': len(values), 'episodes': len(clusters)}
    if draws <= 0 or len(clusters) < 2:
        result['ci95'] = None
        return result
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        total = 0.
        count = 0
        for _ in clusters:
            value, n = clusters[rng.randrange(len(clusters))]
            total += value
            count += n
        estimates.append(total/count)
    estimates.sort()
    result['ci95'] = [percentile(estimates, .025), percentile(estimates, .975)]
    return result


def metrics_with_utility(record):
    values = {k: v for k, v in record.get('metrics', {}).items() if finite_number(v)}
    for target in ('visual_mse', 'state_mse'):
        for control in ('mean', 'swapped', 'zero'):
            key = f'{control}_minus_real_{target}'
            if key not in values and f'{control}_{target}' in values and f'real_{target}' in values:
                values[key] = values[f'{control}_{target}']-values[f'real_{target}']
    return values


def row_metric(row, metric):
    values = row.get('metrics', {})
    if '_minus_real_' in metric:
        control, target = metric.split('_minus_real_', 1)
        if control == 'swapped' and not row.get('swap_valid', True):
            return None
        a, b = values.get(f'{control}_{target}'), values.get(f'real_{target}')
        return a-b if finite_number(a) and finite_number(b) else None
    value = values.get(metric)
    return value if finite_number(value) else None


def summarize(runs, *, step=None, draws=4000, seed=1942):
    warnings = []
    records = {}
    streams = {}
    arms = {}
    for name in ARMS:
        path = runs[name]
        history = read_jsonl(path/'z_validation.jsonl', warnings)
        records[name] = {int(row['step']): row for row in history}
        streams[name] = {int(row['step']): row for row in read_jsonl(path/'training_stream.jsonl', warnings)}
        initial = read_json(path/'initial_state_fingerprint.json')
        run_summary = read_json(path/'run_summary.json')
        progress = read_json(path/'progress.json')
        arms[name] = {
            'run': str(path.resolve()),
            'expected_factors': ARM_FACTORS[name],
            'last_logged_step': progress.get('step') if progress else None,
            'run_summary_phase': run_summary.get('phase') if run_summary else None,
            'initial_state_sha256': initial.get('sha256') if initial else None,
            'observed_factors': {'image_loss': initial.get('loss_type'), 'kl_beta': initial.get('kl_beta')} if initial else None,
            'trainer_snapshot_sha256': hashlib.sha256((path/'train_robotwin_stage1.py').read_bytes()).hexdigest() if (path/'train_robotwin_stage1.py').is_file() else None,
            'available_validation_steps': sorted(records[name]),
        }
    common = sorted(set.intersection(*(set(records[name]) for name in ARMS)))
    selected = step if step is not None else (common[-1] if common else None)
    result = {
        'generated_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'status': 'awaiting_common_validation' if selected is None else 'snapshot',
        'selected_step': selected,
        'common_validation_steps': common,
        'arms': arms,
        'verification': {},
        'warnings': warnings,
        'limitations': [
            'A single seed and 3000 updates cannot establish seed robustness or convergence.',
            'Matching initialization and recorded streams does not prove bitwise determinism of all GPU kernels.',
            'Different training objective values are not comparable; comparisons use common MSE metrics.',
            'Positive swap/mean error gaps alone can reflect worsened predictions; assess real prediction error jointly.',
            'Latent variance and node scale are geometry diagnostics, not action information or policy success.',
            'Mean-z controls are validation means if the trainer uses that protocol; they are not separately trained constant-z models.',
            'Bootstrap intervals are conditional on fixed weights and sampled episodes, with no correction for multiple comparisons.',
        ],
        'bootstrap': {'draws': draws, 'seed': seed, 'unit': 'episode', 'paired_across_arms': True},
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    init_hashes = {name: arms[name]['initial_state_sha256'] for name in ARMS}
    result['verification']['initial_weights'] = {
        'values': init_hashes,
        'all_present': all(init_hashes.values()),
        'all_equal': all(init_hashes.values()) and len(set(init_hashes.values())) == 1,
    }
    observed = {name: arms[name]['observed_factors'] for name in ARMS}
    result['verification']['loss_factors'] = {
        'observed': observed,
        'match_expected': all(observed[name] == ARM_FACTORS[name] for name in ARMS),
    }
    snapshots = {name: arms[name]['trainer_snapshot_sha256'] for name in ARMS}
    result['verification']['trainer_snapshots'] = {
        'values': snapshots,
        'all_present': all(snapshots.values()),
        'all_equal': all(snapshots.values()) and len(set(snapshots.values())) == 1,
    }
    if selected is None:
        return result
    missing = [name for name in ARMS if selected not in records[name]]
    if missing:
        result['status'] = 'requested_step_incomplete'
        warnings.append(f'Validation step {selected} is missing for: {", ".join(missing)}')
        return result
    stream_checks = {}
    for key in HASH_KEYS:
        values = {name: streams[name].get(selected, {}).get(key) for name in ARMS}
        stream_checks[key] = {'values': values, 'all_present': all(values.values()),
                              'all_equal': all(values.values()) and len(set(values.values())) == 1}
    result['verification']['training_streams_at_selected_step'] = stream_checks
    training_match = (result['verification']['initial_weights']['all_equal']
                      and result['verification']['loss_factors']['match_expected']
                      and result['verification']['trainer_snapshots']['all_equal']
                      and all(v['all_equal'] for v in stream_checks.values()))
    result['verification']['initialization_and_recorded_streams_match'] = training_match
    if not training_match:
        warnings.append('Initial weights, trainer snapshots, loss settings, or recorded data/augmentation/epsilon streams are missing or mismatched; causal pairing is not verified.')
    rows = {}
    metric_values = {}
    for name in ARMS:
        record = records[name][selected]
        metric_values[name] = metrics_with_utility(record)
        arms[name]['selected_validation'] = record
        arms[name]['comparison_metrics'] = metric_values[name]
        row_path = runs[name]/f'z_validation_step{selected:06d}.json'
        payload = read_json(row_path)
        rows[name] = payload.get('rows') if payload else None
        if payload and int(payload['step']) != selected:
            raise ValueError(f'Unexpected step in {row_path}')
    all_rows = all(rows.values())
    identity = lambda row: (row.get('episode'), row.get('base'))
    identities_match = all_rows and all([identity(r) for r in rows[name]] == [identity(r) for r in rows['baseline']] for name in ARMS)
    donor_identity = lambda row: (row.get('swap_valid', True), row.get('donor_episode'), row.get('donor_base'))
    donors_match = identities_match and all([donor_identity(r) for r in rows[name]] == [donor_identity(r) for r in rows['baseline']] for name in ARMS)
    result['verification']['validation_rows_present'] = all_rows
    result['verification']['validation_row_identities_match'] = bool(identities_match)
    result['verification']['recorded_swap_donors_match'] = bool(donors_match)
    if not identities_match:
        warnings.append('Matching per-example validation rows are unavailable; confidence intervals are omitted.')
    metrics = sorted(set.intersection(*(set(metric_values[name]) for name in ARMS)))
    focus = [k for k in metrics if k in ('real_visual_mse', 'real_state_mse') or '_minus_real_' in k]
    result['per_arm_intervention_intervals'] = {}
    if identities_match:
        for name in ARMS:
            intervals = {}
            for metric in focus:
                eligible = [(row_metric(row, metric), row['episode']) for row in rows[name]]
                eligible = [(v,e) for v,e in eligible if v is not None]
                if eligible:
                    values, episodes = zip(*eligible)
                    intervals[metric] = episode_bootstrap(values, episodes, draws=draws, seed=seed)
            result['per_arm_intervention_intervals'][name] = intervals
    result['factorial_contrasts'] = {}
    for name, coefficients in CONTRASTS.items():
        effects = {}
        for metric in focus:
            estimate = sum(weight*metric_values[arm][metric] for arm,weight in coefficients.items())
            comparison = {'estimate_from_aggregate': estimate, 'ci95': None,
                          'preferred_sign': 'negative' if metric.startswith('real_') else 'positive',
                          'paired_training_verified': training_match}
            if identities_match and (donors_match or not metric.startswith('swapped_')):
                values, episodes = [], []
                for index, reference in enumerate(rows['baseline']):
                    components = {arm: row_metric(rows[arm][index], metric) for arm in coefficients}
                    if any(v is None for v in components.values()):
                        continue
                    values.append(sum(coefficients[arm]*value for arm,value in components.items()))
                    episodes.append(reference['episode'])
                if values:
                    comparison.update(episode_bootstrap(values, episodes, draws=draws, seed=seed))
            effects[metric] = comparison
        result['factorial_contrasts'][name] = {'coefficients': coefficients, 'metrics': effects}
    result['status'] = 'paired_snapshot' if training_match and identities_match and donors_match else 'unverified_snapshot'
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, help='Parent directory containing baseline, kl_only, mse_only, kl_mse')
    parser.add_argument('--run', action='append', default=[], metavar='ARM=PATH')
    parser.add_argument('--step', type=int, help='Default: newest validation step present in all four arms')
    parser.add_argument('--bootstrap', type=int, default=4000)
    parser.add_argument('--seed', type=int, default=1942)
    parser.add_argument('--output', type=Path, help='Write an atomic summary JSON; otherwise print JSON')
    args = parser.parse_args()
    if args.bootstrap < 0:
        parser.error('--bootstrap must be nonnegative')
    runs = {name: args.runs_root/name for name in ARMS} if args.runs_root else {}
    for spec in args.run:
        name, sep, path = spec.partition('=')
        if not sep or name not in ARMS or not path:
            parser.error('--run must be ARM=PATH with a known arm')
        runs[name] = Path(path)
    if set(runs) != set(ARMS):
        parser.error('Provide --runs-root or all four --run mappings')
    summary = summarize(runs, step=args.step, draws=args.bootstrap, seed=args.seed)
    rendered = json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False)+'\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.output.with_suffix(args.output.suffix+'.tmp')
        temporary.write_text(rendered)
        temporary.replace(args.output)
        print(json.dumps({'output': str(args.output), 'status': summary['status'],
                          'step': summary['selected_step'], 'warnings': summary['warnings']}, ensure_ascii=False))
    else:
        sys.stdout.write(rendered)


if __name__ == '__main__':
    main()
