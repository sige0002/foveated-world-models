"""CPU-only observation and paired analysis of the four launched continuations.

Reads existing processes/artifacts; writes only analysis JSON. Never starts,
restarts, signals or modifies a training process. Default is one snapshot.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

ARMS = tuple(f'seed{s}_{c}' for s in (43, 44) for c in ('replay', 'new'))
CONTROLS = ('zero', 'mean', 'swapped', 'current_only')
METRICS = ('real_visual_mse', *(f'{c}_minus_real_visual_mse' for c in CONTROLS))
IDENTITY = ('episode', 'base', 'donor_index', 'donor_episode', 'donor_base')
RUNS = Path('/media/sadasue/SSD1/foveated-world-models/runs')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    try:
        return json.loads(Path(path).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def read_stream(path):
    """Ignore an unfinished last line; do not count a malformed complete line."""
    try:
        text = path.read_text()
    except FileNotFoundError:
        return [], []
    rows, errors = [], []
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        if not line.endswith('\n'):
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            errors.append(f'malformed complete stream line {number}')
    if [r.get('local_step') for r in rows] != list(range(1, len(rows) + 1)):
        errors.append('stream local steps are not contiguous from 1')
    if any(r.get('absolute_step') != 3000 + r.get('local_step', -1) for r in rows):
        errors.append('stream absolute/local step mismatch')
    return rows, errors


def process_status(job):
    try:
        proc = Path('/proc') / str(job['pid'])
        command = proc.joinpath('cmdline').read_bytes().rstrip(b'\0').decode().split('\0')
        stat = proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
        alive = stat[0] != 'Z'
        matches = command == job['command']
        if 'start_ticks' in job:
            matches &= int(stat[19]) == int(job['start_ticks'])
        return {'alive': alive, 'identity_matches': matches, 'state': stat[0]}
    except (FileNotFoundError, ProcessLookupError):
        return {'alive': False, 'identity_matches': False, 'state': 'absent'}


def paired_streams(streams):
    evidence, errors = {}, []
    keys = ('epsilon_sha256', 'augmentation_rng_before_sha256', 'augmentation_rng_after_sha256',
            'cumulative_epsilon_sha256', 'cumulative_augmentation_rng_sha256')
    for seed in (43, 44):
        replay, new = streams[f'seed{seed}_replay'], streams[f'seed{seed}_new']
        shared = min(len(replay), len(new))
        mismatches = [(i + 1, key) for i in range(shared) for key in keys
                      if replay[i].get(key) is None or replay[i].get(key) != new[i].get(key)]
        same_data = [i + 1 for i in range(shared) if replay[i].get('data_sha256') == new[i].get('data_sha256')]
        evidence[str(seed)] = {'shared_local_step': shared, 'matched': bool(shared) and not mismatches,
                               'first_mismatches': mismatches[:5], 'same_data_steps': same_data[:5]}
        if mismatches or same_data:
            errors.append(f'seed{seed} paired RNG mismatch or unexpectedly identical data')
    a, b = streams['seed43_replay'], streams['seed44_replay']
    shared = min(len(a), len(b))
    evidence['different_seeds'] = {'shared_local_step': shared,
        'epsilon_streams_differ': bool(shared) and a[shared - 1].get('cumulative_epsilon_sha256') != b[shared - 1].get('cumulative_epsilon_sha256'),
        'augmentation_streams_differ': bool(shared) and a[shared - 1].get('cumulative_augmentation_rng_sha256') != b[shared - 1].get('cumulative_augmentation_rng_sha256')}
    if shared and not all(evidence['different_seeds'][k] for k in ('epsilon_streams_differ', 'augmentation_streams_differ')):
        errors.append('different seeds unexpectedly share a random stream')
    return evidence, errors


def compare_validation(reports, reference, absolute_step, draws=4000, initial=None):
    """All comparisons share target-episode bootstrap draws, fixed donor mapping."""
    import numpy as np
    reference_rows = reference['rows']
    identity = [[r[k] for k in IDENTITY] for r in reference_rows]
    episodes = np.array([r['episode'] for r in reference_rows])
    unique, inverse = np.unique(episodes, return_inverse=True)
    assert len(reference_rows) == 512 and len(unique) == 415
    arrays, summaries = {}, {}
    for arm, report in reports.items():
        assert report['samples'] == 512 and report['absolute_step'] == absolute_step
        assert report['precision'] == 'highest FP32 cached DINO/encoder/VAE/decoder; no autocast; TF32 disabled'
        assert [[r[k] for k in IDENTITY] for r in report['rows']] == identity, arm
    inputs = {**reports, 'official': reference}
    if initial is not None:
        assert [[r[k] for k in IDENTITY] for r in initial['rows']] == identity, 'initial3000'
        inputs['initial3000'] = initial
    for arm, report in inputs.items():
        rows = report['rows']
        real = np.array([r['metrics']['real_visual_mse'] for r in rows], dtype=np.float64)
        values = [real]
        for control in CONTROLS:
            values.append(np.array([r['metrics'][control + '_visual_mse'] for r in rows]) - real)
        arrays[arm] = np.column_stack(values)
        assert np.isfinite(arrays[arm]).all(), arm
        summaries[arm] = {key: float(value) for key, value in zip(METRICS, arrays[arm].mean(0))}
    counts = np.bincount(inverse)
    picks = np.random.default_rng(1942).integers(len(unique), size=(draws, len(unique)))
    denominator = counts[picks].sum(1)
    def intervals(values):
        result = {}
        for column, metric in enumerate(METRICS):
            v = values[:, column]
            bootstrap = np.bincount(inverse, weights=v)[picks].sum(1) / denominator
            result[metric] = {'estimate': float(v.mean()), 'ci95': np.quantile(bootstrap, [.025, .975]).tolist()}
        return result
    contrasts = {str(seed): arrays[f'seed{seed}_new'] - arrays[f'seed{seed}_replay'] for seed in (43, 44)}
    return {'absolute_step': absolute_step, 'local_step': absolute_step - 3000, 'samples': 512, 'episodes': 415,
        'bootstrap_draws': draws, 'arm_and_official_estimates': summaries,
        'paired_new_minus_replay': {seed: intervals(v) for seed, v in contrasts.items()},
        'two_seed_average_new_minus_replay': intervals(sum(contrasts.values()) / 2),
        'paired_arm_minus_initial3000': {arm: intervals(arrays[arm] - arrays['initial3000']) for arm in ARMS} if initial is not None else None,
        'sign': 'Negative real MSE contrast favors new pairs; positive control-minus-real contrast means a larger z intervention gap with new pairs.',
        'initial_contrast_sign': 'For arm-minus-initial3000, negative real MSE means improvement after additional updates; positive control-minus-real means a larger z intervention gap. Replay-versus-start measures added updates without new episode/base pairs.',
        'scope': 'Conditional on these two fitted continuations from one initial checkpoint, fixed512 validation pairs and donors. Episode bootstrap does not estimate general training-seed variance or causal action semantics.',
        'identity_sha256': hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()}


def snapshot(group, output, completed, draws):
    errors, models, streams, protocols = [], {}, {}, {}
    launch = read_json(group / 'training_launch.json')
    if launch is None:
        return {'phase': 'waiting_for_training_launch', 'at': datetime.now(timezone.utc).isoformat()}, False
    jobs = {job['arm']: job for job in launch['jobs']}
    assert set(jobs) == set(ARMS)
    manifest = read_json(group / 'manifest.json')
    assert manifest and digest(group / 'manifest.json') == launch['manifest_sha256']
    for arm in ARMS:
        job = jobs[arm]; run = Path(job['output']).resolve()
        assert run.parent == group and run.name == arm
        progress, summary, protocol = (read_json(run / p) for p in ('progress.json', 'run_summary.json', 'protocol.json'))
        protocols[arm] = protocol
        streams[arm], issues = read_stream(run / 'training_stream.jsonl')
        errors.extend(arm + ': ' + issue for issue in issues)
        process = process_status(job)
        complete = bool(summary and summary.get('phase') == 'complete' and summary.get('absolute_step') == 6000
                        and (run / 'checkpoints/final.ckpt').is_file())
        if not complete and not (process['alive'] and process['identity_matches']):
            errors.append(arm + ': process is absent/reused without a complete6000 summary')
        if protocol:
            if protocol['script_sha256'] != launch['runner_sha256'] or protocol['manifest_sha256'] != launch['manifest_sha256']:
                errors.append(arm + ': source or manifest hash differs from launch')
        elif streams[arm]:
            errors.append(arm + ': stream exists without protocol/fingerprints')
        models[arm] = {'pid': job['pid'], 'process': process, 'complete': complete,
                       'stream_local_step': len(streams[arm]), 'progress': progress,
                       'summary_phase': summary.get('phase') if summary else None}
    paired, issues = paired_streams(streams); errors.extend(issues)
    initial_checks = {}
    for key in ('initial_model_sha256', 'initial_optimizer_sha256'):
        hashes = {arm: p.get(key) if p else None for arm, p in protocols.items()}
        initial_checks[key] = {'values': hashes, 'all_four_equal': all(hashes.values()) and len(set(hashes.values())) == 1}
        if len({v for v in hashes.values() if v}) > 1:
            errors.append(key + ': mismatch across arms')
    reference_dir = Path(manifest['source_run']).parent / 'evaluations/final3000-highest-fp32-v2'
    reference = read_json(reference_dir / 'official/results.json')
    assert reference and reference['complete'] and reference['unchanged']
    reference_hash = digest(reference_dir / 'official/results.json')
    initial_reference = read_json(reference_dir / 'kl_mse/results.json')
    assert initial_reference and initial_reference['complete'] and initial_reference['unchanged']
    assert initial_reference['checkpoint_sha256'] == manifest['checkpoint']['sha256']
    initial_hash = digest(reference_dir / 'kl_mse/results.json')
    for step in range(3500, 6001, 500):
        paths = {arm: Path(job['output']) / 'evaluations' / f'step{step:06d}' / 'results.json' for arm, job in jobs.items()}
        reports = {arm: read_json(path) for arm, path in paths.items()}
        if not all(reports.values()):
            continue
        hashes = {arm: digest(path) for arm, path in paths.items()}
        cache_key = {'validation_sha256': hashes, 'official_sha256': reference_hash, 'initial3000_sha256': initial_hash}
        if step in completed and completed[step]['source_hashes'] == cache_key:
            continue
        try:
            assert all(len(streams[arm]) >= step - 3000 for arm in ARMS), 'validation ahead of training stream'
            assert not errors and all(v['all_four_equal'] for v in initial_checks.values()), 'pairing or initial state check failed'
            result = compare_validation(reports, reference, step, draws, initial=initial_reference)
            result['source_hashes'] = cache_key
            completed[step] = result
            write_json(output / f'comparison-step{step:06d}.json', result)
        except (AssertionError, KeyError, ValueError) as exc:
            errors.append(f'validation{step}: {type(exc).__name__}: {exc}')
    all_complete = all(v['complete'] for v in models.values())
    complete = all_complete and not errors and 6000 in completed
    if all_complete and not complete:
        errors.append('all training summaries complete but final comparison is not validated')
    terminal = all_complete or bool(errors)
    status = {'at': datetime.now(timezone.utc).isoformat(), 'phase': 'complete' if complete else 'attention_needed' if errors else 'running',
        'complete': complete, 'arms': models, 'initial_states': initial_checks, 'paired_random_streams': paired,
        'errors': errors, 'available_comparison_steps': sorted(completed), 'source_launch_sha256': digest(group / 'training_launch.json'),
        'watcher_sha256': digest(__file__), 'scope': 'CPU observation only; no process control or training writes.'}
    write_json(output / 'comparisons-live.json', {'complete': complete, 'steps': {str(k): v for k, v in sorted(completed.items())}})
    if complete:
        write_json(output / 'comparisons-final.json', {'complete': True, 'steps': {str(k): v for k, v in sorted(completed.items())}})
    return status, terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--group', type=Path, required=True)
    parser.add_argument('--watch', action='store_true')
    parser.add_argument('--poll-seconds', type=float, default=30)
    parser.add_argument('--bootstrap', type=int, default=4000)
    args = parser.parse_args(); group = args.group.resolve()
    assert group.is_relative_to(RUNS.resolve()) and 1 <= args.poll_seconds <= 60 and args.bootstrap > 0
    output = group / 'paired-analysis'; completed = {}
    while True:
        status, terminal = snapshot(group, output, completed, args.bootstrap)
        write_json(output / 'status.json', status)
        print(json.dumps({'phase': status['phase'], 'steps': status.get('available_comparison_steps', []), 'errors': status.get('errors', [])}), flush=True)
        if not args.watch or terminal:
            break
        time.sleep(args.poll_seconds)


if __name__ == '__main__':
    main()
