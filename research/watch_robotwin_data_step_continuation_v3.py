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
_DIGEST_CACHE = {}


def digest(path):
    path = Path(path)
    stat = path.stat()
    key = (str(path.resolve()), stat.st_ino, stat.st_size, stat.st_mtime_ns)
    if key not in _DIGEST_CACHE:
        h = hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
                h.update(chunk)
        _DIGEST_CACHE[key] = h.hexdigest()
    return _DIGEST_CACHE[key]


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


def read_stream(path, start=1):
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
    if [r.get('local_step') for r in rows] != list(range(start, start + len(rows))):
        errors.append(f'stream local steps are not contiguous from {start}')
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


def cumulative_checks(rows):
    errors, hashes = [], {key: hashlib.sha256() for key in ('data', 'epsilon', 'augmentation_rng')}
    for row in rows:
        for key, h in hashes.items():
            try:
                h.update(bytes.fromhex(row[key + '_sha256']))
                assert row['cumulative_' + key + '_sha256'] == h.hexdigest()
            except (KeyError, ValueError, AssertionError):
                errors.append(f"local{row.get('local_step')} invalid cumulative {key} hash")
                return errors
    return errors


def recovery_segments(old_run, run, protocol):
    """Keep only committed old1..500, then newly executed501 onward."""
    old, errors = read_stream(old_run / 'training_stream.jsonl')
    new, issues = read_stream(run / 'training_stream.jsonl', start=501)
    errors += issues
    replay, issues = read_stream(run / 'replay_verified_prefix.jsonl')
    errors += issues
    if len(old) < 501:
        errors.append('historical stream lacks committed500 prefix and501 boundary')
    prefix = old[:500]
    if len(replay) > 500:
        errors.append('reconstructed prefix exceeds500')
    historical_keys = ('pairs', 'data_sha256', 'epsilon_sha256', 'augmentation_rng_before_sha256',
                       'augmentation_rng_after_sha256', 'augmentation_rng_sha256')
    for i, row in enumerate(replay[:500]):
        if i >= len(prefix) or any(row.get(k) is None or row[k] != prefix[i].get(k) for k in historical_keys):
            errors.append(f'reconstructed prefix differs at local{i + 1}')
            break
    merged = prefix + new
    errors += cumulative_checks(merged)
    proof = read_json(run / 'resume_replay_verification.json')
    resume_state = {'verified': False, 'replayed_batches': len(replay),
                    'original_uncommitted_steps_ignored': max(0, len(old) - 500),
                    'committed_original_prefix_steps': len(prefix), 'new_segment_steps': len(new)}
    if proof and protocol:
        conditions = {
            'prefix500': len(replay) == 500 and proof.get('verified_prefix_batches') == 500,
            'boundary501': len(old) >= 501 and proof.get('boundary_matches_old501') is True
                and proof.get('boundary_next_local_step') == 501
                and proof.get('boundary_cuda_rng_sha256') == old[500].get('augmentation_rng_before_sha256'),
            'own_model': proof.get('model_unchanged_by_replay') is True
                and proof.get('restored_model_sha256') == protocol.get('initial_model_sha256'),
            'own_optimizer': proof.get('optimizer_unchanged_by_replay') is True
                and proof.get('restored_optimizer_sha256') == protocol.get('initial_optimizer_sha256'),
            'own_checkpoint': Path(protocol.get('resume_checkpoint', '')).resolve() == old_run / 'checkpoints/last.ckpt'
                and digest(old_run / 'checkpoints/last.ckpt') == protocol.get('resume_checkpoint_sha256'),
            'resume_offset500': protocol.get('resume_local_step') == 500,
            'historical_cpu_scope': proof.get('cpu_and_pixel_historical_hashes_available') is False,
        }
        resume_state.update(verified=all(conditions.values()), checks=conditions,
            restored_model_sha256=protocol.get('initial_model_sha256'),
            restored_optimizer_sha256=protocol.get('initial_optimizer_sha256'),
            resume_checkpoint_sha256=protocol.get('resume_checkpoint_sha256'))
        if not all(conditions.values()):
            errors.append('resume state or replay boundary verification failed')
    elif new:
        errors.append('new updates exist without replay proof and protocol')
    return merged, replay + new, resume_state, errors


def paired_cpu_streams(streams):
    evidence, errors = {}, []
    keys = ('cpu_rng_before_augmentation_sha256', 'cpu_rng_after_augmentation_sha256')
    for seed in (43, 44):
        a, b = streams[f'seed{seed}_replay'], streams[f'seed{seed}_new']
        shared = min(len(a), len(b))
        bad = [(i + 1, k) for i in range(shared) for k in keys if a[i].get(k) is None or a[i].get(k) != b[i].get(k)]
        evidence[str(seed)] = {'shared_local_step': shared, 'matched': bool(shared) and not bad,
                              'first_mismatches': bad[:5]}
        if bad:
            errors.append(f'seed{seed} reconstructed/recovered CPU augmentation RNG mismatch')
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
    errors, models, streams, protocols, original_protocols, cpu_streams, resume_states = [], {}, {}, {}, {}, {}, {}
    launch_path = group / 'recovery-v2/recovery_training_launch.json'
    launch = read_json(launch_path)
    if launch is None:
        return {'phase': 'waiting_for_recovery_launch', 'at': datetime.now(timezone.utc).isoformat()}, False
    jobs = {job['arm']: job for job in launch['jobs']}
    assert set(jobs) == set(ARMS)
    old_launch_path = group / 'training_launch.json'
    old_launch = read_json(old_launch_path)
    old_jobs = {job['arm']: job for job in old_launch['jobs']}
    assert set(old_jobs) == set(ARMS)
    manifest = read_json(group / 'manifest.json')
    assert manifest and digest(group / 'manifest.json') == launch['manifest_sha256'] == old_launch['manifest_sha256']
    for arm in ARMS:
        job = jobs[arm]; run = Path(job['output']).resolve()
        assert run.parent == group / 'recovery-v2' and run.name == arm
        old_run = Path(old_jobs[arm]['output']).resolve()
        assert old_run == group / arm
        progress, summary, protocol = (read_json(run / p) for p in ('progress.json', 'run_summary.json', 'protocol.json'))
        protocols[arm] = protocol
        original_protocols[arm] = read_json(old_run / 'protocol.json')
        assert original_protocols[arm]['script_sha256'] == old_launch['runner_sha256']
        assert original_protocols[arm]['manifest_sha256'] == old_launch['manifest_sha256']
        streams[arm], cpu_streams[arm], resume_states[arm], issues = recovery_segments(old_run, run, protocol)
        errors.extend(arm + ': ' + issue for issue in issues)
        process = process_status(job)
        complete = bool(summary and summary.get('phase') == 'complete' and summary.get('absolute_step') == 6000
                        and (run / 'checkpoints/final.ckpt').is_file())
        if not complete and not (process['alive'] and process['identity_matches']):
            errors.append(arm + ': process is absent/reused without a complete6000 summary')
        if protocol:
            if protocol['script_sha256'] != launch['runner_sha256'] or protocol['manifest_sha256'] != launch['manifest_sha256']:
                errors.append(arm + ': source or manifest hash differs from launch')
            if protocol.get('previous_protocol') != original_protocols[arm] or Path(protocol.get('resume_run', '')).resolve() != old_run:
                errors.append(arm + ': recovery original-protocol/run binding failed')
            source_copy = run / Path(job['command'][2]).name
            if (source_copy.is_file() and digest(source_copy) != launch['runner_sha256']) or (len(streams[arm]) > 500 and not source_copy.is_file()):
                errors.append(arm + ': frozen runner copy hash differs from launch')
        elif len(streams[arm]) > 500:
            errors.append(arm + ': stream exists without protocol/fingerprints')
        models[arm] = {'pid': job['pid'], 'process': process, 'complete': complete,
                       'stream_local_step': len(streams[arm]), 'progress': progress,
                       'summary_phase': summary.get('phase') if summary else None}
    paired, issues = paired_streams(streams); errors.extend(issues)
    paired_cpu, issues = paired_cpu_streams(cpu_streams); errors.extend(issues)
    initial_checks = {}
    for key in ('initial_model_sha256', 'initial_optimizer_sha256'):
        hashes = {arm: p.get(key) if p else None for arm, p in original_protocols.items()}
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
        selected_jobs = old_jobs if step == 3500 else jobs
        paths = {arm: Path(job['output']) / 'evaluations' / f'step{step:06d}' / 'results.json' for arm, job in selected_jobs.items()}
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
            if step > 3500:
                assert all(v['verified'] for v in resume_states.values()), 'resume proof incomplete'
            result = compare_validation(reports, reference, step, draws, initial=initial_reference)
            result['source_hashes'] = cache_key
            completed[step] = result
            write_json(output / f'comparison-step{step:06d}.json', result)
        except (AssertionError, KeyError, ValueError) as exc:
            errors.append(f'validation{step}: {type(exc).__name__}: {exc}')
    all_complete = all(v['complete'] for v in models.values())
    complete = all_complete and not errors and 6000 in completed and all(v['verified'] for v in resume_states.values())
    if all_complete and not complete:
        errors.append('all training summaries complete but final comparison is not validated')
    terminal = all_complete or bool(errors)
    phase = 'running' if all(v['verified'] for v in resume_states.values()) else 'reconstructing_resume_rng'
    status = {'at': datetime.now(timezone.utc).isoformat(), 'phase': 'complete' if complete else 'attention_needed' if errors else phase,
        'complete': complete, 'arms': models, 'initial_states': initial_checks, 'paired_random_streams': paired,
        'resume_states': resume_states, 'paired_recovered_cpu_augmentation_rng': paired_cpu,
        'errors': errors, 'available_comparison_steps': sorted(completed), 'source_launch_sha256': digest(launch_path),
        'original_launch_sha256': digest(old_launch_path),
        'watcher_sha256': digest(__file__), 'scope': 'CPU observation only; no process control or training writes.',
        'recovery_scope': 'Original3000 initial states are common. Own3500 resumed states may differ and are checked individually. Only committed original1..500 plus recovered501 onward enter analyses. Original501..1000 remain archived evidence and are excluded. CPU RNG equality is verified across recovered paired runs, not directly against unrecorded historical CPU RNG.'}
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
