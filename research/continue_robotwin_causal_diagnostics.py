"""One bounded continuation of the authorized B4 latent diagnostics.

Prepare/review an immutable contract first. --run waits on CPU for the original
four runs and their pinned frozen FP32 evaluator, then either stops for review
or starts exactly one four-arm causal group. No main training, retries, process
termination, official-init arm, or further automatic experimental rounds.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SSD = Path('/media/sadasue/SSD1')
RUNS = SSD / 'foveated-world-models/runs'
OLD = RUNS / 'robotwin-loss-ablation-b4-20261114'
EVAL = OLD / 'evaluations/final3000-highest-fp32'
NEW = RUNS / 'robotwin-causal-v2-b4-20261114'
PREFLIGHT = RUNS / 'robotwin-causal-v2-preflight-20261114'
OLD_ARMS = ('baseline', 'kl_only', 'mse_only', 'kl_mse')
ARMS = ('baseline_replay', 'beta_zero', 'mu_only', 'g_zero')
RUNNER = ROOT / 'research/train_robotwin_stage1_causal_v2.py'
RUNNER_SHA = 'e1d7db1efca25cdff88a7299e7f5cd31c016b0c11a661fd212c73df67647aa22'
EVALUATOR_SHA = 'e9c1282f15e01832d4c12a01b8e74fdf5229bbf8859d5e48e71c1a84b11b5b01'
GIB = 1024**3
POLICY = {'max_groups': 1, 'arms': list(ARMS), 'batch_size': 4, 'steps': 3000,
          'seed': 42, 'warmup_steps': 1000, 'relative_candidate_threshold': .01,
          'ci_lower_must_exceed': 0., 'initial_ram_available_bytes': 90 * GIB,
          'ssd_free_bytes': 70 * GIB, 'estimated_ram_per_pending_arm_bytes': 20 * GIB,
          'ram_reserve_bytes': 10 * GIB, 'main_training_allowed': False,
          'official_init_allowed': False, 'automatic_restarts_allowed': False}
LIMIT = ('The 1%/positive-CI rule only allocates the next bounded diagnostic budget. '
         'It is not a success criterion or proof that z is useful/useless. It uses '
         'the overall512-pair group, one seed and fixed intervention/bootstrap '
         'choices, without multiple-comparison correction. All causal3000 '
         'outcomes require subsequent review; no additional round is automatic.')


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + f'.{os.getpid()}.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    tmp.replace(path)
    directory = os.open(path.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def proc(pid):
    try:
        root = Path(f'/proc/{pid}')
        stat = (root / 'stat').read_text().split(') ', 1)[1].split()
        return {'pid': int(pid), 'state': stat[0], 'start_ticks': stat[19],
                'command': (root / 'cmdline').read_bytes().decode().strip('\0').split('\0')}
    except (FileNotFoundError, ProcessLookupError):
        return None


def process_active(identity):
    current = proc(identity['pid'])
    if not current or current['state'] == 'Z':
        return False
    if current['start_ticks'] != identity['start_ticks']:
        return False  # PID was reused; matching output jobs are separately scanned.
    if current['command'] != identity['command']:
        raise RuntimeError(f'Process command changed without PID/start-time change: {identity["pid"]}')
    return True


def conflicting_original_processes(contract):
    identities = contract['original_processes']
    active = [p for p in identities if process_active(p)]
    outputs = set(contract['original_output_paths'])
    scripts = {str(ROOT / 'research/train_robotwin_stage1.py'),
               str(ROOT / 'research/evaluate_robotwin_loss_ablation_fp32.py')}
    # Also catch still-alive loader workers or a replacement with a reused PID.
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        p = proc(int(entry.name))
        if p and p['state'] != 'Z' and scripts.intersection(p['command']) and outputs.intersection(p['command']):
            if not any(q['pid'] == p['pid'] for q in active):
                active.append(p)
    return active


def resources():
    mount = json.loads(subprocess.check_output(
        ['findmnt', '-J', '-T', str(RUNS), '-o', 'TARGET,SOURCE,FSTYPE'], text=True))['filesystems'][0]
    if mount['target'] != str(SSD) or not RUNS.resolve().is_relative_to(SSD):
        raise RuntimeError('SSD1 must be mounted at the declared physical output path')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    return {'mount': mount, 'ram_available_bytes': available,
            'ssd_free_bytes': shutil.disk_usage(RUNS).free}


def gpu_telemetry():
    return {'devices': subprocess.check_output(['nvidia-smi',
                '--query-gpu=name,memory.total,memory.used,utilization.gpu',
                '--format=csv,noheader,nounits'], text=True).strip(),
            'compute_processes': subprocess.check_output(['nvidia-smi',
                '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'], text=True).strip(),
            'headroom_basis': 'GB10 unified memory: use actual host MemAvailable budget; record GPU allocation/load without rejecting merely because another GPU process exists.'}


def read_stream(path):
    if not path.exists():
        return []
    data = path.read_bytes()
    lines = data.splitlines(keepends=True)
    result = []
    for i, line in enumerate(lines):
        if i == len(lines) - 1 and not line.endswith(b'\n'):
            break  # Writer may currently be appending this final record.
        result.append(json.loads(line))
    if [r['step'] for r in result] != list(range(1, len(result) + 1)) or len(result) > 3000:
        raise RuntimeError(f'Noncontiguous or oversized training stream: {path}')
    return result


def validate_configs():
    import yaml
    reports = read(ROOT / 'research/results/robotwin-z-underuse-20261114/causal-v2-preflight-summary.json')
    configs = {}
    for arm in ARMS:
        path = ROOT / f'configs/robotwin_causal_ablation_20261114/{arm}.yaml'
        cfg = yaml.safe_load(path.read_text())
        executed = yaml.safe_load((PREFLIGHT / arm / 'requested_config.yaml').read_text())
        actual, reference = deepcopy(cfg), deepcopy(executed)
        actual.pop('diagnostic'); reference.pop('diagnostic')
        if actual != reference:
            raise RuntimeError(f'Effective configuration differs from passed preflight: {arm}')
        beta, aux, sampling = {'baseline_replay': (5e-5, 1., 'sampled'),
                              'beta_zero': (0., 1., 'sampled'), 'mu_only': (5e-5, 1., 'mu_only'),
                              'g_zero': (5e-5, 0., 'sampled')}[arm]
        factors = {'image_loss': 'smooth_l1', 'kl_beta': beta, 'lambda_aux': aux,
                   'posterior_sampling': sampling, 'initialization': 'fresh'}
        assert cfg['diagnostic']['protocol'] == 'robotwin_causal_ablation_v2_20261114'
        assert cfg['diagnostic']['arm'] == arm and cfg['diagnostic']['factors'] == factors
        assert cfg['data']['batch_size'] == 4 and cfg['data']['in_order'] is True
        assert cfg['trainer']['max_steps'] == 3000 and cfg['model']['warmup_steps'] == 1000
        assert cfg['trainer']['accumulate_grad_batches'] == 1 and cfg['seed'] == 42
        assert cfg['trainer']['precision'] == 'bf16-mixed' and cfg['trainer']['gradient_clip_val'] == 1.
        assert cfg['trainer']['val_check_interval'] == 500 and cfg['trainer']['limit_val_batches'] == 128
        pre = reports['experiments'][arm]
        assert pre['passed'] and pre['steps'] == 6 and pre['observed_factors']['runner_sha256'] == RUNNER_SHA
        assert pre['prepared_config_sha256'] == digest(path)
        resolved = yaml.safe_load((PREFLIGHT / arm / 'resolved_config.yaml').read_text())
        assert float(resolved['model']['optimizer']['init_args']['lr']) == 3e-4
        configs[arm] = {'path': str(path), 'sha256': digest(path), 'factors': factors}
    return configs


def prepare_contract(path):
    if Path(path).exists():
        raise FileExistsError('Contract is immutable: choose a new versioned contract path')
    assert digest(RUNNER) == RUNNER_SHA
    assert digest(ROOT / 'research/evaluate_robotwin_loss_ablation_fp32.py') == EVALUATOR_SHA
    cfg = validate_configs()
    launch = read(OLD / 'training_launch.json')
    evaluation = read(EVAL / 'evaluator-launch.json')
    assert tuple(job['arm'] for job in launch['jobs']) == OLD_ARMS
    assert evaluation['pid'] == 126286 and evaluation['script_sha256'] == EVALUATOR_SHA
    processes = []
    for job in [*launch['jobs'], evaluation]:
        p = proc(job['pid'])
        if not p or p['state'] == 'Z' or p['command'] != job['command']:
            raise RuntimeError('Prepare the contract while the declared original/evaluator identities are observable')
        processes.append(p)
    pins = read(EVAL / 'launch-pins.json')
    assert pins['script_sha256'] == EVALUATOR_SHA
    baseline_contract = read(OLD / 'baseline/data_contract.json')
    source = Path(evaluation['configuration_source_pins']['source'])
    files = [Path(__file__), RUNNER, ROOT / 'research/train_robotwin_stage1.py',
             ROOT / 'research/evaluate_robotwin_loss_ablation_fp32.py',
             ROOT / 'research/results/robotwin-z-underuse-20261114/causal-v2-preflight-summary.json',
             OLD / 'training_launch.json', EVAL / 'evaluator-launch.json', EVAL / 'launch-pins.json',
             Path(sys.executable).resolve(), *(Path(c['path']) for c in cfg.values())]
    files += list((source / 'latent_action_model').rglob('*.py'))
    files += [source / 'latent_action_model/config/dino_large_vae.yaml']
    for arm in OLD_ARMS:
        files += [OLD / arm / name for name in ('requested_config.yaml', 'resolved_config.yaml',
                  'data_contract.json', 'episode_split.json', 'initial_state_fingerprint.json',
                  'z_validation_step000500.json', 'train_robotwin_stage1.py')]
    data = Path(baseline_contract['data_root'])
    files += [data / 'qfix/audit.json', data / 'meta/info.json']
    files += [data / name for name in baseline_contract['episode_metadata_sha256']]
    mount = resources()['mount']
    result = {'version': 1, 'created_at': now(), 'policy': POLICY, 'limitations': LIMIT,
              'coordinator': str(Path(__file__).resolve()), 'output': str(NEW),
              'original_output_paths': [str(OLD / arm) for arm in OLD_ARMS] + [str(EVAL)],
              'original_processes': processes, 'original_runs': {a: str(OLD / a) for a in OLD_ARMS},
              'evaluation_directory': str(EVAL), 'evaluation_pins': pins,
              'runner': str(RUNNER), 'runner_sha256': RUNNER_SHA, 'configs': cfg,
              'expected_initial_sha256': read(PREFLIGHT / 'baseline_replay/initial_state_fingerprint.json')['sha256'],
              'pinned_files': {str(p.resolve()): digest(p) for p in sorted(set(files))},
              'pinned_mount': mount, 'python': str(ROOT / '.venv/bin/python'),
              'training_source': str(source), 'source_commit': pins['source_commit'],
              'source_patch_sha256': pins['local_patch_sha256'],
              'validation': 'All four selected arms passed six-update GPU preflight; candidate rule has synthetic tests.'}
    write(path, result)
    return result


def validate_contract(contract):
    if contract['version'] != 1 or contract['policy'] != POLICY or tuple(contract['configs']) != ARMS:
        raise RuntimeError('Unsupported or changed launch contract')
    assert Path(contract['output']) == NEW and Path(contract['runner']) == RUNNER
    assert contract['runner_sha256'] == RUNNER_SHA
    for path, expected in contract['pinned_files'].items():
        if digest(path) != expected:
            raise RuntimeError(f'Pinned file changed: {path}')
    if validate_configs() != contract['configs']:
        raise RuntimeError('Config declaration changed')
    source = contract['training_source']
    assert subprocess.check_output(['git', '-C', source, 'rev-parse', 'HEAD'], text=True).strip() == contract['source_commit']
    patch = subprocess.check_output(['git', '-C', source, 'diff', '--binary'])
    assert hashlib.sha256(patch).hexdigest() == contract['source_patch_sha256']
    assert resources()['mount'] == contract['pinned_mount']


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'Expected finite numeric evidence: {value!r}')
    return float(value)


def scheduling_decision(report):
    """Pure allocation rule. Invalid evidence raises instead of becoming negative."""
    if report.get('complete') is not True:
        raise ValueError('Incomplete FP32 report')
    group = report['groups']['all']
    if group['samples'] != 512 or group['episodes'] != 415:
        raise ValueError('Expected the pinned512-pair/415-episode overall group')
    evidence, candidates = {}, []
    for arm in OLD_ARMS:
        metrics = group['arms'][arm]
        real = finite(metrics['real_visual_mse']['estimate'])
        if real < 0:
            raise ValueError('Negative real MSE')
        real_ci = metrics['real_visual_mse']['ci95']
        if not isinstance(real_ci, list) or len(real_ci) != 2:
            raise ValueError('Malformed real-MSE interval')
        real_low, real_high = map(finite, real_ci)
        if real_low < 0 or real_low > real_high:
            raise ValueError('Invalid real-MSE interval')
        evidence[arm] = {'real_visual_mse': real, 'real_visual_mse_ci95': [real_low, real_high], 'controls': {}}
        for control in ('mean', 'swapped'):
            item = metrics[control + '_minus_real_visual_mse']
            estimate = finite(item['estimate'])
            if not isinstance(item['ci95'], list) or len(item['ci95']) != 2:
                raise ValueError('Malformed paired interval')
            low, high = map(finite, item['ci95'])
            if low > high:
                raise ValueError('Reversed paired interval')
            candidate = estimate > .01 * real and low > 0
            evidence[arm]['controls'][control] = {'estimate': estimate, 'ci95': [low, high],
                'threshold_absolute': .01 * real, 'fraction_of_real': estimate / real if real else None,
                'candidate_for_review': candidate}
            if candidate:
                candidates.append({'arm': arm, 'control': control})
    return {'decision': 'candidate_needs_review' if candidates else 'allocate_one_causal_group',
            'candidates': candidates, 'evidence': evidence, 'limitations': LIMIT}


def validate_final_report(contract, report):
    if report.get('complete') is not True:
        raise RuntimeError('FP32 report is incomplete')
    protocol = report['protocol']
    if {key: protocol[key] for key in contract['evaluation_pins']} != contract['evaluation_pins']:
        raise RuntimeError('FP32 report does not match the declared evaluator/source/data pins')
    assert protocol['samples'] == 512 and protocol['checkpoint_step_required'] == 3000
    assert protocol['actual_matmul_precision'] == 'highest' and protocol['tf32_enabled'] is False
    assert protocol['precision'] == 'highest FP32 DINO, encoder, VAE, decoder and state head; no autocast; TF32 disabled'
    assert protocol['batch_size'] == 2 and protocol['bootstrap_draws'] == 4000
    for arm in (*OLD_ARMS, 'official'):
        model = report['models'][arm]
        assert model['complete'] and model['unchanged'] and model['all_parameters_frozen'] and model['no_parameter_gradients']
        assert model['module_hashes_before'] == model['module_hashes_after']
        assert model['strictly_loaded_state_tensors'] == 660
        assert model['native_api_cache_route_parity']['all_close_atol_1e-6_rtol_0']
        assert model['feature_cache_identity_sha256'] == protocol['validation_pair_and_donor_sha256']
        if arm != 'official':
            checkpoint = OLD / arm / 'checkpoints/final.ckpt'
            assert model['global_step'] == 3000 and Path(model['checkpoint']) == checkpoint
            assert digest(checkpoint) == model['checkpoint_sha256']
    pairing = read(EVAL / 'training-pairing-step3000.json')
    assert pairing['status'] == 'paired_snapshot'
    for arm in OLD_ARMS:
        stream = read_stream(OLD / arm / 'training_stream.jsonl')
        assert len(stream) == 3000
        assert read(OLD / arm / 'initial_state_fingerprint.json')['sha256'] == contract['expected_initial_sha256']
        if arm == OLD_ARMS[0]:
            reference = stream[-1]
        else:
            assert all(stream[-1]['cumulative_' + k + '_sha256'] == reference['cumulative_' + k + '_sha256']
                       for k in ('data', 'aug', 'epsilon'))


def original_readiness(contract):
    jobs = read(OLD / 'training_launch.json')['jobs']
    status = {}
    for job in jobs:
        p = OLD / job['arm']
        summary = read(p / 'run_summary.json') if (p / 'run_summary.json').exists() else {}
        complete = summary.get('phase') == 'complete' and summary.get('step') == 3000 and (p / 'checkpoints/final.ckpt').is_file()
        identity = next(x for x in contract['original_processes'] if x['pid'] == job['pid'])
        active = process_active(identity)
        if not active and not complete:
            raise RuntimeError(f'Original arm exited without a complete3000 checkpoint: {job["arm"]}')
        status[job['arm']] = {'complete': complete, 'original_process_active': active}
    watch = read(OLD / 'watch_status.json') if (OLD / 'watch_status.json').exists() else {}
    if watch.get('failed_arms'):
        raise RuntimeError(f'Original watcher reports failures: {watch["failed_arms"]}')
    evaluation_exists = (EVAL / 'results.json').exists()
    evaluator = contract['original_processes'][-1]
    if not process_active(evaluator) and not evaluation_exists:
        raise RuntimeError('Pinned evaluator exited without final results')
    active = conflicting_original_processes(contract)
    return {'arms': status, 'evaluation_results_exist': evaluation_exists,
            'active_original_or_evaluator_processes': active,
            'ready': all(v['complete'] for v in status.values()) and evaluation_exists and not active}


def pairing_status(contract, jobs):
    streams, inits, warnings = {}, {}, []
    for job in jobs:
        run = Path(job['output'])
        stream = read_stream(run / 'training_stream.jsonl')
        streams[job['arm']] = stream
        init_path = run / 'initial_state_fingerprint.json'
        if stream and not init_path.exists():
            warnings.append(f'missing_initial_weight_evidence:{job["arm"]}')
        if init_path.exists():
            inits[job['arm']] = read(init_path)['sha256']
            if inits[job['arm']] != contract['expected_initial_sha256']:
                warnings.append(f'initial_weight_mismatch:{job["arm"]}')
        observed = run / 'observed_factors.json'
        if stream and not observed.exists():
            warnings.append(f'missing_observed_factor_evidence:{job["arm"]}')
        if observed.exists():
            actual = read(observed)
            expected = contract['configs'][job['arm']]['factors']
            if {k: actual[k] for k in expected} != expected or actual['runner_sha256'] != RUNNER_SHA:
                warnings.append(f'observed_factor_mismatch:{job["arm"]}')
    common = min((len(s) for s in streams.values()), default=0)
    if common:
        for key in ('data', 'aug', 'epsilon'):
            field = 'cumulative_' + key + '_sha256'
            if len({s[common - 1][field] for s in streams.values()}) != 1:
                warnings.append(f'cumulative_{key}_mismatch_at_{common}')
            original = read_stream(OLD / 'baseline/training_stream.jsonl')[common - 1]
            if any(s[common - 1][field] != original[field] for s in streams.values()):
                warnings.append(f'original_baseline_{key}_mismatch_at_{common}')
    return {'common_step': common, 'initial_sha256': inits, 'warnings': warnings,
            'complete_group_present': len(jobs) == 4,
            'streams_match': not warnings if common else None}


def launch_once(contract, contract_path, decision):
    # This file survives coordinator death. An uncertain partial launch must
    # be reviewed, never automatically resumed or repeated.
    intent = NEW / 'launch-intent.json'
    fd = os.open(intent, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as stream:
        json.dump({'created_at': now(), 'contract_sha256': digest(contract_path),
                   'maximum_arms': list(ARMS), 'decision': decision}, stream, indent=2)
        stream.flush(); os.fsync(stream.fileno())
    directory = os.open(NEW, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)
    jobs, children = [], {}
    manifest = {'created_at': now(), 'phase': 'launching', 'contract_sha256': digest(contract_path),
                'decision': decision, 'jobs': jobs, 'main_training_started': False}
    write(NEW / 'training_launch.json', manifest)
    failure = None
    for index, arm in enumerate(ARMS):
        try:
            validate_contract(contract)
            if digest(EVAL / 'results.json') != decision['evaluated_results_sha256']:
                raise RuntimeError('The FP32 artifact changed after the budget decision')
            r = resources()
            r['gpu'] = gpu_telemetry()
        except Exception as exc:
            failure = {'reason': 'prelaunch_contract_or_resource_check_failed', 'arm': arm, 'error': repr(exc)}
            break
        required = POLICY['ram_reserve_bytes'] + (4 - index) * POLICY['estimated_ram_per_pending_arm_bytes']
        if r['ram_available_bytes'] < required or r['ssd_free_bytes'] < POLICY['ssd_free_bytes']:
            failure = {'reason': 'insufficient_resource_headroom_before_next_arm', 'arm': arm,
                       'required_ram_bytes': required, 'observed': r}
            break
        if conflicting_original_processes(contract):
            failure = {'reason': 'original_or_evaluator_process_reappeared', 'arm': arm}
            break
        output, log = NEW / arm, NEW / f'{arm}.log'
        if output.exists() or log.exists():
            failure = {'reason': 'output_already_exists', 'arm': arm}
            break
        command = [contract['python'], '-u', str(RUNNER), '--config', contract['configs'][arm]['path'], '--output', str(output)]
        # Record the exact pending command durably before Popen.
        manifest['pending_arm'] = {'arm': arm, 'command': command, 'at': now(), 'resources': r}
        write(NEW / 'training_launch.json', manifest)
        try:
            with log.open('xb') as stream:
                child = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        except Exception as exc:
            failure = {'reason': 'spawn_failed', 'arm': arm, 'error': repr(exc)}
            break
        identity = proc(child.pid)
        if identity is None:
            identity = {'pid': child.pid, 'start_ticks': None, 'command': command, 'state': 'exited_early'}
        job = {'arm': arm, **identity, 'output': str(output), 'log': str(log),
               'launched_at': now(), 'command': command}
        jobs.append(job); children[arm] = child
        manifest.pop('pending_arm', None)
        write(NEW / 'training_launch.json', manifest)
        # A bounded startup handshake catches invalid factors/first-batch failure
        # before committing the remaining resource budget.
        deadline = time.monotonic() + 180
        while (not read_stream(output / 'training_stream.jsonl')
               or not (output / 'initial_state_fingerprint.json').exists()
               or not (output / 'observed_factors.json').exists()) and child.poll() is None and time.monotonic() < deadline:
            time.sleep(2)
        try:
            pair = pairing_status(contract, jobs)
        except Exception as exc:
            failure = {'reason': 'startup_control_evidence_invalid', 'arm': arm, 'error': repr(exc)}
            break
        if child.poll() is not None or not pair['common_step'] or pair['warnings']:
            failure = {'reason': 'startup_or_control_check_failed', 'arm': arm,
                       'returncode': child.poll(), 'pairing': pair}
            break
    manifest.update(phase='running' if len(jobs) == 4 and not failure else 'partial_launch_needs_review',
                    launch_failure=failure, launch_finished_at=now())
    write(NEW / 'training_launch.json', manifest)
    return manifest, children


def monitor(contract, manifest, children):
    alerts = []
    while True:
        status = {}
        for job in manifest['jobs']:
            run = Path(job['output'])
            summary = read(run / 'run_summary.json') if (run / 'run_summary.json').exists() else {}
            complete = summary.get('phase') == 'complete' and summary.get('step') == 3000 and (run / 'checkpoints/final.ckpt').is_file()
            child = children[job['arm']]
            status[job['arm']] = {'pid': job['pid'], 'returncode': child.poll(), 'complete3000': complete,
                                  'last_step': len(read_stream(run / 'training_stream.jsonl'))}
            val = run / 'z_validation_latest.json'
            if val.exists():
                validation = read(val)
                status[job['arm']]['periodic_BF16_validation'] = {
                    'step': validation['step'], 'samples': validation['samples'],
                    'mu_variance_sum': validation['mu_variance_sum'],
                    'metrics': {k: validation['metrics'][k] for k in ('real_visual_mse',
                                'mean_minus_real_visual_mse', 'swapped_minus_real_visual_mse')}}
        try:
            pair = pairing_status(contract, manifest['jobs'])
        except Exception as exc:
            pair = {'common_step': None, 'warnings': [f'pairing_evidence_invalid:{exc!r}']}
        alerts = sorted(set(alerts + pair['warnings']))
        done = all(row['returncode'] is not None for row in status.values())
        if done:
            complete = len(status) == 4 and all(s['complete3000'] and s['returncode'] == 0 for s in status.values())
            phase = 'bounded_group_complete_needs_FP32_review' if complete and not alerts else 'bounded_group_incomplete_or_unpaired_needs_review'
        else:
            phase = 'monitoring_partial_group' if manifest['launch_failure'] else 'monitoring_bounded_group'
        report = {'at': now(), 'phase': phase, 'arms': status, 'pairing': pair,
                  'persistent_alerts': alerts, 'launch_failure': manifest['launch_failure'],
                  'resources': resources(), 'no_more_groups_will_be_launched': True,
                  'main_training_started': False, 'limitations': LIMIT}
        write(NEW / 'coordinator_status.json', report)
        if done:
            write(NEW / 'bounded_group_final.json', report)
            return report
        time.sleep(30)


def run(contract, contract_path):
    NEW.mkdir(parents=True, exist_ok=True)
    if (NEW / 'launch-intent.json').exists():
        raise RuntimeError('Durable launch intent already exists; automatic relaunch is prohibited')
    lock = os.open(NEW / 'coordinator.lock', os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(lock, 'w') as stream:
        json.dump({'identity': proc(os.getpid()), 'contract_sha256': digest(contract_path)}, stream)
        stream.flush(); os.fsync(stream.fileno())
    while True:
        validate_contract(contract)
        readiness = original_readiness(contract)
        state = {'at': now(), 'phase': 'waiting_for_original_FP32_and_process_exit',
                 'readiness': readiness, 'resources': resources(), 'main_training_started': False}
        write(NEW / 'coordinator_status.json', state)
        if readiness['ready']:
            break
        time.sleep(30)
    results = read(EVAL / 'results.json')
    validate_final_report(contract, results)
    decision = scheduling_decision(results)
    decision.update(at=now(), evaluated_results_sha256=digest(EVAL / 'results.json'),
                    contract_sha256=digest(contract_path))
    write(NEW / 'budget_decision.json', decision)
    if decision['decision'] == 'candidate_needs_review':
        final = {'at': now(), 'phase': 'candidate_needs_review', 'decision': decision,
                 'training_jobs_launched': 0, 'main_training_started': False}
        write(NEW / 'coordinator_status.json', final)
        return final
    while True:
        validate_contract(contract)
        r = resources()
        # nvidia-smi queries the driver only: this waiter never imports torch/CUDA.
        r['gpu'] = gpu_telemetry()
        active = conflicting_original_processes(contract)
        if not active and r['ram_available_bytes'] >= 90 * GIB and r['ssd_free_bytes'] >= 70 * GIB:
            break
        write(NEW / 'coordinator_status.json', {'at': now(), 'phase': 'waiting_for_group_resources',
              'resources': r, 'active_original_or_evaluator_processes': active,
              'decision': decision, 'main_training_started': False})
        time.sleep(30)
    for arm in ARMS:
        if (NEW / arm).exists() or (NEW / f'{arm}.log').exists():
            raise RuntimeError('The bounded output group is not empty; refusing duplicate launch')
    if digest(EVAL / 'results.json') != decision['evaluated_results_sha256']:
        raise RuntimeError('The FP32 artifact changed while waiting for group resources')
    manifest, children = launch_once(contract, contract_path, decision)
    return monitor(contract, manifest, children)


def self_test():
    template = {'complete': True, 'groups': {'all': {'samples': 512, 'episodes': 415, 'arms': {}}}}
    for arm in OLD_ARMS:
        template['groups']['all']['arms'][arm] = {'real_visual_mse': {'estimate': 1., 'ci95': [.9, 1.1]},
            'mean_minus_real_visual_mse': {'estimate': 0., 'ci95': [-.001, .001]},
            'swapped_minus_real_visual_mse': {'estimate': 0., 'ci95': [-.001, .001]}}
    assert scheduling_decision(template)['decision'] == 'allocate_one_causal_group'
    count = 1
    for arm in OLD_ARMS:
        for control in ('mean', 'swapped'):
            x = deepcopy(template)
            x['groups']['all']['arms'][arm][control + '_minus_real_visual_mse'] = {'estimate': .010001, 'ci95': [.000001, .02]}
            assert scheduling_decision(x)['decision'] == 'candidate_needs_review'; count += 1
    for estimate, lower in ((.01, .001), (.02, 0.), (-.02, -.03)):
        x = deepcopy(template)
        x['groups']['all']['arms']['baseline']['mean_minus_real_visual_mse'] = {'estimate': estimate, 'ci95': [lower, .03]}
        assert scheduling_decision(x)['decision'] == 'allocate_one_causal_group'; count += 1
    for invalid in (float('nan'), float('inf'), None, True, '0.02'):
        x = deepcopy(template)
        x['groups']['all']['arms']['baseline']['mean_minus_real_visual_mse']['estimate'] = invalid
        try:
            scheduling_decision(x)
        except (ValueError, TypeError):
            count += 1
        else:
            raise AssertionError('Invalid metric was accepted')
    for mutate in (lambda x: x.update(complete=False),
                   lambda x: x['groups']['all']['arms'].pop('kl_only'),
                   lambda x: x['groups']['all']['arms']['baseline']['real_visual_mse'].update(ci95=[float('nan'), 1.]),
                   lambda x: x['groups']['all']['arms']['baseline']['mean_minus_real_visual_mse'].update(ci95=[.2, -.1])):
        x = deepcopy(template); mutate(x)
        try:
            scheduling_decision(x)
        except (ValueError, KeyError):
            count += 1
        else:
            raise AssertionError('Incomplete evidence was accepted')
    configs = validate_configs()
    return {'passed': True, 'synthetic_gate_cases': count, 'validated_configs': list(configs),
            'GPU_initialized': False, 'training_jobs_launched': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--prepare-contract', type=Path)
    mode.add_argument('--self-test', action='store_true')
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--contract', type=Path)
    args = parser.parse_args()
    if args.prepare_contract:
        result = prepare_contract(args.prepare_contract)
        print(json.dumps({'prepared_contract': str(args.prepare_contract), 'sha256': digest(args.prepare_contract),
                          'pinned_files': len(result['pinned_files']), 'training_jobs_launched': 0}), flush=True)
        return
    if args.self_test:
        print(json.dumps(self_test()), flush=True)
        return
    if not args.contract:
        parser.error('--contract is required with --check-only or --run')
    contract = read(args.contract)
    validate_contract(contract)
    if args.check_only:
        print(json.dumps({'contract_valid': True, 'readiness': original_readiness(contract),
                          'resources': resources(), 'training_jobs_launched': 0}), flush=True)
        return
    try:
        final = run(contract, args.contract)
        print(json.dumps(final), flush=True)
    except Exception as exc:
        # Jobs already spawned are left untouched. Durable intent prevents retries.
        write(NEW / 'coordinator_error.json', {'at': now(), 'error': repr(exc),
              'main_training_started': False, 'automatic_relaunch_prohibited': True})
        raise


if __name__ == '__main__':
    main()
