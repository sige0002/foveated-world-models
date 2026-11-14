"""Read-only FP32 probes for the isolated g-loss onset experiment."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import pickle
import random
import time


def prepare_probe(settings, contract, split):
    """Validate immutable inputs and mmap features; no model/GPU construction."""
    import torch
    from research import evaluate_robotwin_loss_ablation_fp32_v2 as ev
    if ev.digest(Path(ev.__file__)) != settings['evaluator_sha256']:
        raise ValueError('Pinned FP32 evaluator changed')
    original = Path(settings['original_run'])
    source_contract = ev.read_json(original / 'data_contract.json')
    for key, value in source_contract.items():
        if key != 'initialization' and contract.get(key) != value:
            raise ValueError(f'Onset data contract differs from original: {key}')
    assert ev.read_json(original / 'episode_split.json')['ids'] == split
    cache_path = Path(settings['cache'])
    assert ev.digest(cache_path) == settings['cache_sha256']
    cache = torch.load(cache_path, map_location='cpu', weights_only=True, mmap=True)
    assert cache['contract'] == source_contract
    assert cache['precision'] == 'highest FP32; TF32 disabled'
    assert cache['native_api_parity_first_batch']
    assert len(cache['rows']) == 512
    assert all(row['episode'] in set(split['val']) for row in cache['rows'])
    assert torch.equal(cache['delta'], cache['states'][:, 1] - cache['states'][:, 0])
    identity_keys = ('episode', 'base', 'swap_valid', 'donor_episode', 'donor_base')
    reference = [{key: row[key] for key in identity_keys}
                 for row in ev.read_json(original / 'z_validation_step000500.json')['rows']]
    assert cache['reference_identity_sha256'] == ev.json_hash(reference)
    assert all(all(row[key] == value for key, value in ref.items() if key != 'metrics')
               for row, ref in zip(cache['rows'], reference))
    for key in ['states', 'delta', 'emb', 'mask']:
        assert ev.tensor_hash(cache[key]) == cache['tensor_sha256'][key]
    stream = [json.loads(line) for line in (original / 'training_stream.jsonl').read_text().splitlines()]
    assert len(stream) == 3000 and [r['step'] for r in stream] == list(range(1, 3001))
    expected_init = ev.read_json(original / 'initial_state_fingerprint.json')['sha256']
    return {'cache': cache, 'settings': settings, 'expected_stream': stream,
            'expected_initial_sha256': expected_init,
            'input_sha256': {'cache': settings['cache_sha256'], 'evaluator': settings['evaluator_sha256'],
                             'original_stream': ev.digest(original / 'training_stream.jsonl'),
                             'original_initial_fingerprint': ev.digest(original / 'initial_state_fingerprint.json')}}


def geometry(codes):
    import torch
    mu = codes['mu'].double().flatten(1)
    noise = codes['logvar'].double().exp().flatten(1).mean(0)
    centered = mu - mu.mean(0)
    covariance = centered.T @ centered / len(mu)
    eigenvalues = torch.linalg.eigvalsh(covariance).clamp_min(0).flip(0)
    proportions = eigenvalues / eigenvalues.sum()
    sn = torch.linalg.eigvalsh(covariance / (noise[:, None] * noise[None, :]).sqrt()).flip(0)
    delta = mu - codes['current_only_mu'].double().flatten(1)
    return {'samples': len(mu), 'mu_variance_trace': float(eigenvalues.sum()),
            'noise_variance_trace': float(noise.sum()),
            'mu_covariance_eigenvalues_descending': eigenvalues.tolist(),
            'mu_top_variance_fraction': float(proportions[0]),
            'mu_entropy_effective_rank': float((-(proportions * proportions.clamp_min(1e-300).log()).sum()).exp()),
            'noise_normalized_signal_eigenvalues_descending': sn.tolist(),
            'snr_directions_above_one': int((sn > 1).sum()),
            'current_only_displacement_rms_over_centered_mu_rms': float((delta.square().sum(1).mean() / eigenvalues.sum()).sqrt()),
            'scope': 'Population covariance; variance concentration and second-moment SNR, not mathematical rank or information bits.'}


class OnsetProbe:
    def __init__(self, prepared, output, preflight=False):
        self.prepared = prepared
        self.output = Path(output)
        self.preflight = preflight
        self.expected_initial_sha256 = prepared['expected_initial_sha256']
        self.steps = {0, 6} if preflight else {0, 10, 50, 100, 250, 500}
        self.records = []
        self.checked_vision = False

    def verify_stream(self, record):
        expected = self.prepared['expected_stream'][record['step'] - 1]
        for key in ['data_sha256', 'aug_sha256', 'epsilon_sha256']:
            if record[key] != expected[key]:
                raise RuntimeError(f'Original seed42 prefix mismatch: step {record["step"]}, {key}')

    def run(self, model, step):
        if step not in self.steps:
            return
        import numpy as np
        import torch
        from research import evaluate_robotwin_loss_ablation_fp32_v2 as ev
        from research.train_robotwin_stage1 import write_json
        started = time.monotonic()
        samples = 512 if step == 500 and not self.preflight else 64
        cache = self.prepared['cache']
        selected = {key: (value[:samples] if isinstance(value, torch.Tensor) and value.ndim and len(value) == 512 else value)
                    for key, value in cache.items()}
        selected['rows'] = deepcopy(cache['rows'][:samples])
        if samples != 512:
            episodes = np.array([row['episode'] for row in selected['rows']])
            rng = np.random.default_rng(1942)
            for _ in range(10000):
                donors = rng.permutation(samples)
                if (episodes[donors] != episodes).all():
                    break
            else:
                raise RuntimeError('Could not select fixed different-episode prefix donors')
            for i, row in enumerate(selected['rows']):
                row.update(donor_index=int(donors[i]), donor_episode=int(episodes[donors[i]]),
                           donor_base=selected['rows'][int(donors[i])]['base'])
            selected['reference_identity_sha256'] = ev.json_hash(selected['rows'])
        training_flags = [(module, module.training) for module in model.modules()]
        parameter_versions = [(parameter, parameter._version, parameter.requires_grad) for parameter in model.parameters()]
        cpu_state = torch.get_rng_state().clone()
        device = model.device
        gpu_state = torch.cuda.get_rng_state(device).clone()
        py_state, np_state = random.getstate(), np.random.get_state()
        py_hash = hashlib.sha256(pickle.dumps(py_state)).hexdigest()
        np_hash = hashlib.sha256(pickle.dumps(np_state)).hexdigest()
        precision = torch.get_float32_matmul_precision()
        tf32, cudnn_tf32 = torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32
        old_samples = ev.SAMPLES
        destination = self.output / 'evaluations' / f'step{step:06d}'
        try:
            with torch.random.fork_rng(devices=[device.index]):
                torch.set_float32_matmul_precision('highest')
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                assert torch.get_float32_matmul_precision() == 'highest'
                model.eval()
                if not self.checked_vision:
                    h = hashlib.sha256()
                    for key, value in model.lam.vision_encoder.state_dict().items():
                        h.update(key.encode() + b'\0' + bytes.fromhex(ev.tensor_hash(value)))
                    assert h.hexdigest() == cache['vision_encoder_fingerprint']['sha256']
                    self.checked_vision = True
                ev.SAMPLES = samples
                with torch.inference_mode(), torch.autocast('cuda', enabled=False):
                    report = ev.evaluate_model(model.lam, selected, 2, destination, official=False)
                codes_path = destination / 'posterior-codes.pt'
                codes = torch.load(codes_path, map_location='cpu', weights_only=True)
                assert codes['row_identity_sha256'] == selected['reference_identity_sha256']
                codes.update(step=step, samples=samples, precision='highest FP32; TF32 disabled',
                             mean_code_sample_count=samples,
                             parent_512_identity_sha256=cache['reference_identity_sha256'])
                torch.save(codes, codes_path)
                rows = report['rows']
                metrics = {key: float(np.mean([row['metrics'][key] for row in rows])) for key in rows[0]['metrics']}
                for control in ['zero', 'mean', 'swapped', 'current_only']:
                    for metric in ['visual_mse', 'state_mse']:
                        metrics[control + '_minus_real_' + metric] = metrics[control + '_' + metric] - metrics['real_' + metric]
                prefix = {key: value[:64] if isinstance(value, torch.Tensor) and len(value) == samples else value
                          for key, value in codes.items()}
                record = {'step': step, 'samples': samples, 'metrics': metrics,
                          'posterior_geometry': geometry(codes), 'fixed_prefix64_geometry': geometry(prefix),
                          'row_identity_sha256': selected['reference_identity_sha256'],
                          'posterior_codes_sha256': ev.digest(codes_path),
                          'precision': 'Shared cached DINO highest FP32; encoder/VAE/decoder FP32; no autocast or TF32.',
                          'elapsed_seconds': time.monotonic() - started}
                report.update(record)
                write_json(destination / 'results.json', report)
        finally:
            ev.SAMPLES = old_samples
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
            # In PyTorch2.11, a legacy allow_tf32 setter AFTER restoring
            # 'medium' leaves mixed precision APIs and get() raises. Restore
            # the high-level precision last, then assert its exact value.
            torch.set_float32_matmul_precision(precision)
            random.setstate(py_state)
            np.random.set_state(np_state)
            for module, flag in training_flags:
                module.training = flag
        assert torch.equal(torch.get_rng_state(), cpu_state)
        assert torch.get_float32_matmul_precision() == precision
        assert torch.backends.cuda.matmul.allow_tf32 == tf32
        assert torch.backends.cudnn.allow_tf32 == cudnn_tf32
        assert torch.equal(torch.cuda.get_rng_state(device), gpu_state)
        assert hashlib.sha256(pickle.dumps(random.getstate())).hexdigest() == py_hash
        assert hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest() == np_hash
        assert all(p._version == version and p.requires_grad == requires_grad for p, version, requires_grad in parameter_versions)
        assert all(module.training == flag for module, flag in training_flags)
        record['rng_parameter_versions_and_training_flags_unchanged'] = True
        self.records.append(record)
        write_json(self.output / 'onset_validation_latest.json', record)
        with (self.output / 'onset_validation.jsonl').open('a') as handle:
            handle.write(json.dumps(record, allow_nan=False) + '\n')
        print(json.dumps({'phase': 'onset_FP32_probe', **record}, allow_nan=False), flush=True)
