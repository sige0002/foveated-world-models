"""Recover paired continuation from a saved state with verified RNG replay.

Uses the pinned upstream training forward, GPU augmentation, AdamW groups,
restored Adam moments and scheduler. Explicit manifests replace epoch-dependent
sampling. Local updates0..2999 are reported as absolute updates3001..6000.
"""
from __future__ import annotations
import argparse
from copy import deepcopy
import fcntl
from functools import partial
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import evaluate_robotwin_loss_ablation_fp32_v2 as ev
from research.train_robotwin_stage1 import install_episode_split, write_json


def tensor_hash(value):
    return ev.tensor_hash(value)


def state_hash(value):
    """Fingerprint nested model/optimizer state, including scalar options."""
    import torch
    h = hashlib.sha256()
    def visit(v):
        if isinstance(v, torch.Tensor):
            h.update(b'tensor'); h.update(bytes.fromhex(tensor_hash(v)))
        elif isinstance(v, dict):
            for k in sorted(v, key=str):
                h.update(repr(k).encode()); visit(v[k])
        elif isinstance(v, (tuple, list)):
            for x in v: visit(x)
        else:
            h.update(repr(v).encode())
    visit(value)
    return h.hexdigest()


def evaluate(model, cache, output, absolute_step, samples):
    import numpy as np
    import torch
    output.mkdir(parents=True, exist_ok=True)
    was_training = model.training
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    old_samples = ev.SAMPLES
    py_rng, np_rng = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            torch.set_float32_matmul_precision('highest')
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            model.eval()
            selected = cache
            if samples != 512:
                selected = {k: (v[:samples] if isinstance(v, torch.Tensor) and v.ndim and len(v) == 512 else v)
                            for k, v in cache.items()}
                selected['rows'] = deepcopy(cache['rows'][:samples])
                episodes = np.array([r['episode'] for r in selected['rows']])
                rng = np.random.default_rng(1942)
                for _ in range(10000):
                    donors = rng.permutation(samples)
                    if (episodes[donors] != episodes).all(): break
                else: raise RuntimeError('No valid smoke donors')
                for i, row in enumerate(selected['rows']):
                    row.update(donor_index=int(donors[i]), donor_episode=int(episodes[donors[i]]),
                               donor_base=selected['rows'][donors[i]]['base'])
            ev.SAMPLES = samples
            with torch.inference_mode():
                result = ev.evaluate_model(model.lam, selected, 2, output, official=False)
            rows = result['rows']
            aggregate = {key: sum(r['metrics'][key] for r in rows) / samples for key in rows[0]['metrics']}
            for control in ('zero', 'mean', 'swapped', 'current_only'):
                for metric in ('visual_mse', 'state_mse'):
                    aggregate[control + '_minus_real_' + metric] = aggregate[control + '_' + metric] - aggregate['real_' + metric]
            result.update(absolute_step=absolute_step, samples=samples, metrics=aggregate,
                          precision='highest FP32 cached DINO/encoder/VAE/decoder; no autocast; TF32 disabled')
            write_json(output / 'results.json', result)
            return {k: v for k, v in result.items() if k != 'rows'}
    finally:
        ev.SAMPLES = old_samples
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = cudnn_tf32
        torch.set_float32_matmul_precision(precision)
        assert torch.get_float32_matmul_precision() == precision
        random.setstate(py_rng); np.random.set_state(np_rng)
        model.train(was_training)


def rng_state(loader_generator):
    import numpy as np
    import torch
    state = np.random.get_state()
    return {'python': random.getstate(),
            'numpy': (state[0], state[1].tolist(), int(state[2]), int(state[3]), float(state[4])),
            'torch_cpu': torch.get_rng_state(), 'torch_cuda': torch.cuda.get_rng_state_all(),
            'loader_generator': loader_generator.get_state()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--arm', choices=('seed43_replay', 'seed43_new', 'seed44_replay', 'seed44_new'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--resume-run', type=Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    assert out.is_relative_to(ev.SSD_RUNS.resolve())
    assert not out.exists() or not any(out.iterdir())
    out.mkdir(parents=True, exist_ok=True)
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(out), '-o', 'TARGET,SOURCE,FSTYPE'], text=True))
    assert mount['filesystems'][0]['target'] == '/media/sadasue/SSD1'
    manifest = ev.read_json(args.manifest)
    for path, expected_hash in manifest['input_file_sha256'].items():
        assert ev.digest(Path(path)) == expected_hash, f'Pinned input changed: {path}'
    source_run = Path(manifest['source_run'])
    arm_info = manifest['arms'][args.arm]
    arm_path = Path(arm_info['path'])
    assert ev.digest(arm_path) == arm_info['sha256']
    arm = ev.read_json(arm_path)
    seed, condition, rows = arm['seed'], arm['condition'], arm['rows']
    assert seed in (43, 44) and len(rows) == 12000
    assert args.arm == f'seed{seed}_{condition}'
    assert manifest['base_step'] == 3000
    checkpoint = Path(manifest['checkpoint']['path'])
    checkpoint_sha = manifest['checkpoint']['sha256']
    assert ev.digest(checkpoint) == checkpoint_sha
    assert manifest['additional_steps'] == 3000 and manifest['batch_size'] == 4
    assert manifest['original_batches_verified'] == 3000
    resume_run = args.resume_run.resolve()
    resume_checkpoint = resume_run / 'checkpoints/last.ckpt'
    resume_checkpoint_sha = ev.digest(resume_checkpoint)
    previous_protocol = ev.read_json(resume_run / 'protocol.json')
    assert previous_protocol['arm'] == args.arm
    assert previous_protocol['script_sha256'] == '17f4279f2761dcdfc9ce3e620d15e63bfad5e4f343b42a5225790a3f0d50f801'
    assert previous_protocol['manifest_sha256'] == ev.digest(args.manifest)
    previous_stream = [json.loads(line) for line in (resume_run / 'training_stream.jsonl').read_text().splitlines()]
    assert all(row['local_step'] == i + 1 for i, row in enumerate(previous_stream))
    import yaml
    cfg = yaml.safe_load((source_run / 'resolved_config.yaml').read_text())
    req = yaml.safe_load((source_run / 'requested_config.yaml').read_text())
    contract = ev.read_json(source_run / 'data_contract.json')
    split = ev.read_json(source_run / 'episode_split.json')['ids']
    train_ids = set(split['train'])
    assert all(r['episode'] in train_ids for r in rows)
    os.environ.update(cfg['environment'])
    sys.path[:0] = [req['official_source'], *req['dependency_roots']]
    assert subprocess.check_output(['git', '-C', req['official_source'], 'rev-parse', 'HEAD'], text=True).strip() == contract['source_commit']
    assert hashlib.sha256(subprocess.check_output(['git', '-C', req['official_source'], 'diff', '--binary'])).hexdigest() == contract['source_patch_sha256']
    assert ev.digest(ROOT / 'research/train_robotwin_stage1.py') == '9ecbea0151c1f902647dc9e41a5a17a431ab5c4319da658b9d0cdeb5104cba27'
    import numpy as np
    import torch
    from torch.utils.data import Dataset, DataLoader
    from lightning.pytorch import seed_everything
    from latent_action_model.core.lam_lightinng import VJEPA_LAM
    from latent_action_model.data_loader.lerobot_dataset import LeRobotLAMDataset
    from latent_action_model.data_loader.collate import lam_collate
    from latent_action_model.data_loader.lerobot_datamodule import _lam_worker_init_fn
    from latent_action_model.core.utils.utils import eef_reconstruction_loss
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    assert torch.cuda.is_available()
    memory = int(next(l.split()[1] for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))) * 1024
    assert memory >= 20 * 1024**3 and torch.cuda.mem_get_info()[0] >= 16 * 1024**3
    install_episode_split(split)
    seed_everything(seed, workers=True)
    mc = deepcopy(cfg['model']); opt_cfg = mc.pop('optimizer')
    opt_args = {k: float(v) for k, v in opt_cfg['init_args'].items()}
    model = VJEPA_LAM(**mc, optimizer=partial(torch.optim.AdamW, **opt_args), weight_decay=opt_args['weight_decay'])
    model.hparams.pop('optimizer', None)
    saved = torch.load(resume_checkpoint, weights_only=True, map_location='cpu', mmap=True)
    resume_local = saved['continuation']['local_updates']
    assert resume_local == 500 and saved['global_step'] == 3000 + resume_local
    assert saved['data_contract'] == contract and saved['continuation']['protocol'] == previous_protocol
    assert len(previous_stream) >= resume_local + 6
    assert model.loss_type == 'l2' and model.lam.vq.beta == 1e-5
    model.load_state_dict(saved['state_dict'], strict=True)
    initial_model_sha = state_hash(model.state_dict())
    assert initial_model_sha == state_hash(saved['state_dict'])
    hp = deepcopy(saved['hyper_parameters'])
    model.cuda().train()
    configured = model.configure_optimizers()
    optimizer = configured['optimizer']; scheduler = configured['lr_scheduler']['scheduler']
    optimizer.load_state_dict(saved['optimizer_states'][0])
    scheduler.load_state_dict(saved['lr_schedulers'][0])
    initial_optimizer_sha = state_hash(optimizer.state_dict())
    assert initial_optimizer_sha == state_hash(saved['optimizer_states'][0])
    assert scheduler.last_epoch == 3000 + resume_local and all(g['lr'] == 3e-4 for g in optimizer.param_groups)
    assert all(float(s['step']) == 3000 + resume_local for s in optimizer.state.values())
    group_names = [[name for p in group['params'] for name, q in model.named_parameters() if p is q]
                   for group in optimizer.param_groups]
    assert [len(x) for x in group_names] == [173, 276]
    assert all(not p.requires_grad for p in model.lam.vision_encoder.parameters())
    del saved
    gc.collect()
    final_eval = source_run.parent / 'evaluations/final3000-highest-fp32-v2'
    report = ev.read_json(final_eval / 'results.json')
    assert report['complete'] and report['models']['kl_mse']['checkpoint_sha256'] == checkpoint_sha
    assert report['protocol']['script_sha256'] == ev.digest(ev.__file__)
    cache_path = Path(report['protocol']['cache'])
    assert ev.digest(cache_path) == ev.read_json(final_eval / 'feature-cache-manifest.json')['sha256']
    cache = torch.load(cache_path, weights_only=True, map_location='cpu')
    assert ev.digest(Path(contract['data_root']) / 'meta/stats_gr00t.json') == cache['stats_sha256']
    assert len(cache['rows']) == 512 and len({r['episode'] for r in cache['rows']}) == 415
    reference = ev.read_json(source_run / 'z_validation_step003000.json')['rows']
    assert all((a['episode'], a['base'], a['donor_episode'], a['donor_base']) ==
               (b['episode'], b['base'], b['donor_episode'], b['donor_base']) for a, b in zip(cache['rows'], reference))
    dc = cfg['data']; keys = ('data_root_dir', 'data_mix', 'num_frames', 'val_tail_ratio', 'video_backend', 'image_hw', 'frame_dt_sec', 'human_frame_dt_sec')
    dataset = LeRobotLAMDataset(**{k: dc[k] for k in keys if k in dc}, mode='train')
    dataset.mixture.set_epoch(0)
    class ManifestDataset(Dataset):
        def __len__(self): return len(rows)
        def __getitem__(self, index):
            row = rows[index]
            sample = dataset._try_get_sample(row['dataset_index'])
            assert (int(sample['trajectory_id']), int(sample['base_index'])) == (row['episode'], row['base'])
            return sample
    loader_generator = torch.Generator().manual_seed(seed + 12001)
    loader = DataLoader(ManifestDataset(), batch_size=4, shuffle=False, num_workers=2,
        collate_fn=partial(lam_collate, max_state_dim=32), worker_init_fn=_lam_worker_init_fn,
        generator=loader_generator, pin_memory=True, prefetch_factor=2, in_order=True)
    protocol = {'source_run': str(source_run), 'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
        'manifest': str(args.manifest.resolve()), 'manifest_sha256': ev.digest(args.manifest),
        'arm_manifest': str(arm_path), 'arm_manifest_sha256': ev.digest(arm_path), 'arm': args.arm,
        'seed': seed, 'condition': condition, 'base_step': 3000,
        'additional_updates': 6 if args.preflight else 3000 - resume_local,
        'target_total_local_updates': 3000,
        'batch_size': 4, 'warmup_restarted': False, 'learning_rate': 3e-4, 'precision': 'bf16-mixed; upstream medium FP32 matmul',
        'validation': 'fixed512/415 highestFP32 shared DINO cache and donor permutation; same official reference',
        'initial_model_sha256': initial_model_sha, 'initial_optimizer_sha256': initial_optimizer_sha,
        'initial_scheduler': scheduler.state_dict(), 'optimizer_parameter_names': group_names,
        'source_commit': contract['source_commit'], 'source_patch_sha256': contract['source_patch_sha256'],
        'script_sha256': ev.digest(__file__), 'upstream_training_core_sha256': ev.digest(Path(req['official_source']) / 'latent_action_model/core/lam_lightinng.py'),
        'mount': mount, 'initial_host_available_bytes': memory,
        'pair_scope': 'New means unseen episode/base pair, not unseen episode or disjoint frames. Both continuation seeds start from the same trained checkpoint.',
        'checkpoint_scope': 'Explicit model/Adam/scheduler from own local500 checkpoint; seed-based actual-data RNG replay before resuming501.',
        'resume_run': str(resume_run), 'resume_checkpoint': str(resume_checkpoint),
        'resume_checkpoint_sha256': resume_checkpoint_sha, 'resume_local_step': resume_local,
        'resume_support_scope': 'This version recovers the legacy local500 checkpoint only; newly saved RNG-rich checkpoints are not accepted by this CLI yet.',
        'discarded_unsaved_original_local_steps': [resume_local + 1, len(previous_stream)],
        'previous_protocol': previous_protocol,
        'stream_scope': 'training_stream contains newly executed501 onward; cumulative hashes include verified original1..500 prefix. replay_verified_prefix records reconstruction separately.',
        'rng_scope': 'Old checkpoint did not save RNG. CPU/Python/NumPy/worker state is reconstructed by same seeded loader/augmentation path; old CUDA and epsilon hashes are checked. Old CPU RNG and augmented-pixel hashes were not recorded, so cannot be directly checked.'}
    write_json(out / 'protocol.json', protocol)
    (out / 'resolved_config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    write_json(out / 'data_contract.json', contract)
    (out / 'episode_split.json').write_bytes((source_run / 'episode_split.json').read_bytes())
    (out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    # Match the two conditions' RNG at the first real continuation update.
    seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision('medium')
    trainable = [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad]
    assert len({id(p) for p in trainable}) == len(trainable)
    state = {}
    def before_vae(module, inputs):
        if not model.training: return
        state['nodes'] = inputs[0]
        with torch.no_grad(), torch.random.fork_rng(devices=[torch.cuda.current_device()]):
            mu, lv = module._encode(inputs[0]); epsilon = torch.randn_like(mu)
            state['expected_z'] = mu + (.5 * lv).exp() * epsilon
            state['epsilon_sha256'] = tensor_hash(epsilon)
    def after_vae(module, inputs, outputs):
        if model.training:
            assert torch.equal(outputs[0].detach(), state.pop('expected_z'))
    def after_lam(module, inputs, outputs):
        if model.training: state['outputs'] = outputs
    handles = [model.lam.vq.register_forward_pre_hook(before_vae), model.lam.vq.register_forward_hook(after_vae),
               model.lam.register_forward_hook(after_lam)]
    updates = resume_local + 6 if args.preflight else 3000
    started = time.monotonic(); durations = []; losses = []; stream_hashes = {k: hashlib.sha256() for k in ('data', 'epsilon', 'augmentation_rng')}
    torch.cuda.reset_peak_memory_stats()
    optimizer.zero_grad(set_to_none=True)
    for local, raw in enumerate(loader, 1):
        tick = time.monotonic()
        expected = rows[(local - 1) * 4:local * 4]
        observed = [[int(ep), int(base)] for ep, base in zip(raw['trajectory_ids'], raw['base_indices'])]
        assert observed == [[r['episode'], r['base']] for r in expected]
        assert ev.json_hash(observed) == arm['data_sha256_by_batch'][local - 1]
        rng_before = tensor_hash(torch.cuda.get_rng_state())
        cpu_rng_before = tensor_hash(torch.get_rng_state())
        batch = model.transfer_batch_to_device(raw, torch.device('cuda'), 0)
        rng_after = tensor_hash(torch.cuda.get_rng_state())
        cpu_rng_after = tensor_hash(torch.get_rng_state())
        # Hash actual augmented tensors for this recovery execution. Historical
        # v1 did not retain these hashes; do not claim historical pixel equality.
        augmentation_pixel_hash = ev.json_hash({key: tensor_hash(batch[key]) for key in ('videos', 'dec_videos')})
        if local <= resume_local:
            epsilon = torch.randn_like(torch.empty((4, 1, 32), device='cuda', dtype=torch.float32))
            replay = {'local_step': local, 'absolute_step': 3000 + local, 'pairs': observed,
                'data_sha256': ev.json_hash(observed), 'epsilon_sha256': tensor_hash(epsilon),
                'augmentation_rng_before_sha256': rng_before, 'augmentation_rng_after_sha256': rng_after,
                'augmentation_rng_sha256': ev.json_hash([rng_before, rng_after]),
                'cpu_rng_before_augmentation_sha256': cpu_rng_before, 'cpu_rng_after_augmentation_sha256': cpu_rng_after,
                'augmentation_pixels_sha256': augmentation_pixel_hash, 'scope': 'reconstructed prefix, no forward/backward/update'}
            old = previous_stream[local - 1]
            for key in ('pairs', 'data_sha256', 'epsilon_sha256', 'augmentation_rng_before_sha256', 'augmentation_rng_after_sha256', 'augmentation_rng_sha256'):
                assert replay[key] == old[key], (local, key)
            for key, h in stream_hashes.items():
                h.update(bytes.fromhex(replay[key + '_sha256']))
                assert h.hexdigest() == old['cumulative_' + key + '_sha256'], (local, key)
            with (out / 'replay_verified_prefix.jsonl').open('a') as f: f.write(json.dumps(replay) + '\n')
            if local % 100 == 0:
                print(json.dumps({'phase': 'rng_prefix_replay', 'verified_batches': local}), flush=True)
            if local == resume_local:
                boundary = tensor_hash(torch.cuda.get_rng_state())
                assert boundary == previous_stream[local]['augmentation_rng_before_sha256']
                assert scheduler.last_epoch == 3000 + resume_local
                proof = {'verified_prefix_batches': local, 'boundary_next_local_step': local + 1,
                    'boundary_cuda_rng_sha256': boundary, 'boundary_matches_old501': True,
                    'restored_model_sha256': initial_model_sha, 'restored_optimizer_sha256': initial_optimizer_sha,
                    'model_unchanged_by_replay': state_hash(model.state_dict()) == initial_model_sha,
                    'optimizer_unchanged_by_replay': state_hash(optimizer.state_dict()) == initial_optimizer_sha,
                    'cpu_rng_sha256': tensor_hash(torch.get_rng_state()), 'python_numpy_rng_sha256': state_hash(rng_state(loader_generator)),
                    'cpu_and_pixel_historical_hashes_available': False}
                assert proof['model_unchanged_by_replay'] and proof['optimizer_unchanged_by_replay']
                write_json(out / 'resume_replay_verification.json', proof)
                torch.save(rng_state(loader_generator), out / 'reconstructed_boundary_rng.pt')
            del batch, raw, epsilon
            continue
        assert all(g['lr'] == 3e-4 for g in optimizer.param_groups)
        audit_step = local <= resume_local + 6 or local % 500 == 0
        cpu_before_forward = torch.get_rng_state()
        python_before_forward = random.getstate()
        numpy_before_forward = state_hash(np.random.get_state())
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=False):
            loss, logs = model.shared_step(batch)
            if audit_step:
                forward = state['outputs']
                state_loss = model.lambda_aux * eef_reconstruction_loss(forward[5], batch['delta_proprio'], batch['state_mask'].any(1), state_loss_type=model.state_loss_type)
                parts = {'image': logs['recon_loss'], 'state': state_loss, 'weighted_kl': logs['vq_loss']}
                torch.testing.assert_close(sum(parts.values()), loss, rtol=0, atol=1e-6)
        scalar = float(loss.detach()); assert math.isfinite(scalar)
        assert torch.equal(cpu_before_forward, torch.get_rng_state())
        assert python_before_forward == random.getstate() and numpy_before_forward == state_hash(np.random.get_state())
        if audit_step:
            gradients = {k: torch.autograd.grad(v, state['nodes'], retain_graph=True)[0].detach().float() for k, v in parts.items()}
            record = {'absolute_step_before_update': 3000 + local - 1, 'losses': {k: float(v.detach()) for k, v in parts.items()},
                      'node_gradient_norms': {k: float(v.norm()) for k, v in gradients.items()}}
            with (out / 'loss_gradient_components.jsonl').open('a') as f: f.write(json.dumps(record) + '\n')
            del gradients, parts, forward, state_loss
        loss.backward()
        if local <= resume_local + 6:
            for name, parameter in model.lam.vq.named_parameters():
                if parameter.requires_grad:
                    assert parameter.grad is not None and torch.isfinite(parameter.grad).all(), name
        norm = torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step(); scheduler.step(); optimizer.zero_grad(set_to_none=True)
        assert scheduler.last_epoch == 3000 + local
        stream = {'local_step': local, 'absolute_step': 3000 + local, 'pairs': observed,
                  'data_sha256': ev.json_hash(observed), 'epsilon_sha256': state['epsilon_sha256'],
                  'augmentation_rng_before_sha256': rng_before, 'augmentation_rng_after_sha256': rng_after,
                  'augmentation_rng_sha256': ev.json_hash([rng_before, rng_after]),
                  'cpu_rng_before_augmentation_sha256': cpu_rng_before, 'cpu_rng_after_augmentation_sha256': cpu_rng_after,
                  'augmentation_pixels_sha256': augmentation_pixel_hash}
        if local <= len(previous_stream):
            for key in ('pairs', 'data_sha256', 'epsilon_sha256', 'augmentation_rng_before_sha256', 'augmentation_rng_after_sha256'):
                assert stream[key] == previous_stream[local - 1][key], (local, key)
        for key, h in stream_hashes.items():
            h.update(bytes.fromhex(stream[key + '_sha256'])); stream['cumulative_' + key + '_sha256'] = h.hexdigest()
        with (out / 'training_stream.jsonl').open('a') as f: f.write(json.dumps(stream) + '\n')
        durations.append(time.monotonic() - tick); losses.append(scalar)
        if local <= resume_local + 6 or local % 20 == 0:
            progress = {'phase': 'training', 'pid': os.getpid(), 'local_step': local, 'absolute_step': 3000 + local,
                        'loss': scalar, 'gradient_norm_before_clip': float(norm), 'lr': optimizer.param_groups[0]['lr'],
                        'seconds_per_update_recent': float(np.median(durations[-20:])), 'elapsed_seconds': time.monotonic() - started,
                        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
            write_json(out / 'progress.json', progress)
            with (out / 'progress.jsonl').open('a') as f: f.write(json.dumps(progress) + '\n')
            print(json.dumps(progress), flush=True)
        state.clear(); model._last_step_tensors = {}
        del loss, logs, batch, raw
        if local == updates or (not args.preflight and local % 500 == 0):
            if not args.preflight:
                # Serialize one run at a time to bound transient CPU memory.
                with (args.manifest.parent / 'checkpoint-write.lock').open('a') as lock:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                    folder = out / 'checkpoints'; folder.mkdir(exist_ok=True)
                    payload = {'global_step': 3000 + local, 'state_dict': model.state_dict(),
                               'optimizer_states': [optimizer.state_dict()], 'lr_schedulers': [scheduler.state_dict()],
                               'hyper_parameters': hp, 'data_contract': contract,
                               'continuation': {'protocol': protocol, 'local_updates': local, 'explicit_state_resume_only': True},
                               'rng_state': rng_state(loader_generator),
                               'data_position': {'next_local_step': local + 1, 'next_manifest_row': local * 4},
                               'stream_cumulative_sha256': {key: h.hexdigest() for key, h in stream_hashes.items()},
                               'precision': torch.get_float32_matmul_precision()}
                    torch.save(payload, folder / 'last.ckpt.tmp')
                    (folder / 'last.ckpt.tmp').replace(folder / 'last.ckpt')
                    del payload
                    if local == updates: os.link(folder / 'last.ckpt', folder / 'final.ckpt')
            rng_before_eval = state_hash(rng_state(loader_generator))
            evaluation = evaluate(model, cache, out / 'evaluations' / f'step{3000 + local:06d}', 3000 + local, 8 if args.preflight else 512)
            assert state_hash(rng_state(loader_generator)) == rng_before_eval
            assert torch.get_float32_matmul_precision() == 'medium'
            write_json(out / 'validation_latest.json', evaluation)
            if args.preflight:
                for repeat in range(2):
                    evaluate(model, cache, out / 'evaluations' / f'restore_repeat{repeat+2}', 3000 + local, 8)
                    assert state_hash(rng_state(loader_generator)) == rng_before_eval
                    assert torch.get_float32_matmul_precision() == 'medium'
                write_json(out / 'repeated_evaluation_restoration.json', {'completed_evaluations': 3, 'precision_restored': 'medium', 'all_rng_unchanged': True})
        if local == updates: break
    for handle in handles: handle.remove()
    result = {'phase': 'preflight_passed' if args.preflight else 'complete', 'local_updates': local, 'absolute_step': 3000 + local,
              'initial_model_sha256': initial_model_sha, 'initial_optimizer_sha256': initial_optimizer_sha,
              'resumed_local_step': resume_local, 'new_updates_executed': local - resume_local,
              'scheduler_last_epoch': scheduler.last_epoch, 'lr': optimizer.param_groups[0]['lr'],
              'seconds_per_update_excluding_first': float(np.median(durations[1:])), 'elapsed_seconds': time.monotonic() - started,
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'preflight_losses': losses if args.preflight else None, 'final_validation': evaluation}
    write_json(out / 'run_summary.json', result)
    print(json.dumps({'phase': result['phase'], 'absolute_step': result['absolute_step']}), flush=True)


if __name__ == '__main__':
    main()
