"""Train pinned upstream Stage 1 on qfix RoboTwin with a saved episode split.

The upstream checkout differs only by the reviewed state-output-scale patch.
This runner changes episode membership, not frame selection or transformations.
"""
from __future__ import annotations

import argparse
from functools import partial
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SSD_RUNS = Path('/media/sadasue/SSD1/foveated-world-models/runs')


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_split(rows, fraction, seed):
    import numpy as np
    rows = sorted(rows, key=lambda r: int(r['episode_index']))
    ids = np.array([int(r['episode_index']) for r in rows], dtype=np.int64)
    if len(set(ids.tolist())) != len(rows):
        raise ValueError('Duplicate episode IDs in metadata')
    n_val = int(len(rows) * fraction)
    if not 0 < n_val < len(rows):
        raise ValueError('Both train and validation must contain episodes')
    val = set(np.random.default_rng(seed).permutation(ids)[:n_val].tolist())
    parts = {name: [r for r in rows if (int(r['episode_index']) in val) == is_val]
             for name, is_val in [('train', False), ('val', True)]}
    split = {name: [int(r['episode_index']) for r in subset] for name, subset in parts.items()}
    assert not set(split['train']) & set(split['val'])
    assert set(split['train']) | set(split['val']) == set(ids.tolist())
    counts = {name: {'episodes': len(subset), 'frames': sum(int(r['length']) for r in subset)}
              for name, subset in parts.items()}
    return split, counts


def install_episode_split(split):
    """Override only the upstream split hook, before it builds step indexing."""
    import numpy as np
    import latent_action_model.data_loader.lerobot_dataset as lam_data
    original = lam_data.LeRobotSingleDataset
    all_ids = set(split['train']) | set(split['val'])

    class EpisodeSplitDataset(original):
        def _build_mode_split_from_trajectories(self):
            if set(self._all_trajectory_ids.tolist()) != all_ids:
                raise ValueError('Episode manifest does not match dataset metadata')
            mode = self._split_mode
            selected = all_ids if mode == 'all' else set(split[mode])
            mask = np.isin(self._all_trajectory_ids, sorted(selected))
            active = self._all_trajectory_ids[mask].astype(np.int64, copy=False)
            self._active_episode_indices = active
            self._active_traj_ids = active
            self._trajectory_ids = active
            self._trajectory_lengths = self._all_trajectory_lengths[mask].astype(np.int64, copy=False)
            self._trajectory_id_to_index_active = {int(ep): i for i, ep in enumerate(active)}

    lam_data.LeRobotSingleDataset = EpisodeSplitDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/stage1_robotwin_qfix.yaml'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    import yaml
    cfg = yaml.safe_load(Path(args.config).read_text())
    diagnostic = cfg.get('diagnostic', {})
    if diagnostic and diagnostic.get('protocol') != 'robotwin_loss_ablation_20261114':
        raise ValueError('Unknown diagnostic protocol')
    output = Path(args.output).resolve()
    if not output.is_relative_to(SSD_RUNS.resolve()):
        raise ValueError('Checkpoint output must physically reside below SSD1 runs')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new empty output directory; this runner starts from scratch')
    output.mkdir(parents=True, exist_ok=True)
    env = {'HF_HOME': cfg['hf_home'], 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'LAM_ENABLE_MANUAL_WANDB': '0', 'WANDB_MODE': 'disabled',
           'NO_ALBUMENTATIONS_UPDATE': '1', 'LAM_WORKER_OMP_THREADS': '1',
           'LAM_PYAV_THREAD_COUNT': '1', 'LAM_PYAV_THREAD_TYPE': 'NONE',
           'OMP_NUM_THREADS': str(cfg['cpu_threads']), 'MKL_NUM_THREADS': str(cfg['cpu_threads']),
           'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1',
           'TOKENIZERS_PARALLELISM': 'false', 'HF_DATASETS_DISABLE_PROGRESS_BARS': '1'}
    os.environ.update(env)
    source = Path(cfg['official_source']).resolve()
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != cfg['official_commit']:
        raise ValueError(f'Wrong official source revision: {head}')
    patch = subprocess.check_output(['git', '-C', str(source), 'diff', '--binary'])
    if patch != (ROOT / 'research/patches/official-state-output-scale.patch').read_bytes():
        raise ValueError('Official checkout has unexpected modifications')
    sys.path[:0] = [str(source), *cfg['dependency_roots']]
    base = yaml.safe_load((source / cfg['official_config']).read_text())
    model_cfg = {**base['model'], **cfg['model']}
    optimizer_cfg = model_cfg.pop('optimizer')
    data_cfg = {**base['data'], **cfg['data'], 'val_tail_ratio': cfg['split']['validation_fraction']}
    trainer_cfg = {**base['trainer'], **cfg['trainer']}
    for key in ('callbacks', 'logger'):
        trainer_cfg.pop(key, None)
    batch_warmup = (data_cfg['batch_size'], model_cfg['warmup_steps'])
    if batch_warmup not in {(4, 1000), (64, 10000)} or trainer_cfg['accumulate_grad_batches'] != 1:
        raise ValueError('Expected the original batch4/warmup1000 or requested batch64/warmup10000, without accumulation')
    if model_cfg['state_output_scale'] != 2:
        raise ValueError('Expected the audited 2*tanh state output')
    if batch_warmup == (64, 10000):
        upstream_lr = float(base['model']['optimizer']['init_args']['lr'])
        if float(optimizer_cfg['init_args']['lr']) != upstream_lr:
            raise ValueError('The batch64 experiment must preserve the upstream Stage-1 learning rate')
    if diagnostic:
        if batch_warmup != (4, 1000) or trainer_cfg['max_steps'] != 3000:
            raise ValueError('The controlled loss experiment requires batch4/warmup1000/3000 updates')
        if not data_cfg['in_order'] or trainer_cfg['val_check_interval'] != 500:
            raise ValueError('Controlled runs require ordered data and validation every500 updates')
        if model_cfg['loss_type'] not in {'smooth_l1', 'l2'} or float(model_cfg['vq_kwargs']['beta']) not in {1e-5, 5e-5}:
            raise ValueError('Unexpected loss factors for the controlled experiment')
    dataset = Path(data_cfg['data_root_dir']) / 'robotwin_merged'
    if not (dataset / 'qfix/audit.json').is_file() or not (dataset / 'videos').is_symlink():
        raise ValueError('Expected the audited qfix derivative with linked original videos')

    import numpy as np
    import pyarrow.parquet as pq
    rows = []
    ep_files = sorted((dataset / 'meta/episodes').rglob('*.parquet'))
    for path in ep_files:
        rows.extend(pq.read_table(path, columns=['episode_index', 'length']).to_pylist())
    split, counts = make_split(rows, **{'fraction': cfg['split']['validation_fraction'],
                                      'seed': cfg['split']['seed']})
    for part in counts.values():
        part['batches_at_batch4'] = math.ceil(part['frames'] / 4)
    manifest = {'method': 'episode_random_without_replacement', **cfg['split'],
                'ids': split, 'counts': counts, 'episode_overlap': 0}
    write_json(output / 'episode_split.json', manifest)
    contract = {'source_commit': head, 'source_patch_sha256': hashlib.sha256(patch).hexdigest(),
                'data_root': str(dataset), 'video_target': str((dataset / 'videos').resolve()),
                'qfix_audit_sha256': sha256(dataset / 'qfix/audit.json'),
                'episode_metadata_sha256': {str(p.relative_to(dataset)): sha256(p) for p in ep_files},
                'split_sha256': sha256(output / 'episode_split.json'),
                'embodiment_id': 1, 'state_valid_dim': 16, 'state_padded_dim': 32,
                'target': 'future-minus-current including grippers', 'state_output_scale': 2.0,
                'normalization': 'upstream dataset statistics; XYZ/gripper minmax, quaternion passthrough',
                'frame_dt_sec': data_cfg['frame_dt_sec'], 'initialization': 'fresh; frozen local DINOv3 only'}
    write_json(output / 'data_contract.json', contract)
    (output / 'upstream.patch').write_bytes(patch)
    shutil.copy2(__file__, output / 'train_robotwin_stage1.py')
    shutil.copy2(args.config, output / 'requested_config.yaml')
    resolved = {'model': {**model_cfg, 'optimizer': optimizer_cfg}, 'data': data_cfg,
                'trainer': trainer_cfg, 'split': cfg['split'], 'environment': env}
    if diagnostic:
        resolved['diagnostic'] = diagnostic
    (output / 'resolved_config.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False))
    print(json.dumps({'phase': 'split', 'counts': counts, 'overlap': 0}), flush=True)
    install_episode_split(split)

    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from latent_action_model.data_loader.lerobot_datamodule import LeRobotDataModule
    from latent_action_model.core.lam_lightinng import VJEPA_LAM
    from latent_action_model.core.utils.utils import eef_reconstruction_loss

    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_num_interop_threads(1)
    seed_everything(cfg['seed'], workers=True)

    class SplitDataModule(LeRobotDataModule):
        def setup(self, stage=None):
            if self.train_dataset is None:
                super().setup(stage)

        def train_dataloader(self):
            loader = super().train_dataloader()
            if diagnostic:
                # Worker seeds must not depend on validation iterator creation.
                loader.generator = torch.Generator().manual_seed(cfg['seed'] + 12001)
            return loader

        def val_dataloader(self):
            # Fixed validation prefix must not depend on worker completion order.
            prior = self.in_order
            self.in_order = True
            loader = super().val_dataloader()
            self.in_order = prior
            if diagnostic:
                loader.generator = torch.Generator().manual_seed(cfg['seed'] + 22001)
            return loader

    dm = SplitDataModule(**data_cfg)
    dm.setup('fit')
    audit = {}
    for name, ds in [('train', dm.train_dataset), ('val', dm.val_dataset)]:
        single = ds.mixture.datasets[0]
        assert set(single.trajectory_ids.tolist()) == set(split[name])
        assert single.embodiment_id == 1
        assert len(ds) == counts[name]['frames']
        # Every local episode boundary maps to its own absolute metadata range.
        lengths = single._subset_lengths
        starts = np.r_[0, np.cumsum(lengths)[:-1]]
        for i, start in enumerate(starts):
            assert single._local_to_abs_index(int(start)) == int(single._subset_from_indices[i])
            end = int(start + lengths[i] - 1)
            assert single._local_to_abs_index(end) == int(single._subset_from_indices[i] + lengths[i] - 1)
        allowed = set(split[name])
        candidates = [ds.mixture.sample_step(i) for i in range(1024)]
        assert all(int(ep) in allowed and 0 <= step < single._all_trajectory_lengths[
            single._trajectory_id_to_index[int(ep)]] for _, ep, step in candidates)
        sample = ds[0]
        assert sample['embodiment_id'] == 1 and sample['proprio'].shape == (2, 16)
        assert torch.isfinite(sample['proprio']).all()
        audit[name] = {'all_episode_boundaries_checked': len(starts),
                       'random_candidates_checked': len(candidates),
                       'sample_episode': sample['trajectory_id'],
                       'sample_video_shape': list(sample['frames'].shape),
                       'candidate_unique_episodes': len(set(int(ep) for _, ep, _ in candidates))}
    write_json(output / 'loader_audit.json', audit)
    if args.prepare_only:
        print(json.dumps({'phase': 'prepared', 'audit': audit}), flush=True)
        return
    if not torch.cuda.is_available():
        raise RuntimeError('GPU unavailable; no CPU training fallback')
    snapshot = subprocess.check_output(['nvidia-smi'], text=True)
    (output / 'gpu_before.txt').write_text(snapshot)
    (output / 'mount.txt').write_text(subprocess.check_output(['findmnt', '-T', str(output)], text=True))

    opt_args = {k: float(v) for k, v in optimizer_cfg['init_args'].items()}
    model = VJEPA_LAM(**model_cfg, optimizer=partial(torch.optim.AdamW, **opt_args),
                     weight_decay=opt_args['weight_decay'])
    model.hparams.pop('optimizer', None)
    assert model.lam.state_decoder.output_scale == 2
    modules = {'IDM': model.lam.encoder, 'LaWM': model.lam.decoder,
               'g': model.lam.state_decoder, 'DINO': model.lam.vision_encoder}
    parameters = {k: {'total': sum(p.numel() for p in m.parameters()),
                      'trainable': sum(p.numel() for p in m.parameters() if p.requires_grad)}
                  for k, m in modules.items()}
    assert parameters['DINO']['trainable'] == 0
    assert all(parameters[k]['trainable'] > 0 for k in ('IDM', 'LaWM', 'g'))
    write_json(output / 'parameters.json', parameters)

    def tensor_bytes(tensor):
        return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()

    if diagnostic:
        init_hash = hashlib.sha256()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                init_hash.update(name.encode() + b'\0')
                init_hash.update(str(tuple(parameter.shape)).encode() + b'\0')
                init_hash.update(tensor_bytes(parameter))
        write_json(output / 'initial_state_fingerprint.json', {
            'sha256': init_hash.hexdigest(), 'seed': cfg['seed'],
            'scope': 'All named trainable parameters before optimizer creation; frozen DINO excluded',
            'loss_type': model.loss_type, 'kl_beta': model.lam.vq.beta,
            'protocol': diagnostic,
        })

    class Monitor(Callback):
        def __init__(self):
            self.started = time.monotonic()
            self.times = []
            self.last_time = None
            self.gradient_checks = []
            self.stream_hashes = {k: hashlib.sha256() for k in ('data', 'aug', 'epsilon')}
            self.validation_cache = []
            self.vae_hooks = []

        def on_train_start(self, trainer, pl_module):
            self.started = self.last_time = time.monotonic()
            if diagnostic:
                def before_vae(module, inputs):
                    if not pl_module.training:
                        return
                    nodes = inputs[0]
                    if self.audit_this_step:
                        self.step_nodes = nodes
                    # Replay the exact impending random draw, restoring both CPU
                    # and CUDA RNG state. The post-hook verifies exact sampled z.
                    with torch.no_grad(), torch.random.fork_rng(devices=[nodes.device.index]):
                        mu, logvar = module._encode(nodes)
                        epsilon = torch.randn_like(mu)
                        self.expected_sampled_z = mu + (.5 * logvar).exp() * epsilon
                        self.epsilon_sha256 = hashlib.sha256(tensor_bytes(epsilon)).hexdigest()

                def after_vae(module, inputs, outputs):
                    if pl_module.training:
                        if not torch.equal(outputs[0].detach(), self.expected_sampled_z):
                            raise RuntimeError('Replayed VAE epsilon does not reproduce the actual sampled z')
                        del self.expected_sampled_z

                self.vae_hooks = [pl_module.lam.vq.register_forward_pre_hook(before_vae),
                                  pl_module.lam.vq.register_forward_hook(after_vae)]
                def capture_lam(module, inputs, outputs):
                    if pl_module.training and self.audit_this_step:
                        self.step_outputs = outputs
                self.vae_hooks.append(pl_module.lam.register_forward_hook(capture_lam))

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            if not diagnostic:
                return
            self.audit_this_step = trainer.global_step in {0, 9, 99, 499, 999, 1999, 2999}
            if self.audit_this_step:
                self.step_batch = batch
            pairs = list(zip(map(int, batch['trajectory_ids']), map(int, batch['base_indices'])))
            self.data_sha256 = hashlib.sha256(json.dumps(pairs, separators=(',', ':')).encode()).hexdigest()
            aug_hash = hashlib.sha256()
            for key in ('videos', 'dec_videos', 'states', 'delta_proprio'):
                aug_hash.update(key.encode() + b'\0')
                aug_hash.update(tensor_bytes(batch[key]))
            self.aug_sha256 = aug_hash.hexdigest()

        def on_before_backward(self, trainer, pl_module, loss):
            if not diagnostic or not self.audit_this_step:
                return
            out, batch = self.step_outputs, self.step_batch
            with torch.autocast('cuda', dtype=torch.bfloat16):
                image_loss = (torch.nn.functional.smooth_l1_loss(out[0], out[2], beta=.1)
                              if pl_module.loss_type == 'smooth_l1' else torch.nn.functional.mse_loss(out[0], out[2]))
                state_loss = pl_module.lambda_aux * eef_reconstruction_loss(
                    out[5], batch['delta_proprio'], batch['state_mask'].any(1),
                    state_loss_type=pl_module.state_loss_type)
                parts = {'image': image_loss, 'state': state_loss, 'weighted_kl': out[9]}
                torch.testing.assert_close(sum(parts.values()), loss, rtol=0, atol=1e-6)
            grads = {key: torch.autograd.grad(value, self.step_nodes, retain_graph=True)[0].detach().float()
                     for key, value in parts.items()}
            cosines = {}
            for left, right in [('image', 'state'), ('image', 'weighted_kl'), ('state', 'weighted_kl')]:
                denominator = grads[left].norm() * grads[right].norm()
                cosines[left + '__' + right] = (float((grads[left] * grads[right]).sum() / denominator)
                                                if float(denominator) > 0 else None)
            entry = {'step_before_update': trainer.global_step,
                     'lr_used': trainer.optimizers[0].param_groups[0]['lr'],
                     'losses': {key: float(value.detach()) for key, value in parts.items()},
                     'node_gradient_norms': {key: float(value.norm()) for key, value in grads.items()},
                     'pairwise_cosines': cosines,
                     'scope': 'Actual training batch and sampled epsilon, gradients of each loss at the float32 VAE input activation; normal backward follows.'}
            with (output / 'loss_gradient_components.jsonl').open('a') as f:
                f.write(json.dumps(entry, allow_nan=False) + '\n')
            del self.step_outputs, self.step_nodes, self.step_batch

        def on_validation_start(self, trainer, pl_module):
            self.z_validation_codes = []
            self.z_validation_rows = []
            if diagnostic:
                self.validation_rng = torch.random.fork_rng(devices=[pl_module.device.index])
                self.validation_rng.__enter__()
                self.python_rng = random.getstate()
                self.numpy_rng = np.random.get_state()
                self.validation_cache = []

        def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
            # A small fixed-prefix diagnostic: standard validation loss alone
            # cannot detect a predictor that ignores the individual latent.
            if batch_idx >= math.ceil(512 / data_cfg['batch_size']):
                return
            lam = pl_module.lam
            captured = {}
            hook = None
            if diagnostic:
                hook = lam.encoder.register_forward_hook(lambda module, inputs, value: captured.update(nodes=value.detach()))
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                out = lam.get_latent_action(batch['videos'], batch['states'], batch['dec_videos'],
                    state_mask=batch['state_mask'], embodiment_ids=batch['embodiment_ids'],
                    predict_future_frame=True)
                z = out['quantized']
                if diagnostic:
                    hook.remove()
                    with torch.autocast('cuda', enabled=False):
                        nodes = captured['nodes'].float()
                        mu, logvar = lam.vq._encode(nodes)
                    torch.testing.assert_close(z, mu, rtol=0, atol=0)
                    self.validation_cache.append({
                        'x': out['dec_in'].detach().cpu(), 'y': out['tgt'].detach().cpu(),
                        'z': z.detach().cpu(), 'logvar': logvar.detach().cpu(),
                        'nodes': nodes.detach().cpu(), 'pred': out['recon'].detach().cpu(),
                        'state_pred': out['s_pred'].detach().cpu(), 'states': batch['states'].detach().cpu(),
                        'delta': batch['delta_proprio'].detach().cpu(),
                        'emb': batch['embodiment_ids'].detach().cpu(),
                        'episode': torch.as_tensor(batch['trajectory_ids']).cpu(),
                        'base': batch['base_indices'].detach().cpu(),
                    })
                    return
                controls = {'zero': torch.zeros_like(z), 'swapped': z.roll(1, 0)}
                preds = {'real': out['recon']}
                states = {'real': out['s_pred']}
                for name, code in controls.items():
                    preds[name] = lam.decoder(out['dec_in'], code)
                    states[name] = lam.state_decoder(code, batch['states'][:, :1], batch['embodiment_ids'])
            episodes = torch.as_tensor(batch['trajectory_ids'], device=z.device)
            valid_swap = episodes != episodes.roll(1, 0)
            row = {'swap_valid': valid_swap.cpu()}
            target = out['tgt'].float()
            for name in preds:
                row[name + '_visual_mse'] = (preds[name].float() - target).square().flatten(1).mean(1).cpu()
                row[name + '_state_mse'] = (states[name].float()[:, :16] - batch['delta_proprio'][:, :16]).square().mean(1).cpu()
                row[name + '_output_difference_mse'] = (preds[name].float() - preds['real'].float()).square().flatten(1).mean(1).cpu()
            self.z_validation_codes.append(z.detach().float().cpu().flatten(1))
            self.z_validation_rows.append(row)

        def on_before_optimizer_step(self, trainer, pl_module, optimizer):
            if trainer.global_step < 3:
                norms = {}
                for name, module in modules.items():
                    grads = [p.grad for p in module.parameters() if p.grad is not None]
                    norm = torch.stack([g.float().norm() for g in grads]).norm().item() if grads else 0.0
                    if not math.isfinite(norm) or (name != 'DINO' and norm <= 0):
                        raise RuntimeError(f'Invalid gradient in {name}: {norm}')
                    if name == 'DINO' and norm != 0:
                        raise RuntimeError('Frozen DINO unexpectedly has gradients')
                    norms[name] = norm
                self.gradient_checks.append({'step_before_update': trainer.global_step,
                                             'lr_used': optimizer.param_groups[0]['lr'], 'norms': norms})
                write_json(output / 'gradient_checks.json', self.gradient_checks)

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            loss = outputs['loss'] if isinstance(outputs, dict) else outputs
            value = float(loss.detach().float().item())
            if not math.isfinite(value):
                raise RuntimeError(f'Nonfinite training loss at {trainer.global_step}')
            now = time.monotonic()
            self.times.append(now - self.last_time)
            self.last_time = now
            step = trainer.global_step
            if diagnostic:
                hashes = {'data': self.data_sha256, 'aug': self.aug_sha256, 'epsilon': self.epsilon_sha256}
                stream = {'step': step, **{key + '_sha256': value for key, value in hashes.items()}}
                for key, value in hashes.items():
                    self.stream_hashes[key].update(bytes.fromhex(value))
                    stream['cumulative_' + key + '_sha256'] = self.stream_hashes[key].hexdigest()
                with (output / 'training_stream.jsonl').open('a') as f:
                    f.write(json.dumps(stream) + '\n')
            if step <= 8 or step % 20 == 0:
                record = {'phase': 'training', 'step': step, 'pid': os.getpid(),
                          'train_loss': value, 'elapsed_seconds': now - self.started,
                          'recent_seconds_per_update': float(np.median(self.times[-20:])),
                          'lr_next_update': trainer.optimizers[0].param_groups[0]['lr'],
                          'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
                          'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
                          'metrics': {k: float(v) for k, v in trainer.callback_metrics.items()
                                      if isinstance(v, torch.Tensor) and v.numel() == 1}}
                write_json(output / 'progress.json', record)
                with (output / 'progress.jsonl').open('a') as f:
                    f.write(json.dumps(record, allow_nan=False) + '\n')
                print(json.dumps(record, allow_nan=False), flush=True)

        def on_validation_end(self, trainer, pl_module):
            metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if k.startswith('val')}
            if not metrics or not all(math.isfinite(v) for v in metrics.values()):
                raise RuntimeError('Missing or nonfinite validation metrics')
            record = {'step': trainer.global_step, 'metrics': metrics}
            write_json(output / 'validation_latest.json', record)
            with (output / 'validation.jsonl').open('a') as f:
                f.write(json.dumps(record, allow_nan=False) + '\n')
            if diagnostic:
                self.finish_diagnostic_validation(trainer, pl_module)
                random.setstate(self.python_rng)
                np.random.set_state(self.numpy_rng)
                self.validation_rng.__exit__(None, None, None)
            if self.z_validation_rows:
                codes = torch.cat(self.z_validation_codes).double()
                rows = {k: torch.cat([r[k] for r in self.z_validation_rows]) for k in self.z_validation_rows[0]}
                valid = rows.pop('swap_valid')
                z_record = {'step': trainer.global_step, 'samples': len(codes),
                            'different_episode_swap_samples': int(valid.sum()),
                            'mu_variance_sum': float(codes.var(dim=0, unbiased=False).sum()),
                            'mean_dimension_std': float(codes.std(dim=0, unbiased=False).mean()),
                            'metrics': {k: float(v.mean()) for k, v in rows.items() if not k.startswith('swapped_')},
                            'scope': 'Frozen validation inference; zero and within-batch different-episode code swaps. Same weights, not separately trained baselines.'}
                if valid.any():
                    z_record['metrics'].update({k: float(v[valid].mean()) for k, v in rows.items() if k.startswith('swapped_')})
                    for metric in ['visual_mse', 'state_mse']:
                        z_record['metrics']['swapped_minus_real_' + metric] = float((rows['swapped_' + metric] - rows['real_' + metric])[valid].mean())
                write_json(output / 'z_validation_latest.json', z_record)
                with (output / 'z_validation.jsonl').open('a') as f:
                    f.write(json.dumps(z_record, allow_nan=False) + '\n')
                print(json.dumps({'phase': 'z_validation', **z_record}, allow_nan=False), flush=True)
            self.last_time = time.monotonic()
            print(json.dumps({'phase': 'validation', **record}), flush=True)

        def finish_diagnostic_validation(self, trainer, pl_module):
            cache = {key: torch.cat([row[key] for row in self.validation_cache]) for key in self.validation_cache[0]}
            n = len(cache['z'])
            codes = cache['z'].float().flatten(1).double()
            nodes = cache['nodes'].float().flatten(1).double()
            episodes = cache['episode'].numpy()
            rng = np.random.default_rng(1942)
            for _ in range(10000):
                donors = rng.permutation(n)
                if (episodes[donors] != episodes).all():
                    break
            else:
                raise RuntimeError('Could not find a different-episode latent permutation')
            mean = cache['z'].mean(0, keepdim=True)
            variants = {'real': cache['z'], 'zero': torch.zeros_like(cache['z']),
                        'mean': mean.expand_as(cache['z']), 'swapped': cache['z'][donors]}
            metrics = {}
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                for name, latent in variants.items():
                    predictions, state_predictions = [], []
                    if name == 'real':
                        pred, sp = cache['pred'], cache['state_pred']
                    else:
                        for start in range(0, n, data_cfg['batch_size']):
                            sl = slice(start, start + data_cfg['batch_size'])
                            z = latent[sl].to(pl_module.device)
                            predictions.append(pl_module.lam.decoder(cache['x'][sl].to(pl_module.device), z).cpu())
                            state_predictions.append(pl_module.lam.state_decoder(z, cache['states'][sl, :1].to(pl_module.device),
                                                      cache['emb'][sl].to(pl_module.device)).cpu())
                        pred, sp = torch.cat(predictions), torch.cat(state_predictions)
                    metrics[name + '_visual_mse'] = (pred.float() - cache['y'].float()).square().flatten(1).mean(1)
                    metrics[name + '_state_mse'] = (sp.float()[:, :16] - cache['delta'][:, :16]).square().mean(1)
                    metrics[name + '_output_difference_mse'] = (pred.float() - cache['pred'].float()).square().flatten(1).mean(1)
            metrics['persistence_visual_mse'] = (cache['x'].float() - cache['y'].float()).square().flatten(1).mean(1)
            node_var = float(nodes.var(0, unbiased=False).sum())
            node_energy = float(nodes.square().sum(1).mean())
            mu_var = float(codes.var(0, unbiased=False).sum())
            noise_var = float(cache['logvar'].double().exp().flatten(1).sum(1).mean())
            aggregate = {key: float(value.mean()) for key, value in metrics.items()}
            for name in ('mean', 'swapped', 'zero'):
                for metric in ('visual_mse', 'state_mse'):
                    aggregate[name + '_minus_real_' + metric] = float((metrics[name + '_' + metric] - metrics['real_' + metric]).mean())
            z_record = {'step': trainer.global_step, 'samples': n, 'different_episode_swap_samples': n,
                        'mu_variance_sum': mu_var, 'mean_dimension_std': float(codes.std(0, unbiased=False).mean()),
                        'noise_variance_sum': noise_var, 'noise_fraction': noise_var / (noise_var + mu_var),
                        'pre_norm_node_mean_norm': float(nodes.norm(dim=1).mean()),
                        'pre_norm_node_coordinate_std_mean': float(nodes.std(-1, unbiased=False).mean()),
                        'pre_norm_node_variance_fraction': node_var / node_energy if node_energy else 0.,
                        'metrics': aggregate,
                        'scope': 'Fixed validation prefix, same global different-episode permutation seed1942. Mean is diagnostic validation-mean code, not a separately trained forecast baseline.'}
            rows = [{'episode': int(cache['episode'][i]), 'base': int(cache['base'][i]),
                     'swap_valid': True, 'donor_episode': int(cache['episode'][donors[i]]),
                     'donor_base': int(cache['base'][donors[i]]),
                     'metrics': {key: float(value[i]) for key, value in metrics.items()}} for i in range(n)]
            write_json(output / f'z_validation_step{trainer.global_step:06d}.json', {**z_record, 'rows': rows})
            write_json(output / 'z_validation_latest.json', z_record)
            with (output / 'z_validation.jsonl').open('a') as f:
                f.write(json.dumps(z_record, allow_nan=False) + '\n')
            print(json.dumps({'phase': 'z_validation', **z_record}, allow_nan=False), flush=True)
            self.validation_cache = []

        def on_train_end(self, trainer, pl_module):
            for hook in self.vae_hooks:
                hook.remove()

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            checkpoint['data_contract'] = contract

    class KeepBestAndLast(ModelCheckpoint):
        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking or trainer.global_step % trainer_cfg['val_check_interval']:
                return
            if self._should_skip_saving_checkpoint(trainer):
                return
            candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, candidates)
            # Refresh latest even when validation did not improve (Lightning 2.6).
            self._save_last_checkpoint(trainer, candidates)

    monitor = Monitor()
    callbacks = [monitor]
    if args.preflight:
        trainer_cfg.update(max_steps=6, val_check_interval=6, limit_val_batches=2,
                           enable_checkpointing=False, log_every_n_steps=1)
    elif diagnostic:
        trainer_cfg['enable_checkpointing'] = False
    else:
        callbacks.append(KeepBestAndLast(dirpath=output / 'checkpoints', filename='best-{step:06d}',
                         monitor='val_loss', mode='min', save_top_k=1, save_last=True,
                         save_on_train_epoch_end=False))
    (output / 'effective_trainer.yaml').write_text(yaml.safe_dump(trainer_cfg, sort_keys=False))
    trainer = Trainer(**trainer_cfg, default_root_dir=str(output), callbacks=callbacks,
                      logger=CSVLogger(str(output), name='metrics'))
    torch.cuda.reset_peak_memory_stats()
    trainer.fit(model, datamodule=dm)
    if not args.preflight:
        trainer.save_checkpoint(output / 'checkpoints/final.ckpt')
    report = {'phase': 'preflight_passed' if args.preflight else 'complete',
              'step': trainer.global_step, 'gradient_checks': monitor.gradient_checks,
              'seconds_per_update_excluding_first': float(np.median(monitor.times[1:])),
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved(),
              'parameters': parameters, 'counts': counts,
              'torch': torch.__version__, 'cuda': torch.version.cuda,
              'metrics': {k: float(v) for k, v in trainer.callback_metrics.items()}}
    write_json(output / 'run_summary.json', report)
    print(json.dumps(report, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
