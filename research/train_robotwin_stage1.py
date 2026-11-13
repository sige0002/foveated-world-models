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
    if data_cfg['batch_size'] != 4 or trainer_cfg['accumulate_grad_batches'] != 1:
        raise ValueError('This run is authorized for batch 4 without gradient accumulation')
    if model_cfg['state_output_scale'] != 2 or model_cfg['warmup_steps'] != 1000:
        raise ValueError('Expected 2*tanh and warmup 1000')
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
    (output / 'resolved_config.yaml').write_text(yaml.safe_dump(resolved, sort_keys=False))
    print(json.dumps({'phase': 'split', 'counts': counts, 'overlap': 0}), flush=True)
    install_episode_split(split)

    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from latent_action_model.data_loader.lerobot_datamodule import LeRobotDataModule
    from latent_action_model.core.lam_lightinng import VJEPA_LAM

    torch.set_num_threads(cfg['cpu_threads'])
    torch.set_num_interop_threads(1)
    seed_everything(cfg['seed'], workers=True)

    class SplitDataModule(LeRobotDataModule):
        def setup(self, stage=None):
            if self.train_dataset is None:
                super().setup(stage)

        def val_dataloader(self):
            # Fixed validation prefix must not depend on worker completion order.
            prior = self.in_order
            self.in_order = True
            loader = super().val_dataloader()
            self.in_order = prior
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

    class Monitor(Callback):
        def __init__(self):
            self.started = time.monotonic()
            self.times = []
            self.last_time = None
            self.gradient_checks = []

        def on_train_start(self, trainer, pl_module):
            self.started = self.last_time = time.monotonic()

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
                self.gradient_checks.append({'step_before_update': trainer.global_step, 'norms': norms})
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
            self.last_time = time.monotonic()
            print(json.dumps({'phase': 'validation', **record}), flush=True)

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            checkpoint['data_contract'] = contract

    class KeepBestAndLast(ModelCheckpoint):
        def on_validation_end(self, trainer, pl_module):
            if trainer.sanity_checking or trainer.global_step % 2000:
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
