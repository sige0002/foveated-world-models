"""Run the pinned official Stage-1 model with timestamp-indexed RoboMIND videos."""
from __future__ import annotations

import argparse
import functools
import json
import os
from pathlib import Path
import sys
import time
import hashlib

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'vendor/lawam'))
# No model downloads or external experiment logging, even with upstream defaults.
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['LAM_ENABLE_MANUAL_WANDB'] = '0'


def merge(base, override):
    result = dict(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_config(path):
    import yaml
    path = Path(path).resolve()
    config = yaml.safe_load(path.read_text())
    parent = config.pop('extends', None)
    return merge(load_config(path.parent / parent), config) if parent else config


def check_resume_contract(checkpoint, contract):
    if checkpoint.get('data_contract') != contract:
        raise ValueError('Checkpoint data contract differs or is absent. Start a new run; '
                         'legacy rot6D/padded-image checkpoints cannot resume corrected training.')


def relocated_checkpoint_state(state, directory):
    """Preserve top-k bookkeeping when an old path aliases the moved directory."""
    if not state.get('dirpath') or Path(state['dirpath']).resolve() != Path(directory).resolve():
        return state
    state = dict(state)
    state['dirpath'] = str(directory)
    for key in ('best_model_path', 'kth_best_model_path', 'last_model_path'):
        if state.get(key):
            state[key] = str(Path(state[key]).resolve())
    if 'best_k_models' in state:
        state['best_k_models'] = {str(Path(p).resolve()): score for p, score in state['best_k_models'].items()}
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(ROOT / 'configs/stage1_robomind.yaml'))
    parser.add_argument('--train-manifest')
    parser.add_argument('--val-manifest')
    parser.add_argument('--vision-model', help='Existing local DINOv3 directory; no downloads')
    parser.add_argument('--output', default=str(ROOT / 'runs/stage1'))
    parser.add_argument('--resume', help='Lightning .ckpt to resume optimizer and step state')
    parser.add_argument('--check-data', action='store_true', help='Validate all images and labels without loading DINO')
    args = parser.parse_args()

    import yaml
    from torch.utils.data import DataLoader
    from vendor.dataset.robomind import RoboMINDDataset, check_disjoint, robomind_collate

    config = load_config(args.config)
    model_config = config['model']
    data_config = config['data']
    if args.vision_model:
        model_config['vision_model_id'] = args.vision_model
    manifests = [args.train_manifest or data_config.get('train_manifest'),
                 args.val_manifest or data_config.get('val_manifest')]
    options = dict(image_hw=model_config['image_hw'], max_state_dim=model_config['max_state_dim'],
                   num_embodiments=model_config['num_embodiments'],
                   horizon_seconds=data_config['horizon_seconds'],
                   tolerance_seconds=data_config['tolerance_seconds'])
    contract = None
    if data_config.get('format') == 'lerobot_v3':
        if any(manifests):
            parser.error('LeRobot uses data.root and its episode metadata, not NPZ manifests')
        from vendor.dataset.lerobot_robomind import LeRobotSource, LeRobotRoboMINDDataset, lerobot_collate
        source = LeRobotSource(data_config['root'])
        train = LeRobotRoboMINDDataset(source, 'train', **options)
        val = LeRobotRoboMINDDataset(source, 'validation', **options)
        contract = train.contract
        collate_fn = lerobot_collate
    elif data_config.get('format') == 'robomind_video':
        if not all(manifests):
            parser.error('Legacy format requires train/val manifests')
        train = RoboMINDDataset(manifests[0], **options)
        val = RoboMINDDataset(manifests[1], require_observed_endpoints=True, **options)
        collate_fn = robomind_collate
    else:
        parser.error('data.format must be lerobot_v3 or legacy robomind_video')
    collate = functools.partial(collate_fn, max_state_dim=model_config['max_state_dim'])
    check_disjoint(train, val)
    if args.check_data:
        for dataset in (train, val):
            for index in range(len(dataset)):
                collate([dataset[index]])
        print(json.dumps({'train_pairs': len(train), 'val_pairs': len(val), 'status': 'valid'}))
        return

    vision = Path(model_config['vision_model_id']).expanduser().resolve()
    if not (vision / 'config.json').is_file() or not any(vision.glob('*.safetensors')):
        parser.error(f'Local model config/safetensors not found: {vision}')
    vc = json.loads((vision / 'config.json').read_text())
    expected = config['expected_vision']
    for key, value in expected.items():
        if vc.get(key) != value:
            parser.error(f'DINO {key}={vc.get(key)!r}; this config requires {value!r}')
    if 'dinov3' not in str(vision).lower():
        parser.error('Upstream selects its encoder by path name; local path must contain dinov3')
    model_config['vision_model_id'] = str(vision)

    import torch
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import Callback, ModelCheckpoint
    from lightning.pytorch.loggers import CSVLogger
    from latent_action_model.core.lam_lightinng import VJEPA_LAM

    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=True, mmap=True)
        if contract is not None or saved.get('data_contract') is not None:
            check_resume_contract(saved, contract)
        del saved

    class DataContractCheckpoint(Callback):
        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            checkpoint['data_contract'] = contract

    class RelocatableModelCheckpoint(ModelCheckpoint):
        def _generate_state_key(self, **kwargs):
            # Keep compatibility with checkpoints saved by the original callback.
            return f"ModelCheckpoint{repr(kwargs)}"

        def load_state_dict(self, state_dict):
            super().load_state_dict(relocated_checkpoint_state(state_dict, self.dirpath))

        def on_validation_end(self, trainer, pl_module):
            interval = config.get('checkpoint_every_n_steps')
            if interval is None:
                return super().on_validation_end(trainer, pl_module)
            if trainer.sanity_checking or trainer.global_step % int(interval):
                return
            if self._should_skip_saving_checkpoint(trainer):
                return
            candidates = self._monitor_candidates(trainer)
            self._save_topk_checkpoint(trainer, candidates)
            # Lightning 2.6 otherwise refreshes last only when top-k improved.
            self._save_last_checkpoint(trainer, candidates)

    class TrainingProgress(Callback):
        def on_train_start(self, trainer, pl_module):
            self.started = time.monotonic()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            step = trainer.global_step
            if step == 1 or step % 20 == 0:
                loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
                record = {'step': step, 'elapsed_seconds': time.monotonic() - self.started,
                          'train_loss': float(loss.detach().cpu()) if isinstance(loss, torch.Tensor) else None}
                temporary = output / 'progress.tmp'
                temporary.write_text(json.dumps(record, allow_nan=False))
                temporary.replace(output / 'progress.json')
                print(json.dumps(record, allow_nan=False), flush=True)

    output = Path(args.output).resolve()
    if contract is not None and (output / 'resolved_config.yaml').exists():
        previous = output / 'data_contract.json'
        if not previous.exists() or json.loads(previous.read_text()) != contract:
            raise ValueError('Output contains a different data contract. Use a new output directory.')
    output.mkdir(parents=True, exist_ok=True)
    seed_everything(config['seed'], workers=True)
    split_report = {
        name: {'metadata': str(dataset.path), 'sha256': hashlib.sha256(dataset.path.read_bytes()).hexdigest(),
               'episodes': len(dataset.episodes), 'video_groups': len(dataset.groups), 'pairs': len(dataset),
               'tail_pairs': sum(p[3] for p in dataset.pairs)}
        for name, dataset in (('train', train), ('validation', val))
    }
    split_report['episode_and_video_groups_disjoint'] = True
    (output / 'split_verification.json').write_text(json.dumps(split_report, indent=2))
    if contract is not None:
        (output / 'data_contract.json').write_text(json.dumps(contract, indent=2))
        config['data']['root'] = str(source.root)
    else:
        for key, path in zip(('train_manifest', 'val_manifest'), manifests):
            config['data'][key] = str(Path(path).resolve())
    (output / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    opt = {key: float(value) for key, value in config['optimizer'].items()}
    model = VJEPA_LAM(**model_config, optimizer=functools.partial(torch.optim.AdamW, **opt),
                     weight_decay=opt['weight_decay'])
    # Upstream saves the optimizer callable in hparams. Keep checkpoints compatible
    # with weights-only loading; numeric optimizer settings live in resolved_config.
    model.hparams.pop('optimizer', None)
    modules = {'IDM': model.lam.encoder, 'LaWM': model.lam.decoder,
               'state_predictor': model.lam.state_decoder, 'DINO': model.lam.vision_encoder}
    parameters = {name: {'total': sum(p.numel() for p in module.parameters()),
                         'trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)}
                  for name, module in modules.items()}
    if parameters['DINO']['trainable']:
        raise RuntimeError('DINO must remain frozen in Stage 1')
    if any(parameters[name]['trainable'] == 0 for name in ('IDM', 'LaWM', 'state_predictor')):
        raise RuntimeError('IDM, LaWM and state predictor must be trainable')
    loaders = [DataLoader(dataset, batch_size=data_config['batch_size'], shuffle=shuffle,
                          num_workers=data_config['num_workers'], collate_fn=collate,
                          pin_memory=torch.cuda.is_available())
               for dataset, shuffle in ((train, True), (val, False))]
    checkpoint = RelocatableModelCheckpoint(dirpath=output / 'checkpoints', save_last=True,
                                 save_top_k=1, monitor='val_loss', mode='min',
                                 save_on_train_epoch_end=False)
    trainer = Trainer(**config['trainer'], default_root_dir=output,
                      logger=CSVLogger(str(output), name='metrics'),
                      callbacks=[checkpoint, DataContractCheckpoint(), TrainingProgress()])
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    trainer.fit(model, train_dataloaders=loaders[0], val_dataloaders=loaders[1], ckpt_path=args.resume)
    fit_seconds = time.perf_counter() - started
    trainer.save_checkpoint(output / 'checkpoints/final.ckpt')
    report = {
        'global_step': trainer.global_step, 'fit_seconds_including_validation': fit_seconds,
        'parameters': parameters,
        'metrics': {k: float(v.detach().cpu()) for k, v in trainer.callback_metrics.items() if v.numel() == 1},
        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
        'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
        'torch_version': torch.__version__, 'cuda_version': torch.version.cuda,
        'checkpoint': str(output / 'checkpoints/final.ckpt'),
    }
    (output / 'run_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
