"""Freeze a trained Stage1 IDM and cache paired features for decoder-only experiments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['LAM_ENABLE_MANUAL_WANDB'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'vendor/lawam')]
import numpy as np
import torch
import yaml
from vendor.dataset.robomind import RoboMINDDataset, check_disjoint, robomind_collate
from latent_action_model.core.lam_lightinng import VJEPA_LAM


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


@torch.inference_mode()
def extract(model, dataset, count, label):
    indices = np.linspace(0, len(dataset)-1, min(count, len(dataset)), dtype=int)
    current, future, codes, rows = [], [], [], []
    for n, index in enumerate(indices):
        sample = dataset[int(index)]
        batch = model.transfer_batch_to_device(robomind_collate([sample],
                max_state_dim=model.lam.encoder.max_state_dim), torch.device('cpu'), 0)
        out = model.lam._run(batch['videos'], batch['states'], batch['dec_videos'],
                            state_mask=batch['state_mask'], embodiment_ids=batch['embodiment_ids'],
                            vq_training=False, predict_future_frame=False)
        h, target, z = out[1], out[2], out[7]
        current.append(h.cpu().clone())
        future.append(target.cpu().clone())
        codes.append(z.cpu().clone())
        row = dict(index=int(index), episode=str(sample['trajectory_id']), dt=sample['dt'])
        if label == 'validation':
            pred = model.lam.decoder(h, z)
            row['reference_mse'] = float((pred-target).square().mean())
            row['reference_zero_z_mse'] = float((model.lam.decoder(h, torch.zeros_like(z))-target).square().mean())
        rows.append(row)
        if (n+1) % 8 == 0 or n+1 == len(indices):
            print(f'{label}: {n+1}/{len(indices)}', flush=True)
    tensors = dict(current=torch.cat(current), future=torch.cat(future), codes=torch.cat(codes))
    assert all(torch.isfinite(t).all() for t in tensors.values())
    return dict(**tensors, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True, help='Named immutable step checkpoint, not last.ckpt')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--train-pairs', type=int, default=128)
    parser.add_argument('--val-pairs', type=int, default=64)
    args = parser.parse_args()
    if min(args.train_pairs, args.val_pairs) < 2:
        parser.error('pair counts must be at least two')
    if args.checkpoint.name == 'last.ckpt':
        parser.error('Use an immutable step checkpoint while training is running')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    config = yaml.safe_load((args.run/'resolved_config.yaml').read_text())
    model = VJEPA_LAM(**config['model']).eval().requires_grad_(False)
    # Holding the file open preserves this checkpoint even if training removes its name.
    with args.checkpoint.open('rb') as handle:
        saved = torch.load(handle, map_location='cpu', weights_only=True)
        handle.seek(0)
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    model.load_state_dict(saved['state_dict'], strict=True)
    step = saved['global_step']
    del saved
    torch.save(dict(state_dict=model.state_dict(), global_step=step), args.output/'reference_weights.pt')
    (args.output/'resolved_config.yaml').write_text(yaml.safe_dump(config))
    options = dict(image_hw=config['model']['image_hw'], max_state_dim=config['model']['max_state_dim'],
                   num_embodiments=config['model']['num_embodiments'],
                   horizon_seconds=config['data']['horizon_seconds'],
                   tolerance_seconds=config['data']['tolerance_seconds'], require_observed_endpoints=True)
    train = RoboMINDDataset(config['data']['train_manifest'], **options)
    val = RoboMINDDataset(config['data']['val_manifest'], **options)
    check_disjoint(train, val)
    started = time.perf_counter()
    for name, dataset, count in [('train', train, args.train_pairs), ('validation', val, args.val_pairs)]:
        result = extract(model, dataset, count, name)
        torch.save(result, args.output/f'{name}.pt')
        del result
    report = dict(checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest.hexdigest(),
                  checkpoint_step=step, model_config=config['model'], augmentation=False,
                  code_source='frozen posterior IDM Gaussian mean; sees true future',
                  selection='uniform indices across each split; observed endpoints only',
                  manifests={name: dict(path=str(ds.path), sha256=sha256(ds.path))
                             for name, ds in [('train',train),('validation',val)]},
                  cache_sha256={name:sha256(args.output/f'{name}.pt') for name in ['train','validation']},
                  extraction_seconds=time.perf_counter()-started)
    (args.output/'cache_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)


if __name__ == '__main__':
    main()
