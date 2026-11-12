"""Paired feature errors and action-code ablations on uniformly sampled validation pairs.

Posterior reconstruction only: the IDM still observes the true future.
All stochastic comparisons reuse noise; no best-of-N selection is used.
"""
import argparse
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
from vendor.dataset.robomind import RoboMINDDataset, robomind_collate
from latent_action_model.core.lam_lightinng import VJEPA_LAM
from latent_action_model.core.utils.latent_flow import LatentFlowDecoder, DeterministicRefinementDecoder


@torch.inference_mode()
def evaluate(run, count, samples, seed):
    config = yaml.safe_load((run/'resolved_config.yaml').read_text())
    model = VJEPA_LAM(**config['model']).eval()
    checkpoint = torch.load(run/'checkpoints/final.ckpt', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    del checkpoint
    dataset = RoboMINDDataset(config['data']['val_manifest'],
                              image_hw=config['model']['image_hw'],
                              max_state_dim=config['model']['max_state_dim'],
                              num_embodiments=config['model']['num_embodiments'],
                              horizon_seconds=config['data']['horizon_seconds'],
                              tolerance_seconds=config['data']['tolerance_seconds'],
                              require_observed_endpoints=True)
    indices = np.linspace(0, len(dataset)-1, min(count, len(dataset)), dtype=int)
    cache = []
    for index in indices:
        sample = dataset[int(index)]
        batch = model.transfer_batch_to_device(robomind_collate([sample],
                    max_state_dim=config['model']['max_state_dim']), torch.device('cpu'), 0)
        out = model.lam._run(batch['videos'], batch['states'], batch['dec_videos'],
                            state_mask=batch['state_mask'], embodiment_ids=batch['embodiment_ids'],
                            vq_training=False, predict_future_frame=False)
        cache.append((out[1], out[2], out[7], sample))
    decoder = model.lam.decoder
    stochastic = isinstance(decoder, LatentFlowDecoder) and not isinstance(decoder, DeterministicRefinementDecoder)
    generator = torch.Generator().manual_seed(seed)
    rows = []
    for i, (current, target, action, meta) in enumerate(cache):
        predictions, zero_predictions, rolled_predictions = [], [], []
        other_action = cache[(i+1) % len(cache)][2]
        elapsed = 0.
        for _ in range(samples if stochastic else 1):
            noise = torch.randn(current.shape, generator=generator)
            options = dict(noise=noise) if stochastic else {}
            start = time.perf_counter()
            prediction = decoder(current, action, **options)
            elapsed += time.perf_counter()-start
            predictions.append(prediction)
            zero_predictions.append(decoder(current, torch.zeros_like(action), **options))
            rolled_predictions.append(decoder(current, other_action, **options))
        pred = torch.stack(predictions)
        zero = torch.stack(zero_predictions)
        rolled = torch.stack(rolled_predictions)
        rows.append(dict(index=int(indices[i]), episode=str(meta['trajectory_id']), dt=meta['dt'],
                         copy_mse=float((current-target).square().mean()),
                         sample_mse=float((pred-target).square().mean()),
                         mean_mse=float((pred.mean(0)-target).square().mean()),
                         zero_z_mse=float((zero-target).square().mean()),
                         rolled_z_mse=float((rolled-target).square().mean()),
                         z_sensitivity=float((pred-zero).square().mean()),
                         sample_variance=float(pred.var(0, unbiased=False).mean()),
                         seconds_per_sample=elapsed/len(predictions)))
    metrics = ('copy_mse', 'sample_mse', 'mean_mse', 'zero_z_mse', 'rolled_z_mse',
               'z_sensitivity', 'sample_variance', 'seconds_per_sample')
    return dict(run=str(run.resolve()), seed=seed, requested_samples=samples,
                pairs=len(rows), episodes=len({r['episode'] for r in rows}),
                sampling='uniform indices across validation; posterior IDM; CPU float32',
                metrics={key: float(np.mean([r[key] for r in rows])) for key in metrics}, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=64)
    parser.add_argument('--samples', type=int, default=4)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.pairs < 2 or args.samples < 1:
        parser.error('pairs must be >=2 and samples >=1')
    if args.output.exists():
        parser.error('output already exists')
    torch.set_num_threads(4)
    results = []
    for run in args.runs:
        torch.manual_seed(args.seed)
        result = evaluate(run, args.pairs, args.samples, args.seed)
        results.append(result)
        print(run.name, json.dumps(result['metrics']), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
