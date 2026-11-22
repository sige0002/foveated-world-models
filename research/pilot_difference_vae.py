"""Paired RoboTwin cached-feature Stage 1 pilot.

From the repository root, with the Stage 1 environment activated, run
PYTHONPATH=.:vendor/lawam python -m research.pilot_difference_vae TRAIN VAL OUTPUT

TRAIN is matched-state-probes-fp32-20261114/train2048-highest-fp32-features.pt.
VAL is final3000-highest-fp32/features512-highest-fp32.pt. Both are under
robotwin-loss-ablation-b4-20261114/evaluations on the experiment drive.
"""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn
from latent_action_model.core.utils.lam_encoder import LAMEncoder
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2, StatePredictor
from latent_action_model.core.vq import VAEQuantizer
from research.idm.difference_idm import DifferenceIDM


class Pilot(nn.Module):
    def __init__(self, representation):
        super().__init__()
        dims = dict(context_dim=128, input_dim=768, num_layers=2, num_heads=4,
                    ffn_expansion_factor=4, dropout=0., grid_hw=(16, 16), code_dim=32)
        encoder = LAMEncoder(**dims, num_frames=2, modal_mask=True)
        self.idm = DifferenceIDM(encoder, nn.Identity(), 1., representation)
        self.vq = VAEQuantizer(code_dim=32, layer_norm=True)
        self.decoder = LAMDecoder_v2(**dims)
        self.g = StatePredictor(latent_dim=128, code_dim=32, dropout=0.)
        if hasattr(self.g, 'output_scale'):
            self.g.output_scale = 2.


def batch(cache, ids, training):
    x, y, delta = [cache[k][ids].cuda() for k in ('x', 'y', 'delta')]
    if training:
        state = F.pad(cache['state'][ids].cuda(), (0, 16))[:, None]
    else:
        state = cache['states'][ids, :1].cuda()
        assert cache['mask'][ids, :16].all() and not cache['mask'][ids, 16:].any()
        assert (cache['emb'][ids] == 1).all()
    return x, y, state, delta[:, :16]


def predictions(model, x, state, z):
    emb = torch.ones(len(x), device=x.device, dtype=torch.long)
    return model.decoder(x, z), model.g(z, state, emb).reshape(len(x), 32)[:, :16]


@torch.no_grad()
def evaluate(model, cache, ids):
    model.eval()
    codes = []
    for ix in ids.split(8):
        x, y, _, _ = batch(cache, ix, False)
        codes.append(model.vq._encode(model.idm(x, y))[0])
    codes = torch.cat(codes)
    swapped = codes.roll(1, 0)
    metrics = {}
    for name, latent in [('real', codes), ('swapped', swapped)]:
        errors = []
        for ix, z in zip(ids.split(8), latent.split(8)):
            x, y, state, delta = batch(cache, ix, False)
            visual, action = predictions(model, x, state, z)
            errors.append(torch.stack([(visual-y).square().mean(), (action-delta).square().mean()]))
        metrics[name] = dict(zip(('visual_mse', 'state_mse'), torch.stack(errors).mean(0).tolist()))
    return metrics


def run(args):
    train, val = [torch.load(p, mmap=True, weights_only=False, map_location='cpu')
                  for p in (args.train_cache, args.val_cache)]
    episodes = {row['episode'] for row in train['rows']}
    seen = set()
    indices = []
    for i, row in enumerate(val['rows']):
        ep = row['episode']
        assert ep not in episodes
        if ep not in seen:
            seen.add(ep)
            indices.append(i)
    ids = torch.tensor(indices[:128])
    assert len(ids) == 128
    batches = torch.randint(len(train['rows']), (args.steps, 8),
                            generator=torch.Generator().manual_seed(args.seed + 1))
    result = dict(seed=args.seed, steps=args.steps, batch_size=8, eval_episodes=len(ids),
                  train_identity=train.get('identity_sha256'), val_indices=ids.tolist(), arms={})
    for arm in ('raw', 'difference'):
        torch.manual_seed(args.seed)
        model = Pilot(arm).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        for step, ix in enumerate(batches, 1):
            x, y, state, delta = batch(train, ix, True)
            z, _, _, _, kl = model.vq(model.idm(x, y))
            visual, action = predictions(model, x, state, z)
            loss = F.smooth_l1_loss(visual, y, beta=.1) + F.mse_loss(action, delta) + kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if step in (2, args.steps):
                print(arm, step, loss.item(), time.monotonic()-start,
                      torch.cuda.max_memory_allocated(), flush=True)
        seconds = time.monotonic()-start
        result['arms'][arm] = dict(train_seconds=seconds, metrics=evaluate(model, val, ids))
        del model, optimizer
    print(json.dumps(result['arms'], indent=2))
    args.output.write_text(json.dumps(result, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_cache', type=Path)
    parser.add_argument('val_cache', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--steps', type=int, default=800)
    parser.add_argument('--seed', type=int, default=42)
    run(parser.parse_args())
