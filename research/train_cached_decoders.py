"""Matched-input decoder-only comparison using an immutable trained-IDM cache."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'vendor/lawam'))
import torch
from torch.nn import functional as F
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from latent_action_model.core.utils.latent_flow import LatentFlowDecoder, DeterministicRefinementDecoder


def file_hash(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def make_decoder(kind, dim, layers, feature_dim, code_dim, grid):
    options = dict(context_dim=dim, input_dim=feature_dim, code_dim=code_dim,
                   num_layers=layers, num_heads=4, grid_hw=(grid, grid),
                   dropout=0.0, ffn_expansion_factor=4)
    if kind.startswith('direct'):
        return LAMDecoder_v2(**options, last_ln=kind == 'direct')
    cls = LatentFlowDecoder if kind == 'flow' else DeterministicRefinementDecoder
    return cls(**options, source='current', noise_scale=0.1, steps=1 if kind == 'residual' else 4)


@torch.inference_mode()
def evaluate(model, data, batch_size=8):
    model.eval()
    gen = torch.Generator().manual_seed(2026)
    rows = []
    stochastic = type(model) is LatentFlowDecoder
    for start in range(0, len(data['current']), batch_size):
        h, target, z = [data[key][start:start+batch_size] for key in ('current','future','codes')]
        # Globally roll codes, not within each batch (last batch may have one item).
        idx = (torch.arange(start, start+len(h))+1) % len(data['codes'])
        wrong_z = data['codes'][idx]
        predictions, wrong = [], []
        for _ in range(4 if stochastic else 1):
            options = dict(noise=torch.randn(h.shape, generator=gen)) if stochastic else {}
            predictions.append(model(h, z, **options))
            wrong.append(model(h, wrong_z, **options))
        pred, shuffled = torch.stack(predictions), torch.stack(wrong)
        mse = (pred-target).square().flatten(2).mean(2).mean(0)
        mean_mse = (pred.mean(0)-target).square().flatten(1).mean(1)
        wrong_mse = (shuffled-target).square().flatten(2).mean(2).mean(0)
        copy = (h-target).square().flatten(1).mean(1)
        for i in range(len(h)):
            rows.append(dict(**data['rows'][start+i], mse=float(mse[i]), mean_mse=float(mean_mse[i]),
                             shuffled_mse=float(wrong_mse[i]), copy_mse=float(copy[i])))
    return dict(metrics={k:sum(r[k] for r in rows)/len(rows)
                         for k in ('mse','mean_mse','shuffled_mse','copy_mse')}, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--variants', nargs='+', choices=['direct','direct_unorm','residual','refinement','flow'],
                        default=['direct','direct_unorm','residual','refinement','flow'])
    args = parser.parse_args()
    if min(args.steps, args.batch_size, args.dim, args.layers) < 1 or args.dim % 4:
        parser.error('positive settings required; dim must be divisible by 4')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    manifest = json.loads((args.cache/'cache_manifest.json').read_text())
    for name in ('train','validation'):
        if file_hash(args.cache/f'{name}.pt') != manifest['cache_sha256'][name]:
            raise ValueError('Cache hash mismatch: '+name)
    train = torch.load(args.cache/'train.pt', weights_only=True)
    val = torch.load(args.cache/'validation.pt', weights_only=True)
    assert {r['episode'] for r in train['rows']}.isdisjoint({r['episode'] for r in val['rows']})
    _, _, tokens, features = train['current'].shape
    grid = math.isqrt(tokens)
    if grid*grid != tokens:
        raise ValueError('Square token grid required')
    report = dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  cache_manifest=manifest, results=[])
    for seed in args.seeds:
        for kind in args.variants:
            torch.manual_seed(seed)
            model = make_decoder(kind, args.dim, args.layers, features, train['codes'].shape[-1], grid)
            groups = [dict(params=[p for p in model.parameters() if p.ndim >= 2], weight_decay=0.01),
                      dict(params=[p for p in model.parameters() if p.ndim < 2], weight_decay=0.)]
            optimizer = torch.optim.AdamW(groups, lr=3e-4)
            indices_gen = torch.Generator().manual_seed(seed+1000)
            history = []
            started = time.perf_counter()
            for step in range(1, args.steps+1):
                model.train()
                idx = torch.randint(len(train['current']), (args.batch_size,), generator=indices_gen)
                h, target, z = [train[key][idx] for key in ('current','future','codes')]
                optimizer.zero_grad(set_to_none=True)
                if kind == 'flow':
                    loss = model.training_output(h, z, target).velocity_loss
                else:
                    loss = F.mse_loss(model(h, z), target)
                if not torch.isfinite(loss):
                    raise RuntimeError('Nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                optimizer.step()
                if step % 100 == 0 or step == args.steps:
                    validation = evaluate(model, val, args.batch_size)
                    history.append(dict(step=step, train_objective=float(loss.detach()), **validation['metrics']))
                    print(seed,kind,history[-1],flush=True)
            model.eval()
            train_eval = evaluate(model, train, args.batch_size)
            result = dict(seed=seed, variant=kind, history=history, validation=validation,
                          train_metrics=train_eval['metrics'], elapsed_seconds=time.perf_counter()-started,
                          parameters=sum(p.numel() for p in model.parameters()))
            report['results'].append(result)
            torch.save(dict(state_dict=model.state_dict(), variant=kind, dim=args.dim, layers=args.layers,
                            seed=seed, steps=args.steps), args.output/f'{kind}-seed{seed}.pt')
            (args.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
