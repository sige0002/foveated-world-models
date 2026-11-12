"""Matched-input decoder-only comparison using an immutable trained-IDM cache."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'vendor/lawam'))
import torch
from torch.nn import functional as F
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from latent_action_model.core.utils.latent_flow import LatentFlowDecoder, DeterministicRefinementDecoder
from research.endpoint_flow import EndpointFlowDecoder
from research.wide_head_flow import WideHeadFlowDecoder
from research.prior_correction import PriorCorrectionDecoder
from research.centered_prior_correction import CenteredPriorCorrectionDecoder
from research.spatial_tangent_decoder import SpatialTangentDecoder
from research.projected_prior_decoder import ProjectedPriorDecoder
from research.horizon_prior_decoder import HorizonPriorDecoder


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
    if kind in ('prior_horizon_residual', 'prior_horizon_flow'):
        return HorizonPriorDecoder(**options, source='current', noise_scale=0.1,
                                   steps=1 if kind == 'prior_horizon_residual' else 4,
                                   deterministic=kind == 'prior_horizon_residual')
    if kind in ('prior_projected_residual', 'prior_projected_flow'):
        return ProjectedPriorDecoder(**options, source='current', noise_scale=0.1,
                                     steps=1 if kind == 'prior_projected_residual' else 4,
                                     deterministic=kind == 'prior_projected_residual')
    if kind in ('prior_tangent_residual', 'prior_rotated_residual'):
        return SpatialTangentDecoder(**options, source='current', noise_scale=0.1,
                                     steps=1, deterministic=True,
                                     rotated=kind == 'prior_rotated_residual')
    if kind.startswith('direct'):
        return LAMDecoder_v2(**options, last_ln=kind == 'direct')
    if kind in ('prior_centered_flow', 'prior_centered_residual'):
        return CenteredPriorCorrectionDecoder(**options, source='current', noise_scale=0.1,
                                              steps=1 if kind == 'prior_centered_residual' else 4,
                                              deterministic=kind == 'prior_centered_residual')
    if kind in ('prior_flow', 'prior_residual', 'prior_terminal', 'prior_refinement'):
        return PriorCorrectionDecoder(**options, source='current', noise_scale=0.1,
                                      steps=1 if kind == 'prior_residual' else 4,
                                      deterministic=kind in ('prior_residual', 'prior_refinement'))
    if kind in ('wide_flow', 'wide_residual', 'wide_refinement', 'wide_terminal'):
        return WideHeadFlowDecoder(**options, source='current', noise_scale=0.1,
                                   steps=1 if kind == 'wide_residual' else 4,
                                   deterministic=kind in ('wide_residual', 'wide_refinement'))
    if kind in ('endpoint_anchored', 'noisy_residual'):
        return EndpointFlowDecoder(**options, source='current', noise_scale=0.1,
                                   steps=1 if kind == 'noisy_residual' else 4,
                                   source_time_probability=1.0 if kind == 'noisy_residual' else 0.5)
    cls = {'flow': LatentFlowDecoder, 'endpoint_flow': EndpointFlowDecoder}.get(kind, DeterministicRefinementDecoder)
    return cls(**options, source='current', noise_scale=0.1, steps=1 if kind == 'residual' else 4)


@torch.inference_mode()
def evaluate(model, data, batch_size=8):
    model.eval()
    gen = torch.Generator().manual_seed(2026)
    rows = []
    stochastic = getattr(model, 'stochastic', isinstance(model, LatentFlowDecoder)
                         and not isinstance(model, DeterministicRefinementDecoder))
    for start in range(0, len(data['current']), batch_size):
        h, target, z = [data[key][start:start+batch_size] for key in ('current','future','codes')]
        # Globally roll codes, not within each batch (last batch may have one item).
        idx = (torch.arange(start, start+len(h))+1) % len(data['codes'])
        wrong_z = data['codes'][idx]
        predictions, wrong = [], []
        for _ in range(4 if stochastic else 1):
            options = dict(noise=torch.randn(h.shape, generator=gen)) if stochastic else {}
            if getattr(model, 'requires_baseline', False):
                options['baseline'] = data['baseline'][start:start+len(h)]
            if getattr(model, 'requires_physical_time', False):
                options['physical_time'] = h.new_tensor([r['dt'] for r in data['rows'][start:start+len(h)]])
            predictions.append(model(h, z, **options))
            wrong.append(model(h, wrong_z, **options))
        pred, shuffled = torch.stack(predictions), torch.stack(wrong)
        mse = (pred-target).square().flatten(2).mean(2).mean(0)
        mean_mse = (pred.mean(0)-target).square().flatten(1).mean(1)
        wrong_mse = (shuffled-target).square().flatten(2).mean(2).mean(0)
        copy = (h-target).square().flatten(1).mean(1)
        prior_error = ((data['baseline'][start:start+len(h)]-target).square().flatten(1).mean(1)
                       if 'baseline' in data else None)
        for i in range(len(h)):
            rows.append(dict(**data['rows'][start+i], mse=float(mse[i]), mean_mse=float(mean_mse[i]),
                             shuffled_mse=float(wrong_mse[i]), copy_mse=float(copy[i])))
            if prior_error is not None:
                rows[-1]['prior_mse'] = float(prior_error[i])
    metric_names = ('mse','mean_mse','shuffled_mse','copy_mse')
    if 'baseline' in data:
        metric_names += ('prior_mse',)
    return dict(metrics={k:sum(r[k] for r in rows)/len(rows)
                         for k in metric_names}, rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=400)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--dim', type=int, default=128)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    parser.add_argument('--projection-basis', type=Path)
    parser.add_argument('--skip-train-evaluation', action='store_true',
                        help='Skip diagnostic final train-set inference; training and validation unchanged')
    parser.add_argument('--variants', nargs='+', choices=['direct','direct_unorm','residual','refinement','flow','endpoint_flow','endpoint_anchored','noisy_residual','wide_flow','wide_residual','wide_refinement','wide_terminal','prior_residual','prior_flow','prior_terminal','prior_refinement','prior_centered_flow','prior_centered_residual','prior_tangent_residual','prior_rotated_residual','prior_projected_residual','prior_projected_flow','prior_horizon_residual','prior_horizon_flow'],
                        default=['direct','direct_unorm','residual','refinement','flow'])
    args = parser.parse_args()
    if any(kind.startswith('prior_projected_') for kind in args.variants) and args.projection_basis is None:
        parser.error('Projected variants require a train-only projection basis')
    if min(args.steps, args.batch_size, args.dim, args.layers, args.threads) < 1 or args.dim % 4:
        parser.error('positive settings required; dim must be divisible by 4')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    manifest = json.loads((args.cache/'cache_manifest.json').read_text())
    for name in ('train','validation'):
        if file_hash(args.cache/f'{name}.pt') != manifest['cache_sha256'][name]:
            raise ValueError('Cache hash mismatch: '+name)
    train = torch.load(args.cache/'train.pt', weights_only=True)
    val = torch.load(args.cache/'validation.pt', weights_only=True)
    if any(kind.startswith('prior_') for kind in args.variants):
        if any('baseline' not in data for data in (train, val)):
            raise ValueError('Prior correction requires a verified current-only baseline cache')
    assert {r['episode'] for r in train['rows']}.isdisjoint({r['episode'] for r in val['rows']})
    _, _, tokens, features = train['current'].shape
    grid = math.isqrt(tokens)
    if grid*grid != tokens:
        raise ValueError('Square token grid required')
    report = dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
                  cache_manifest=manifest, results=[])
    if args.projection_basis is not None:
        projection = torch.load(args.projection_basis, weights_only=True)
        if projection['source_train_sha256'] != manifest['cache_sha256']['train']:
            raise ValueError('Projection basis was fitted on a different training cache')
        report['projection_basis_sha256'] = file_hash(args.projection_basis)
    for seed in args.seeds:
        for kind in args.variants:
            torch.manual_seed(seed)
            model = make_decoder(kind, args.dim, args.layers, features, train['codes'].shape[-1], grid)
            if isinstance(model, ProjectedPriorDecoder):
                model.set_basis(projection['basis'])
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
                options = ({'baseline': train['baseline'][idx]}
                           if getattr(model, 'requires_baseline', False) else {})
                if getattr(model, 'requires_physical_time', False):
                    options['physical_time'] = h.new_tensor([train['rows'][i]['dt'] for i in idx.tolist()])
                if kind in ('flow', 'endpoint_flow', 'endpoint_anchored', 'noisy_residual', 'wide_flow', 'prior_flow', 'prior_centered_flow', 'prior_projected_flow', 'prior_horizon_flow'):
                    loss = model.training_output(h, z, target, **options).velocity_loss
                else:
                    loss = F.mse_loss(model(h, z, **options), target)
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
            train_eval = None if args.skip_train_evaluation else evaluate(model, train, args.batch_size)
            result = dict(seed=seed, variant=kind, history=history, validation=validation,
                          train_metrics=None if train_eval is None else train_eval['metrics'], elapsed_seconds=time.perf_counter()-started,
                          parameters=sum(p.numel() for p in model.parameters()))
            report['results'].append(result)
            torch.save(dict(state_dict=model.state_dict(), variant=kind, dim=args.dim, layers=args.layers,
                            seed=seed, steps=args.steps), args.output/f'{kind}-seed{seed}.pt')
            (args.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
