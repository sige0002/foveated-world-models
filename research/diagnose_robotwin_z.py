"""Frozen-checkpoint localization of latent underuse on matched RoboTwin pairs.

Saves features, layer counterfactuals, posterior noise, and target-informed oracle
latent optimization. Oracle fitting is a diagnostic, never a held-out forecast.
"""
import argparse
from contextlib import nullcontext
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.train_robotwin_stage1 import install_episode_split, write_json


def geometry(t):
    x = t.double().flatten(1)
    c = x - x.mean(0)
    variation = float(c.square().sum() / len(x))
    common = float(x.mean(0).square().sum())
    gram = c @ c.T / len(x)
    squares = float(gram.square().sum())
    return dict(variance_sum=variation, common_mean_energy=common,
                variance_fraction=variation / (common + variation) if common + variation else 0.,
                mean_norm=float(x.norm(dim=1).mean()),
                effective_rank=variation**2 / squares if squares else 0.)


def main():
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--kind', choices=['local', 'official'], required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--samples', type=int, default=128)
    p.add_argument('--oracle-samples', type=int, default=16)
    p.add_argument('--oracle-steps', type=int, default=80)
    a = p.parse_args()
    assert not (a.output / 'results.json').exists()
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((a.run / 'resolved_config.yaml').read_text())
    req = yaml.safe_load((a.run / 'requested_config.yaml').read_text())
    contract = json.loads((a.run / 'data_contract.json').read_text())
    os.environ.update(cfg['environment'])
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    source = Path(req['official_source'])
    if a.kind == 'official':
        source = source.parent / 'LaWAM'
    sys.path[:0] = [str(source), *req['dependency_roots']]
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == contract['source_commit']
    patch = subprocess.check_output(['git', '-C', str(source), 'diff', '--binary'])
    assert not patch if a.kind == 'official' else hashlib.sha256(patch).hexdigest() == contract['source_patch_sha256']
    split = json.loads((a.run / 'episode_split.json').read_text())['ids']
    install_episode_split(split)
    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Subset
    from latent_action_model.core.lam_lightinng import VJEPA_LAM
    from latent_action_model.data_loader.lerobot_dataset import LeRobotLAMDataset
    from latent_action_model.data_loader.collate import lam_collate
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    started = time.monotonic()
    base = a.run.parents[1]
    release = base / 'models/lawam-lam-bd993da'
    checkpoint = (a.run / 'checkpoints/last.ckpt' if a.kind == 'local'
                  else release / 'checkpoints/pytorch_model.pt')
    mc = dict(cfg['model'] if a.kind == 'local' else yaml.safe_load((release / 'dino_large_vae.yaml').read_text())['model'])
    mc['vision_model_id'] = cfg['model']['vision_model_id']
    mc.pop('optimizer')
    model = VJEPA_LAM(**mc).eval().requires_grad_(False)
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
    if a.kind == 'local':
        assert saved['global_step'] == 10000 and saved['data_contract'] == contract
    state = saved['state_dict']
    expected = model.state_dict()
    for key in list(state):
        if key not in expected and key.startswith('lam.vision_encoder.model.layer.'):
            new = key.replace('lam.vision_encoder.model.layer.', 'lam.vision_encoder.model.model.layer.', 1)
            assert new in expected and new not in state
            state[new] = state.pop(key)
    model.load_state_dict(state, strict=True)
    del state, saved, expected
    model.cuda()
    lam = model.lam
    assert not lam.encoder.add_state
    lam.state_decoder = None
    amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)
    if a.cache.exists():
        cache = torch.load(a.cache, weights_only=True, map_location='cpu')
        assert cache['contract'] == contract and len(cache['rows']) == a.samples
    else:
        assert a.kind == 'local'
        dc = cfg['data']
        keys = ('data_root_dir', 'data_mix', 'num_frames', 'val_tail_ratio', 'video_backend',
                'image_hw', 'frame_dt_sec', 'human_frame_dt_sec')
        ds = LeRobotLAMDataset(**{k: dc[k] for k in keys if k in dc}, mode='val')
        for single in ds.mixture.datasets:
            for transform in single._lam_transforms_by_video_key.values():
                transform.eval()
        dl = DataLoader(Subset(ds, range(a.samples)), batch_size=4, num_workers=2,
                        collate_fn=partial(lam_collate, max_state_dim=32))
        cache = dict(x=[], y=[], rows=[], contract=contract)
        with torch.no_grad():
            for batch in dl:
                cache['rows'].extend([dict(episode=int(ep), base=int(b)) for ep, b in zip(batch['trajectory_ids'], batch['base_indices'])])
                batch = model.transfer_batch_to_device(batch, torch.device('cuda'), 0)
                with amp():
                    out = lam.get_latent_action(batch['videos'], None, batch['dec_videos'],
                                                embodiment_ids=batch['embodiment_ids'])
                cache['x'].append(out['dec_in'].cpu())
                cache['y'].append(out['tgt'].cpu())
        cache['x'] = torch.cat(cache['x'])
        cache['y'] = torch.cat(cache['y'])
        a.cache.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, a.cache)
        del dl, ds
    x, y = cache['x'], cache['y']
    episodes = np.array([r['episode'] for r in cache['rows']])
    rng = np.random.default_rng(1942)
    for _ in range(10000):
        donors = rng.permutation(a.samples)
        if (episodes[donors] != episodes).all():
            break
    else:
        raise RuntimeError('No different-episode permutation')

    def encode(future, fp32=False):
        values = {}
        def hook(name, last_query=False):
            def fn(module, inp, out):
                val = out[:, -1:] if last_query else out
                values.setdefault(name, []).append(val.detach().float().cpu())
            return fn
        hooks = [lam.encoder.QFormer.q_cross_attn.register_forward_hook(hook('initial_query'))]
        for i in [0, 1, 3, 7, 11, 15, 19, 23]:
            hooks.append(lam.encoder.QFormer.layers[i].register_forward_hook(hook(f'layer_{i}', True)))
        with torch.no_grad():
            for start in range(0, a.samples, 4):
                sl = slice(start, start + 4)
                xx = torch.cat([x[sl], future[sl]], 1).float().cuda()
                with (nullcontext() if fp32 else amp()):
                    nodes = lam.encoder(xx)
                pre = lam.vq.pre_norm(nodes.float().mean(1, keepdim=True))
                mu, lv = lam.vq._encode(nodes.float())
                for key, val in dict(nodes=nodes, pre_norm=pre, mu=mu, logvar=lv).items():
                    values.setdefault(key, []).append(val.detach().float().cpu())
        for h in hooks:
            h.remove()
        return {k: torch.cat(v) for k, v in values.items()}

    enc = {}
    for name, future, fp32 in [('real', y, False), ('current', x, False),
                               ('other_future', y[donors], False), ('fp32', y, True)]:
        enc[name] = encode(future, fp32)
        print(json.dumps({'phase': 'encoded', 'kind': a.kind, 'mode': name}), flush=True)
    mu, lv = enc['real']['mu'], enc['real']['logvar']
    m, l = mu.double().flatten(1), lv.double().flatten(1)
    var = l.exp()
    c = m - m.mean(0)
    cov = c.T @ c / len(m) + torch.diag(var.mean(0))
    sign, logdet = torch.linalg.slogdet(cov)
    assert sign > 0
    signal, noise = float(c.square().sum() / len(m)), float(var.sum(1).mean())
    posterior = dict(signal_variance_sum=signal, noise_variance_sum=noise,
                     signal_to_noise_trace_ratio=signal / noise,
                     mean_sigma=float(var.sqrt().mean()),
                     kl_to_prior=float(.5 * (m.square() + var - 1 - l).sum(1).mean()),
                     empirical_mixture_information_upper_bound_nats=float(.5 * (logdet - l.sum(1).mean())),
                     information_scope='Upper bound for one noisy sample from the finite empirical mixture; not deterministic means or population information')
    layer_report = {}
    for key in enc['real']:
        r = enc['real'][key]
        layer_report[key] = {'real': geometry(r)}
        for name in ['current', 'other_future', 'fp32']:
            delta = enc[name][key] - r
            layer_report[key][name] = dict(mean_difference_norm=float(delta.flatten(1).norm(dim=1).mean()),
                relative_rms_difference=float(delta.square().mean().sqrt() / r.square().mean().sqrt().clamp_min(1e-30)),
                geometry=geometry(enc[name][key]))

    def metrics(pred, target):
        return {'mse': (pred.float() - target.float()).square().flatten(1).mean(1),
                'smooth_l1': F.smooth_l1_loss(pred.float(), target.float(), beta=.1, reduction='none').flatten(1).mean(1)}
    # This constant is diagnostic and estimated on these validation codes, not a training-only baseline.
    mean = mu.mean(0, keepdim=True)
    gen = torch.Generator().manual_seed(492)
    eps = torch.randn(mu.shape, generator=gen)
    z_options = dict(real=mu, zero=torch.zeros_like(mu), validation_mean=mean.expand_as(mu),
                     other=mu[donors], current=enc['current']['mu'],
                     spread10=mean + 10 * (mu - mean), spread100=mean + 100 * (mu - mean),
                     sampled=mu + eps * (.5 * lv).exp(),
                     constant_posterior=mean + eps * (.5 * lv.mean(0, keepdim=True)).exp())
    decoded = {}
    real_pred = []
    with torch.no_grad():
        for name, z in z_options.items():
            rows = []
            for start in range(0, a.samples, 4):
                sl = slice(start, start + 4)
                with amp():
                    pred = lam.decoder(x[sl].cuda(), z[sl].cuda())
                rows.append({k: v.cpu() for k, v in metrics(pred, y[sl].cuda()).items()})
                if name == 'real':
                    real_pred.append(pred.float().cpu())
            decoded[name] = {k: float(torch.cat([r[k] for r in rows]).mean()) for k in rows[0]}
    # Compare replay to the independent, previously completed 512-pair evaluation.
    ref_dir = 'step10000-z-utility' if a.kind == 'local' else 'official-release-visual-matched-20261114'
    ref = json.loads((a.run / 'evaluations' / ref_dir / 'results.json').read_text())
    for own, old in zip(cache['rows'], ref['rows']):
        assert own['episode'] == old['episode'] and own['base'] == old['base']
    ref_mse = np.mean([r['metrics']['real']['visual_mse'] for r in ref['rows'][:a.samples]])
    assert abs(ref_mse - decoded['real']['mse']) < 1e-6

    oracle = []
    oracle_codes = []
    for start in range(0, a.oracle_samples, 4):
        sl = slice(start, start + 4)
        xx, yy = x[sl].cuda(), y[sl].cuda()
        z = mu[sl].cuda().clone().requires_grad_(True)
        opt = torch.optim.Adam([z], lr=.05)
        trace = []
        for step in range(a.oracle_steps + 1):
            opt.zero_grad(set_to_none=True)
            with amp():
                pred = lam.decoder(xx, z)
                loss = F.smooth_l1_loss(pred.float(), yy.float(), beta=.1)
            assert torch.isfinite(loss)
            if step in [0, 5, 20, a.oracle_steps]:
                trace.append({'step': step, 'smooth_l1': float(loss.detach()),
                              'mse': float((pred.float() - yy.float()).square().mean().detach()),
                              'z_mean_norm': float(z.detach().norm(dim=-1).mean())})
            if step == a.oracle_steps:
                break
            loss.backward()
            assert z.grad is not None and torch.isfinite(z.grad).all()
            opt.step()
        oracle_codes.append(z.detach().cpu())
        oracle.append(dict(start=start, trace=trace))
        print(json.dumps({'phase': 'oracle', 'kind': a.kind, **oracle[-1]}), flush=True)
    torch.save({'encoding': enc, 'oracle_codes': torch.cat(oracle_codes), 'donors': torch.tensor(donors)}, a.output / 'tensors.pt')
    report = dict(complete=True, kind=a.kind, checkpoint=str(checkpoint), source=str(source),
                  source_commit=contract['source_commit'], source_patch_sha256=hashlib.sha256(patch).hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  samples=a.samples, rows=cache['rows'], posterior=posterior, layers=layer_report,
                  decoded=decoded, oracle=oracle, elapsed_seconds=time.monotonic() - started,
                  peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                  scope='Frozen weights. Oracle sees each evaluated future target and optimizes only its code: capacity diagnostic, not prediction performance or a trained solution.')
    write_json(a.output / 'results.json', report)
    print(json.dumps({'complete': a.kind, 'posterior': posterior, 'decoded': decoded}), flush=True)


if __name__ == '__main__':
    main()
