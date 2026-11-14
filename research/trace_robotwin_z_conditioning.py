"""Trace the actual latent input, AdaLN conditioning and output interventions.

Frozen local/released checkpoints. Native four-image replay plus 128 cached pairs
at BF16 and highest FP32. Constants are diagnostic validation means, not fitted
forecast baselines. All input changes hold the current features fixed.
"""
import argparse
from contextlib import nullcontext
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.diagnose_robotwin_z import geometry
from research.train_robotwin_stage1 import install_episode_split, write_json


def main():
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    cfg = yaml.safe_load((a.run / 'resolved_config.yaml').read_text())
    req = yaml.safe_load((a.run / 'requested_config.yaml').read_text())
    os.environ.update(cfg['environment'])
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    sys.path[:0] = [req['official_source'], *req['dependency_roots']]
    split = json.loads((a.run / 'episode_split.json').read_text())['ids']
    install_episode_split(split)
    import torch
    from torch.utils.data import DataLoader, Subset
    from latent_action_model.core.lam_lightinng import VJEPA_LAM
    from latent_action_model.data_loader.lerobot_dataset import LeRobotLAMDataset
    from latent_action_model.data_loader.collate import lam_collate
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    diag = a.run / 'evaluations/z-mechanism-20261114'
    cache = torch.load(diag / 'features.pt', weights_only=True)
    dc = cfg['data']
    keys = ('data_root_dir', 'data_mix', 'num_frames', 'val_tail_ratio', 'video_backend',
            'image_hw', 'frame_dt_sec', 'human_frame_dt_sec')
    ds = LeRobotLAMDataset(**{k: dc[k] for k in keys if k in dc}, mode='val')
    for single in ds.mixture.datasets:
        for transform in single._lam_transforms_by_video_key.values():
            transform.eval()
    batch_cpu = next(iter(DataLoader(Subset(ds, range(4)), batch_size=4, num_workers=0,
                      collate_fn=partial(lam_collate, max_state_dim=32))))
    assert [int(ep) for ep in batch_cpu['trajectory_ids']] == [r['episode'] for r in cache['rows'][:4]]
    reports = {}
    for kind in ['local', 'official']:
        old = json.loads((diag / kind / 'results.json').read_text())
        old_tensors = torch.load(diag / kind / 'tensors.pt', weights_only=True)
        high = torch.load(diag / f'{kind}-highest-precision.pt', weights_only=True)
        mc = dict(cfg['model']) if kind == 'local' else dict(yaml.safe_load((Path(old['checkpoint']).parents[1] / 'dino_large_vae.yaml').read_text())['model'])
        mc['vision_model_id'] = cfg['model']['vision_model_id']
        mc.pop('optimizer')
        model = VJEPA_LAM(**mc).eval().requires_grad_(False)
        saved = torch.load(old['checkpoint'], weights_only=True, map_location='cpu', mmap=True)['state_dict']
        expected = model.state_dict()
        for key in list(saved):
            if key not in expected and key.startswith('lam.vision_encoder.model.layer.'):
                new = key.replace('lam.vision_encoder.model.layer.', 'lam.vision_encoder.model.model.layer.', 1)
                assert new in expected
                saved[new] = saved.pop(key)
        model.load_state_dict(saved, strict=True)
        del saved, expected
        model.cuda()
        lam = model.lam
        lam.state_decoder = None
        captures = {}
        hooks = []

        def input_hook(module, args, kwargs):
            actions = kwargs['actions'] if 'actions' in kwargs else args[1]
            captures['z_received'] = actions.detach().clone()
        hooks.append(lam.decoder.register_forward_pre_hook(input_hook, with_kwargs=True))

        def hook(name):
            def fn(module, args, out):
                captures[name] = out.detach()
            return fn
        hooks.append(lam.decoder.action_in_proj.register_forward_hook(hook('projected_condition')))
        for i, block in enumerate(lam.decoder.dec_layers):
            hooks.append(block.adaLN_modulation.register_forward_hook(hook(f'modulation_{i}')))
            hooks.append(block.register_forward_hook(hook(f'block_{i}')))
        hooks.append(lam.decoder.last_ln.register_forward_hook(hook('last_norm')))
        batch = model.transfer_batch_to_device(batch_cpu, torch.device('cuda'), 0)
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            out = lam.get_latent_action(batch['videos'], None, batch['dec_videos'],
                embodiment_ids=batch['embodiment_ids'], predict_future_frame=True)
        received = captures['z_received']
        assert torch.equal(received, out['quantized'])
        torch.testing.assert_close(out['dec_in'].cpu(), cache['x'][:4], rtol=0, atol=1e-6)
        torch.testing.assert_close(out['tgt'].cpu(), cache['y'][:4], rtol=0, atol=1e-6)
        full_check = dict(shape=list(received.shape), decoder_input_equals_encoder_posterior_exactly=True,
            difference_max_abs=float((received - out['quantized']).abs().max()),
            z_norm=float(received.norm(dim=-1).mean()),
            prediction_mse=float((out['recon'].float() - out['tgt'].float()).square().mean()))
        report = {'full_path_four_real_pairs': full_check}
        del out, batch
        donors = old_tensors['donors']
        for precision in ['native_bf16', 'highest_fp32']:
            torch.set_float32_matmul_precision('medium' if precision == 'native_bf16' else 'highest')
            mu = old_tensors['encoding']['real']['mu'] if precision == 'native_bf16' else high['real']['mu64'].float()
            means = mu.mean(0, keepdim=True).expand_as(mu)
            options = {'real': mu, 'mean': means, 'zero': torch.zeros_like(mu), 'other': mu[donors]}
            all_rows = {mode: [] for mode in options}
            deltas = {mode: {} for mode in ['mean', 'zero', 'other']}
            real_values = {'z_received': [], 'projected_condition': []}
            real_values.update({f'modulation_{i}': [] for i in range(12)})
            true_change = []
            with torch.no_grad():
                for start in range(0, len(mu), 4):
                    sl = slice(start, start + 4)
                    x, y = cache['x'][sl].float().cuda(), cache['y'][sl].float().cuda()
                    true_change.append((y - x).square().flatten(1).mean(1).cpu())
                    outputs = {}
                    for mode, z in options.items():
                        with (torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'native_bf16' else nullcontext()):
                            pred = lam.decoder(x, z[sl].cuda())
                        assert torch.equal(captures['z_received'].cpu(), z[sl])
                        outputs[mode] = {k: v.float() for k, v in captures.items()}
                        outputs[mode]['prediction'] = pred.float()
                        all_rows[mode].append((pred.float() - y).square().flatten(1).mean(1).cpu())
                    for key in real_values:
                        real_values[key].append(outputs['real'][key].cpu())
                    for mode in deltas:
                        for key in outputs['real']:
                            real, other = outputs['real'][key], outputs[mode][key]
                            val = torch.stack([(other - real).square().flatten(1).mean(1),
                                               real.square().flatten(1).mean(1),
                                               (other - real).abs().flatten(1).max(1).values], dim=1).cpu()
                            deltas[mode].setdefault(key, []).append(val)
                    del outputs
            true_rms = float(torch.cat(true_change).mean().sqrt())
            stats = {}
            for mode, entries in deltas.items():
                stats[mode] = {}
                for key, values in entries.items():
                    v = torch.cat(values)
                    delta_rms, real_rms = v[:, 0].mean().sqrt(), v[:, 1].mean().sqrt()
                    stats[mode][key] = dict(difference_rms=float(delta_rms), real_rms=float(real_rms),
                        difference_over_real_rms=float(delta_rms / real_rms.clamp_min(1e-30)),
                        max_abs_difference=float(v[:, 2].max()))
                stats[mode]['prediction']['difference_over_true_change_rms'] = stats[mode]['prediction']['difference_rms'] / true_rms
            geom = {k: geometry(torch.cat(v)) for k, v in real_values.items()}
            metrics = {mode: float(torch.cat(values).mean()) for mode, values in all_rows.items()}
            if precision == 'native_bf16':
                for mode, old_name in [('real', 'real'), ('zero', 'zero'), ('other', 'other')]:
                    assert abs(metrics[mode] - old['decoded'][old_name]['mse']) < 1e-6
            report[precision] = dict(samples=len(mu), geometry=geom, interventions=stats,
                                     future_mse=metrics, true_change_rms=true_rms)
            print(json.dumps({'kind': kind, 'precision': precision, 'metrics': metrics,
                'mean_z_output_change_fraction': stats['mean']['prediction']['difference_over_true_change_rms']}), flush=True)
        for h in hooks:
            h.remove()
        reports[kind] = report
        del model, lam, captures
        torch.cuda.empty_cache()
    write_json(a.output, dict(complete=True, reports=reports,
        scope='Frozen checkpoints; direct z interventions at fixed current inputs. Modulation/variance ratios are coordinate-dependent geometry. No training-cause claim.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
