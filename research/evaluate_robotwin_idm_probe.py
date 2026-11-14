"""Frozen IDM probes of measured state changes with episode-held-out evaluation.

Fit small ridge readouts on CPU (at most 48 inputs); encode real images on GPU.
Choose regularization using only grouped cross-validation of training episodes.
Original model weights never change; readout parameters are saved under the run.
"""
import argparse
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


def main():
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-samples', type=int, default=2048)
    p.add_argument('--val-samples', type=int, default=512)
    a = p.parse_args()
    assert a.output.resolve().is_relative_to(a.run.resolve())
    assert not (a.output / 'results.json').exists()
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load((a.run / 'resolved_config.yaml').read_text())
    req = yaml.safe_load((a.run / 'requested_config.yaml').read_text())
    contract = json.loads((a.run / 'data_contract.json').read_text())
    os.environ.update(cfg['environment'])
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    source = Path(req['official_source'])
    sys.path[:0] = [str(source), *req['dependency_roots']]
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == contract['source_commit']
    assert hashlib.sha256(subprocess.check_output(['git', '-C', str(source), 'diff', '--binary'])).hexdigest() == contract['source_patch_sha256']
    split = json.loads((a.run / 'episode_split.json').read_text())['ids']
    install_episode_split(split)
    import numpy as np
    import torch
    from torch.utils.data import DataLoader, Subset
    from latent_action_model.core.lam_lightinng import VJEPA_LAM
    from latent_action_model.data_loader.lerobot_dataset import LeRobotLAMDataset
    from latent_action_model.data_loader.collate import lam_collate
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    started = time.monotonic()
    cache_path = a.output / 'codes.pt'
    if cache_path.exists():
        all_data = torch.load(cache_path, weights_only=True)
    else:
        mc = dict(cfg['model'])
        mc.pop('optimizer')
        model = VJEPA_LAM(**mc).eval().requires_grad_(False)
        saved = torch.load(a.run / 'checkpoints/last.ckpt', weights_only=True, map_location='cpu', mmap=True)
        assert saved['global_step'] == 10000 and saved['data_contract'] == contract
        model.load_state_dict(saved['state_dict'], strict=True)
        del saved
        model.cuda()
        lam = model.lam
        assert not lam.encoder.add_state
        dc = cfg['data']
        keys = ('data_root_dir', 'data_mix', 'num_frames', 'val_tail_ratio', 'video_backend',
                'image_hw', 'frame_dt_sec', 'human_frame_dt_sec')
        all_data = {}
        with torch.no_grad():
            for mode, count in [('train', a.train_samples), ('val', a.val_samples)]:
                ds = LeRobotLAMDataset(**{k: dc[k] for k in keys if k in dc}, mode=mode)
                for single in ds.mixture.datasets:
                    for transform in single._lam_transforms_by_video_key.values():
                        transform.eval()
                dl = DataLoader(Subset(ds, range(count)), batch_size=4, num_workers=2,
                                collate_fn=partial(lam_collate, max_state_dim=32))
                rows = []
                cache = {k: [] for k in ['state', 'delta', 'native_z', 'native_current_z',
                                        'highest_z', 'highest_current_z', 'g_prediction']}
                for batch in dl:
                    rows.extend([dict(episode=int(ep), base=int(b)) for ep, b in zip(batch['trajectory_ids'], batch['base_indices'])])
                    assert all(r['episode'] in split[mode] for r in rows[-4:])
                    batch = model.transfer_batch_to_device(batch, torch.device('cuda'), 0)
                    assert torch.equal(batch['delta_proprio'], batch['states'][:, 1] - batch['states'][:, 0])
                    torch.set_float32_matmul_precision('medium')
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        out = lam.get_latent_action(batch['videos'], batch['states'], batch['dec_videos'],
                            state_mask=batch['state_mask'], embodiment_ids=batch['embodiment_ids'])
                        current_nodes = lam.encoder(out['dec_in'].expand(-1, 2, -1, -1))
                        g = lam.state_decoder(out['quantized'], batch['states'][:, :1], batch['embodiment_ids'])
                    native_current, _ = lam.vq._encode(current_nodes.float())
                    torch.set_float32_matmul_precision('highest')
                    high_nodes = lam.encoder(torch.cat([out['dec_in'], out['tgt']], 1).float())
                    high_current_nodes = lam.encoder(out['dec_in'].expand(-1, 2, -1, -1).float())
                    high, _ = lam.vq._encode(high_nodes)
                    high_current, _ = lam.vq._encode(high_current_nodes)
                    vals = dict(state=batch['states'][:, 0, :16], delta=batch['delta_proprio'][:, :16],
                                native_z=out['quantized'].flatten(1), native_current_z=native_current.flatten(1),
                                highest_z=high.flatten(1), highest_current_z=high_current.flatten(1),
                                g_prediction=g[:, :16])
                    for key, val in vals.items():
                        assert torch.isfinite(val).all()
                        cache[key].append(val.detach().float().cpu())
                    if len(rows) % 256 == 0:
                        print(json.dumps({'phase': 'encoding', 'split': mode, 'samples': len(rows)}), flush=True)
                cache = {k: torch.cat(v) for k, v in cache.items()}
                cache['rows'] = rows
                all_data[mode] = cache
                del ds, dl
        torch.save(all_data, cache_path)
        del model, lam
        torch.cuda.empty_cache()
    tr, va = all_data['train'], all_data['val']
    assert len(tr['rows']) == a.train_samples and len(va['rows']) == a.val_samples
    train_eps = np.array([r['episode'] for r in tr['rows']])
    val_eps = np.array([r['episode'] for r in va['rows']])
    assert not (set(train_eps) & set(val_eps))
    reference = json.loads((a.run / 'evaluations/step10000-z-utility/results.json').read_text())
    for r, prev in zip(va['rows'], reference['rows']):
        assert r['episode'] == prev['episode'] and r['base'] == prev['base']
    yy = tr['delta'].numpy().astype(np.float64)
    vy = va['delta'].numpy().astype(np.float64)
    state_tr = tr['state'].numpy().astype(np.float64)
    state_va = va['state'].numpy().astype(np.float64)
    rng = np.random.default_rng(42)
    unique = rng.permutation(np.unique(train_eps))
    fold_map = {ep: i % 5 for i, ep in enumerate(unique)}
    folds = np.array([fold_map[ep] for ep in train_eps])
    alphas = np.logspace(-6, 2, 17)
    stats = json.loads((Path(contract['data_root']) / 'meta/stats_gr00t.json').read_text())['observation.state']
    xyz = [0, 1, 2, 8, 9, 10]
    quat = [3, 4, 5, 6, 11, 12, 13, 14]
    grip = [7, 15]
    xyz_scale = np.array([(stats['max'][i] - stats['min'][i]) / 2 for i in xyz])

    def metrics(pred):
        error = (pred - vy)**2
        return dict(state_mse=float(error.mean()), xyz_mse=float(error[:, xyz].mean()),
                    quaternion_mse=float(error[:, quat].mean()), gripper_mse=float(error[:, grip].mean()),
                    position_error_cm=float(np.linalg.norm(((pred - vy)[:, xyz] * xyz_scale).reshape(-1, 2, 3), axis=-1).mean() * 100))

    def fit(x, y, alpha):
        xm, xs, ym = x.mean(0), x.std(0), y.mean(0)
        xs = np.maximum(xs, 1e-12)
        xx = (x - xm) / xs
        w = np.linalg.solve(xx.T @ xx / len(x) + alpha * np.eye(x.shape[1]), xx.T @ (y - ym) / len(x))
        return dict(xm=xm, xs=xs, ym=ym, w=w)

    def predict(model, x):
        return (x - model['xm']) / model['xs'] @ model['w'] + model['ym']

    models, predictions, fits = {}, {}, {}
    for precision in ['native', 'highest']:
        zz, vz = tr[precision + '_z'].numpy().astype(np.float64), va[precision + '_z'].numpy().astype(np.float64)
        cc, vc = tr[precision + '_current_z'].numpy().astype(np.float64), va[precision + '_current_z'].numpy().astype(np.float64)
        designs = {'z': (zz, vz), 'current_only_z': (cc, vc),
                   'state_z': (np.c_[state_tr, zz], np.c_[state_va, vz]),
                   'state_current_only_z': (np.c_[state_tr, cc], np.c_[state_va, vc])}
        if precision == 'native':
            designs['state'] = (state_tr, state_va)
        for name, (x, vx) in designs.items():
            label = name if name == 'state' else precision + '_' + name
            risks = []
            for alpha in alphas:
                cv = np.empty_like(yy)
                for fold in range(5):
                    test = folds == fold
                    fitted = fit(x[~test], yy[~test], alpha)
                    cv[test] = predict(fitted, x[test])
                risks.append(float(((cv - yy)**2).mean()))
            best = int(np.argmin(risks))
            fitted = fit(x, yy, alphas[best])
            pred = predict(fitted, vx)
            models[label] = {'alpha': float(alphas[best]), 'cv_risk': risks[best], 'cv_curve': risks, 'metrics': metrics(pred)}
            predictions[label] = pred
            fits[label] = fitted
            print(json.dumps({'phase': 'probe_fit', 'name': label, **models[label]}), flush=True)
    predictions['constant'] = np.broadcast_to(yy.mean(0), vy.shape)
    predictions['g_existing'] = va['g_prediction'].numpy().astype(np.float64)
    expected_g = np.mean([r['metrics']['real']['state_mse'] for r in reference['rows'][:a.val_samples]])
    assert abs(metrics(predictions['g_existing'])['state_mse'] - expected_g) < 1e-6
    for name in ['constant', 'g_existing']:
        models[name] = {'metrics': metrics(predictions[name])}
    rng = np.random.default_rng(1942)
    for _ in range(10000):
        donors = rng.permutation(len(val_eps))
        if (val_eps[donors] != val_eps).all():
            break
    else:
        raise RuntimeError('Unable to construct different-episode donors')
    for precision in ['native', 'highest']:
        vz = va[precision + '_z'].numpy().astype(np.float64)
        for name in ['z', 'state_z']:
            label = precision + '_' + name
            design = vz[donors] if name == 'z' else np.c_[state_va, vz[donors]]
            pred = predict(fits[label], design)
            predictions[label + '_shuffled'] = pred
            models[label + '_shuffled'] = {'metrics': metrics(pred)}

    def contrast(treatment, control):
        # Difference > 0 means the treatment has lower error.
        delta = ((predictions[control] - vy)**2).mean(1) - ((predictions[treatment] - vy)**2).mean(1)
        eps = np.unique(val_eps)
        sums = np.array([delta[val_eps == ep].sum() for ep in eps])
        counts = np.array([(val_eps == ep).sum() for ep in eps])
        rng = np.random.default_rng(142)
        draws = rng.integers(len(eps), size=(4000, len(eps)))
        boot = sums[draws].sum(1) / counts[draws].sum(1)
        return dict(treatment=treatment, control=control, control_minus_treatment=float(delta.mean()),
                    ci95=np.quantile(boot, [.025, .975]).tolist(),
                    relative_improvement_percent=float(100 * delta.mean() / models[control]['metrics']['state_mse']))
    contrasts = []
    for precision in ['native', 'highest']:
        contrasts.extend([contrast(precision + '_z', 'constant'),
                          contrast(precision + '_z', precision + '_current_only_z'),
                          contrast(precision + '_state_z', 'state'),
                          contrast(precision + '_state_z', precision + '_state_current_only_z'),
                          contrast(precision + '_state_z', precision + '_state_z_shuffled')])
    tensor_fits = {k: {n: torch.from_numpy(v.copy()) for n, v in m.items()} for k, m in fits.items()}
    torch.save({'fits': tensor_fits, 'predictions': {k: torch.from_numpy(v.copy()) for k, v in predictions.items()},
                'donors': torch.from_numpy(donors)}, a.output / 'readouts.pt')
    report = dict(complete=True, step=10000, train_samples=len(yy), val_samples=len(vy),
        train_episodes=len(set(train_eps)), val_episodes=len(set(val_eps)),
        checkpoint=str(a.run / 'checkpoints/last.ckpt'), source_commit=contract['source_commit'],
        source_patch_sha256=contract['source_patch_sha256'], alphas=alphas.tolist(),
        selection='Ridge alpha chosen using five episode-disjoint folds within training only. Input standardization refit in each fold.',
        precision='native: BF16 mixed with upstream medium setting; highest: cached image features with FP32 highest encoder/head',
        scope='Frozen IDM, linear readout of normalized state delta. Detects linearly accessible information, not complete information or policy success. Evaluation episodes held out for the original local run.',
        models=models, contrasts=contrasts, elapsed_seconds=time.monotonic() - started,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(a.output / 'results.json', report)
    print(json.dumps({'complete': True, 'contrasts': contrasts}), flush=True)


if __name__ == '__main__':
    main()
