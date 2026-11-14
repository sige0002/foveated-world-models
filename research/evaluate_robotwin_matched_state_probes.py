"""Matched frozen posterior-mean readouts: official, old10k and two3000 arms.

All models share highest-FP32 DINO features and the same normalized16D delta
labels. Only external ridge heads are fitted; no backbone/g/optimizer updates.
"""
import argparse
from copy import deepcopy
from functools import partial
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import evaluate_robotwin_loss_ablation_fp32_v2 as api
from research.train_robotwin_stage1 import install_episode_split, write_json

RUNS = Path('/media/sadasue/SSD1/foveated-world-models/runs')
OLD = RUNS / 'robotwin-qfix-delta2-b4-20261114'
GROUP = RUNS / 'robotwin-loss-ablation-b4-20261114'
EVAL = GROUP / 'evaluations/final3000-highest-fp32-v2'
OLD_CODES = OLD / 'evaluations/idm-state-probe-20261114/codes.pt'
XYZ, QUAT, GRIP = [0, 1, 2, 8, 9, 10], [3, 4, 5, 6, 11, 12, 13, 14], [7, 15]


def identities(rows):
    return [{'episode': int(r['episode']), 'base': int(r['base'])} for r in rows]


def fit_ridge(x, y, alpha):
    import numpy as np
    xm, xs, ym = x.mean(0), np.maximum(x.std(0), 1e-12), y.mean(0)
    centered = (x - xm) / xs
    weights = np.linalg.solve(centered.T @ centered / len(x) + alpha * np.eye(x.shape[1]),
                              centered.T @ (y - ym) / len(x))
    return {'xm': xm, 'xs': xs, 'ym': ym, 'w': weights}


def predict(fit, x):
    return (x - fit['xm']) / fit['xs'] @ fit['w'] + fit['ym']


def select_ridge(x, y, folds, alphas):
    import numpy as np
    risks = []
    for alpha in alphas:
        heldout = np.empty_like(y)
        for fold in range(5):
            test = folds == fold
            assert test.any() and (~test).any()
            heldout[test] = predict(fit_ridge(x[~test], y[~test], alpha), x[test])
        risks.append(float(((heldout - y)**2).mean()))
    selected = int(np.argmin(risks))
    return fit_ridge(x, y, alphas[selected]), {'alpha': float(alphas[selected]), 'cv_risk': risks[selected], 'cv_curve': risks}


def metric_rows(prediction, target, xyz_scale):
    import numpy as np
    error = (prediction - target)**2
    return {'state_mse': error.mean(1), 'xyz_mse': error[:, XYZ].mean(1),
            'quaternion_mse': error[:, QUAT].mean(1), 'gripper_mse': error[:, GRIP].mean(1),
            'position_error_cm': np.linalg.norm(((prediction - target)[:, XYZ] * xyz_scale).reshape(-1, 2, 3), axis=-1).mean(1) * 100}


def build_train_cache(model, cfg, contract, reference, path, vision_hash, batch_size):
    import torch
    from torch.utils.data import DataLoader, Subset
    from latent_action_model.data_loader.lerobot_dataset import LeRobotLAMDataset
    from latent_action_model.data_loader.collate import lam_collate
    from latent_action_model.data_loader.lerobot_datamodule import _lam_worker_init_fn
    from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug
    split = api.read_json(OLD / 'episode_split.json')['ids']
    install_episode_split(split)
    keys = ('data_root_dir', 'data_mix', 'num_frames', 'val_tail_ratio', 'video_backend',
            'image_hw', 'frame_dt_sec', 'human_frame_dt_sec')
    ds = LeRobotLAMDataset(**{k: cfg['data'][k] for k in keys if k in cfg['data']}, mode='train')
    assert len(ds.mixture.datasets) == 1
    assert set(ds.mixture.datasets[0].trajectory_ids.tolist()) == set(split['train'])
    for single in ds.mixture.datasets:
        for transform in single._lam_transforms_by_video_key.values():
            transform.eval()
    loader = DataLoader(Subset(ds, range(2048)), batch_size=batch_size, num_workers=2,
        collate_fn=partial(lam_collate, max_state_dim=32), worker_init_fn=_lam_worker_init_fn,
        pin_memory=True, prefetch_factor=2, in_order=True,
        generator=torch.Generator().manual_seed(42))
    chunks = {k: [] for k in ('x', 'y', 'state', 'delta')}
    rows, pixels = [], []
    parity_videos = None
    for batch in loader:
        count = len(batch['videos']); start = len(rows)
        current_rows = [{'episode': int(ep), 'base': int(base)} for ep, base in zip(batch['trajectory_ids'], batch['base_indices'])]
        assert current_rows == reference['rows'][start:start + count]
        assert all(r['episode'] in split['train'] for r in current_rows)
        assert torch.equal(batch['delta_proprio'], batch['states'][:, 1] - batch['states'][:, 0])
        state, delta = batch['states'][:, 0, :16], batch['delta_proprio'][:, :16]
        assert torch.equal(state, reference['state'][start:start + count])
        assert torch.equal(delta, reference['delta'][start:start + count])
        mask = batch['state_mask'].any(1)
        assert mask[:, :16].all() and not mask[:, 16:].any()
        videos, dec_videos = gpu_two_view_video_aug(batch['videos'].cuda(),
            output_size=tuple(cfg['model']['image_hw']), training=False,
            dual_view_aug=cfg['model']['dual_view_aug'])
        assert torch.equal(videos, dec_videos)
        features = model.vision_encoder.encode(torch.cat([videos, dec_videos], dim=1), n=-2)
        assert features.dtype == torch.float32 and features.shape[1] == 4
        x, y = features[:, 2:3], features[:, -1:]
        if parity_videos is None:
            parity_videos = videos.cpu().clone()
        pixels.extend(api.tensor_hash(v) for v in videos)
        rows.extend(current_rows)
        for key, value in {'x': x, 'y': y, 'state': state, 'delta': delta}.items():
            assert torch.isfinite(value).all()
            chunks[key].append(value.cpu())
        if len(rows) % 128 == 0:
            print(json.dumps({'phase': 'train_FP32_DINO_cache', 'samples': len(rows)}), flush=True)
    cache = {k: torch.cat(values) for k, values in chunks.items()}
    cache.update(rows=rows, parity_videos=parity_videos, contract=contract,
        old_codes_sha256=api.digest(OLD_CODES), identity_sha256=api.json_hash(rows),
        vision_encoder_fingerprint=vision_hash, pixel_sha256_per_pair=pixels,
        precision='highest FP32; TF32 disabled', batch_size=batch_size, frame_offset=48,
        feature_layer=-2, preprocessing='dataset mode=train with transforms.eval; upstream gpu_two_view_video_aug(training=False)')
    cache['tensor_sha256'] = {k: api.tensor_hash(v) for k, v in cache.items() if isinstance(v, torch.Tensor)}
    temporary = path.with_suffix('.pt.tmp')
    torch.save(cache, temporary); temporary.replace(path)
    return cache


def validate_train_cache(cache, reference, contract, vision_hash):
    import torch
    assert cache['rows'] == reference['rows'] and len(cache['rows']) == 2048
    assert cache['identity_sha256'] == api.json_hash(reference['rows'])
    assert cache['old_codes_sha256'] == api.digest(OLD_CODES)
    assert cache['contract'] == contract and cache['vision_encoder_fingerprint'] == vision_hash
    assert cache['precision'] == 'highest FP32; TF32 disabled'
    assert torch.equal(cache['state'], reference['state']) and torch.equal(cache['delta'], reference['delta'])
    for key, value in cache['tensor_sha256'].items():
        assert api.tensor_hash(cache[key]) == value


def posterior_parity(model, cache):
    """Compare only native IDM/VAE: neither released nor local g is invoked."""
    import torch
    videos = cache['parity_videos'].cuda(); n = len(videos)
    native = model.get_latent_action(videos, None, videos,
        embodiment_ids=torch.ones(n, device='cuda', dtype=torch.long), predict_future_frame=False)
    inputs, _, _ = api.native_feature_views(cache['x'][:n].cuda(), cache['y'][:n].cuda())
    mu, _ = model.vq._encode(model.encoder(inputs))
    torch.testing.assert_close(mu, native['quantized'], rtol=0, atol=1e-6)
    return {'samples': n, 'atol': 1e-6, 'maximum_absolute_difference': float((mu - native['quantized']).abs().max())}


def encode(model, cache, batch_size):
    import torch
    zs, currents = [], []
    for start in range(0, len(cache['x']), batch_size):
        x, y = cache['x'][start:start + batch_size].cuda(), cache['y'][start:start + batch_size].cuda()
        real, _, _ = api.native_feature_views(x, y)
        current, _, _ = api.native_feature_views(x, x)
        mu, _ = model.vq._encode(model.encoder(real))
        current_mu, _ = model.vq._encode(model.encoder(current))
        assert torch.isfinite(mu).all() and torch.isfinite(current_mu).all()
        zs.append(mu.flatten(1).cpu()); currents.append(current_mu.flatten(1).cpu())
    return {'z': torch.cat(zs), 'current_only_z': torch.cat(currents)}


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, choices=(4, 8, 16), default=16)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args(); output = args.output.resolve()
    assert output.is_relative_to(RUNS.resolve())
    assert not (output / 'results.json').exists()
    output.mkdir(parents=True, exist_ok=True)
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(output), '-o', 'TARGET,SOURCE,FSTYPE'], text=True))
    assert mount['filesystems'][0]['target'] == '/media/sadasue/SSD1'
    final = api.read_json(EVAL / 'results.json'); assert final['complete']
    assert final['protocol']['script_sha256'] == api.digest(Path(api.__file__))
    cfg = yaml.safe_load((GROUP / 'baseline/resolved_config.yaml').read_text())
    req = yaml.safe_load((GROUP / 'baseline/requested_config.yaml').read_text())
    contract = api.read_json(GROUP / 'baseline/data_contract.json')
    old_contract = api.read_json(OLD / 'data_contract.json'); assert contract == old_contract
    assert api.read_json(OLD / 'episode_split.json') == api.read_json(GROUP / 'baseline/episode_split.json')
    os.environ.update(cfg['environment'])
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    sys.path[:0] = [req['official_source'], *req['dependency_roots']]
    assert subprocess.check_output(['git', '-C', req['official_source'], 'rev-parse', 'HEAD'], text=True).strip() == contract['source_commit']
    assert api.digest(ROOT / 'research/train_robotwin_stage1.py') == '9ecbea0151c1f902647dc9e41a5a17a431ab5c4319da658b9d0cdeb5104cba27'
    patch = subprocess.check_output(['git', '-C', req['official_source'], 'diff', '--binary'])
    import hashlib
    assert hashlib.sha256(patch).hexdigest() == contract['source_patch_sha256']
    import numpy as np
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1); torch.manual_seed(42)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    old = torch.load(OLD_CODES, weights_only=True, map_location='cpu')
    val_path = Path(final['protocol']['cache'])
    assert api.digest(val_path) == api.read_json(EVAL / 'feature-cache-manifest.json')['sha256']
    val = torch.load(val_path, weights_only=True, map_location='cpu')
    refs = api.read_json(GROUP / 'baseline/z_validation_step000500.json')['rows']
    reference_rows = [{k: r[k] for k in ('episode', 'base', 'swap_valid', 'donor_episode', 'donor_base')} for r in refs]
    vision_hash = final['models']['baseline']['module_hashes_before']['vision_encoder']
    api.validate_cache(val, reference_rows, contract, vision_hash)
    assert identities(val['rows']) == old['val']['rows']
    assert torch.equal(val['states'][:, 0, :16], old['val']['state'])
    assert torch.equal(val['delta'][:, :16], old['val']['delta'])
    train_eps = np.array([r['episode'] for r in old['train']['rows']])
    val_eps = np.array([r['episode'] for r in old['val']['rows']])
    assert len(train_eps) == 2048 and len(val_eps) == 512 and not set(train_eps) & set(val_eps)
    split = api.read_json(OLD / 'episode_split.json')['ids']
    assert set(train_eps) <= set(split['train']) and set(val_eps) <= set(split['val'])
    ordered = np.random.default_rng(42).permutation(np.unique(train_eps))
    mapping = {int(ep): i % 5 for i, ep in enumerate(ordered)}
    folds = np.array([mapping[int(ep)] for ep in train_eps])
    donors = np.array([r['donor_index'] for r in val['rows']]); assert (val_eps[donors] != val_eps).all()
    release = Path(final['protocol']['release'])
    models = {'baseline3000': (GROUP / 'baseline', 3000), 'official': (release, None),
              'old10000': (OLD, 10000), 'combined3000': (GROUP / 'kl_mse', 3000)}
    protocol = {'stage': 'prepared_no_GPU', 'script_sha256': api.digest(__file__),
        'evaluator_helper_sha256': api.digest(api.__file__), 'final_FP32_report_sha256': api.digest(EVAL / 'results.json'),
        'old_codes_path': str(OLD_CODES), 'old_codes_sha256': api.digest(OLD_CODES),
        'train_identity_sha256': api.json_hash(old['train']['rows']), 'val_identity_sha256': api.json_hash(old['val']['rows']),
        'split_sha256': api.digest(OLD / 'episode_split.json'), 'source_commit': contract['source_commit'],
        'source_patch_sha256': contract['source_patch_sha256'], 'val_cache': str(val_path),
        'val_cache_sha256': api.digest(val_path), 'vision_encoder_fingerprint': vision_hash,
        'train_samples': 2048, 'val_samples': 512, 'train_episodes': len(set(train_eps)), 'val_episodes': len(set(val_eps)),
        'batch_size': args.batch_size, 'models': {k: {'run': str(v[0]), 'step': v[1]} for k, v in models.items()},
        'input': 'Deterministic posterior mean mu only; no sampled epsilon. HighestFP32 DINO/encoder/VAE, TF32 disabled.',
        'labels': 'Common16D normalized future-minus-current state delta, including both grippers; official g14 is never called.',
        'selection': 'Five training-episode folds, same fold map for every model/design.17ridge alphas1e-6..1e2. Fit input mean/std within each fold; centerY for an unpenalized intercept.',
        'interventions': 'zero and shuffled RAW latent at evaluation, then existing fitted scaler/head; no refit. Current-only designs are separately fit. Constant is training-target mean; no-change is zero delta.',
        'bootstrap': '4000 shared target-episode resamples, seed142; paired95% percentile intervals, fixed trained heads/models.',
        'fold_map': mapping, 'donors': donors.tolist(), 'mount': mount}
    write_json(output / 'protocol.json', protocol)
    if args.prepare_only:
        assert not torch.cuda.is_initialized()
        print(json.dumps({'prepared': True, 'output': str(output), 'train_samples': 2048, 'val_samples': 512,
                          'old_val_IDs_and_labels_exact': True, 'CUDA_initialized': False}), flush=True)
        return
    available = int(next(s.split()[1] for s in Path('/proc/meminfo').read_text().splitlines() if s.startswith('MemAvailable:'))) * 1024
    assert available >= 24 * 1024**3
    assert torch.cuda.is_available()
    free_cuda, total_cuda = torch.cuda.mem_get_info(); assert free_cuda >= 12 * 1024**3
    protocol.update(stage='executing_frozen_probes', available_host_memory_before=available, cuda_free_before=free_cuda,
                    gpu_before=subprocess.check_output(['nvidia-smi'], text=True))
    write_json(output / 'protocol.json', protocol)
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    train_path = output / 'train2048-highest-fp32-features.pt'
    train = torch.load(train_path, weights_only=True, map_location='cpu') if train_path.exists() else None
    encoded, evidence = {}, {}
    with torch.inference_mode():
        for name, (run, step) in models.items():
            official = name == 'official'
            mc = yaml.safe_load((release / 'dino_large_vae.yaml').read_text())['model'] if official else yaml.safe_load((run / 'resolved_config.yaml').read_text())['model']
            if official:
                mc['vision_model_id'] = cfg['model']['vision_model_id']
            checkpoint = (release / 'checkpoints/pytorch_model.pt' if official else run / ('checkpoints/last.ckpt' if step == 10000 else 'checkpoints/final.ckpt'))
            saved_required_step = api.FINAL_STEP
            try:
                api.FINAL_STEP = step if step is not None else 3000
                model, info = api.load_model(checkpoint, deepcopy(mc), contract, official=official)
            finally:
                api.FINAL_STEP = saved_required_step
            before = api.module_hashes(model); assert before['vision_encoder'] == vision_hash
            reference_name = {'baseline3000': 'baseline', 'combined3000': 'kl_mse', 'official': 'official'}.get(name)
            if reference_name is not None:
                assert info['checkpoint_sha256'] == final['models'][reference_name]['checkpoint_sha256']
                assert before == final['models'][reference_name]['module_hashes_before']
            def forbid_g(module, inputs):
                raise RuntimeError('External probes must never invoke released/local g')
            g_guard = model.state_decoder.register_forward_pre_hook(forbid_g)
            model.cuda()
            if train is None:
                train = build_train_cache(model, cfg, contract, old['train'], train_path, vision_hash, args.batch_size)
            validate_train_cache(train, old['train'], contract, vision_hash)
            parity = {'train': posterior_parity(model, train), 'val': posterior_parity(model, val)}
            encoded[name] = {'train': encode(model, train, args.batch_size), 'val': encode(model, val, args.batch_size)}
            after = api.module_hashes(model); assert before == after
            assert all(not p.requires_grad and p.grad is None for p in model.parameters())
            evidence[name] = {**info, 'module_hashes_before': before, 'module_hashes_after': after,
                              'unchanged': True, 'g_never_called': True, 'posterior_native_route_parity': parity}
            torch.save(encoded[name], output / (name + '-posterior-means.pt'))
            g_guard.remove(); del model; gc.collect(); torch.cuda.empty_cache()
            print(json.dumps({'phase': 'posterior_encoding_complete', 'model': name}), flush=True)
    yy, vy = old['train']['delta'].numpy().astype(np.float64), old['val']['delta'].numpy().astype(np.float64)
    st, sv = old['train']['state'].numpy().astype(np.float64), old['val']['state'].numpy().astype(np.float64)
    stats_path = Path(contract['data_root']) / 'meta/stats_gr00t.json'
    stats = api.read_json(stats_path)['observation.state']
    xyz_scale = np.array([(stats['max'][i] - stats['min'][i]) / 2 for i in XYZ])
    alphas = np.logspace(-6, 2, 17)
    eps, inverse = np.unique(val_eps, return_inverse=True); counts = np.bincount(inverse)
    bootstrap = np.random.default_rng(142).integers(len(eps), size=(4000, len(eps)))
    denominator = counts[bootstrap].sum(1)
    reports, all_predictions, all_metric_rows = {}, {}, {}
    for name, codes in encoded.items():
        zt, zv = codes['train']['z'].numpy().astype(np.float64), codes['val']['z'].numpy().astype(np.float64)
        ct, cv = codes['train']['current_only_z'].numpy().astype(np.float64), codes['val']['current_only_z'].numpy().astype(np.float64)
        designs = {'z': (zt, zv), 'state': (st, sv), 'state_z': (np.c_[st, zt], np.c_[sv, zv]),
                   'current_only_z': (ct, cv), 'state_current_only_z': (np.c_[st, ct], np.c_[sv, cv])}
        fits, selected, predictions = {}, {}, {}
        for design, (x, vx) in designs.items():
            fits[design], selected[design] = select_ridge(x, yy, folds, alphas)
            predictions[design] = predict(fits[design], vx)
        predictions['constant'] = np.broadcast_to(yy.mean(0), vy.shape).copy()
        predictions['no_change'] = np.zeros_like(vy)
        for design in ('z', 'state_z'):
            predictions[design + '_zero'] = predict(fits[design], np.zeros_like(zv) if design == 'z' else np.c_[sv, np.zeros_like(zv)])
            predictions[design + '_shuffled'] = predict(fits[design], zv[donors] if design == 'z' else np.c_[sv, zv[donors]])
        m = {k: metric_rows(v, vy, xyz_scale) for k, v in predictions.items()}
        pairs = [('z', 'constant'), ('z', 'no_change'), ('z', 'current_only_z'), ('z', 'z_zero'),
                 ('z', 'z_shuffled'), ('state_z', 'state'), ('state_z', 'state_current_only_z'),
                 ('state_z', 'state_z_zero'), ('state_z', 'state_z_shuffled')]
        contrasts = []
        for treatment, control in pairs:
            result = {'treatment': treatment, 'control': control, 'metrics': {}}
            for metric in m[treatment]:
                delta = m[control][metric] - m[treatment][metric]
                sums = np.bincount(inverse, weights=delta); draws = sums[bootstrap].sum(1) / denominator
                result['metrics'][metric] = {'control_minus_treatment': float(delta.mean()),
                    'ci95': np.quantile(draws, [.025, .975]).tolist()}
            contrasts.append(result)
        reports[name] = {**evidence[name], 'head_selection': selected,
            'metrics': {k: {metric: float(v.mean()) for metric, v in value.items()} for k, value in m.items()},
            'contrasts': contrasts}
        saved_fits = {k: {key: torch.from_numpy(value.copy()) for key, value in fit.items()} for k, fit in fits.items()}
        torch.save({'fits': saved_fits, 'predictions': {k: torch.from_numpy(v.copy()) for k, v in predictions.items()},
                    'donors': torch.from_numpy(donors), 'folds': torch.from_numpy(folds)}, output / (name + '-readouts.pt'))
        all_predictions[name], all_metric_rows[name] = predictions, m
    # Compare each representation with the released model using identical labels/episodes.
    official_comparisons = {}
    for name in reports:
        if name == 'official':
            continue
        official_comparisons[name] = {}
        for design in ('z', 'state_z', 'current_only_z', 'state_current_only_z'):
            official_comparisons[name][design] = {}
            for metric in all_metric_rows[name][design]:
                delta = all_metric_rows[name][design][metric] - all_metric_rows['official'][design][metric]
                draws = np.bincount(inverse, weights=delta)[bootstrap].sum(1) / denominator
                official_comparisons[name][design][metric] = {'local_minus_official': float(delta.mean()),
                    'ci95': np.quantile(draws, [.025, .975]).tolist()}
    protocol.update(train_cache=str(train_path), train_cache_sha256=api.digest(train_path), stats_sha256=api.digest(stats_path))
    result = {'complete': True, 'protocol': protocol, 'models': reports,
        'paired_comparisons_to_official': official_comparisons,
        'elapsed_seconds': time.monotonic() - started, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': ['Measures linearly accessible normalized state-change information, not all information or policy success.',
            'Input standardization can amplify tiny mu variation: an external readout gain does not prove the actual visual decoder uses z.',
            'Raw zero is a fixed-fit intervention and may be out of distribution; it differs from a separately fitted constant/state-only head.',
            'The comparison target is the official representation and its intervention relationships, not merely a positive local reconstruction gap.',
            'Quaternion metrics are componentwise delta MSE, not a rotation-angle error.',
            'Official pretraining membership is unknown; original local training/evaluation episodes are disjoint.',
            'Intervals condition on fitted heads, hyperparameter selection, model weights and fixed donor mapping; no multiple-comparison correction.']}
    write_json(output / 'results.json', result)
    print(json.dumps({'complete': True, 'output': str(output / 'results.json'), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
