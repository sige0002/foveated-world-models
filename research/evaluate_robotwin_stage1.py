"""Frozen RoboTwin checkpoint evaluation using the saved upstream data contract.

Compare future-informed z with zero, train-mean, other-episode, and current-only
z on the same held-out images. No parameter updates or dataset modifications.
"""
from __future__ import annotations

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


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=512)
    p.add_argument('--train-mean-samples', type=int, default=256)
    p.add_argument('--release-hardlink', action='store_true')
    p.add_argument('--visual-only', action='store_true',
                   help='Compare visual prediction without evaluating the state head')
    p.add_argument('--release-config', type=Path,
                   help='Model YAML distributed with an official release checkpoint')
    p.add_argument('--source', type=Path, help='Explicit upstream implementation directory')
    p.add_argument('--expected-checkpoint-sha256')
    p.add_argument('--reference-checkpoint', type=Path,
                   help='Verify the released visual backbone against this local checkpoint')
    p.add_argument('--reference-results', type=Path,
                   help='Require identical sample identities, donors and persistence metrics')
    a = p.parse_args()
    if a.release_config and not (a.visual_only and a.source and
                                a.expected_checkpoint_sha256 and a.reference_checkpoint):
        p.error('Release evaluation requires --visual-only, --source, '
                '--expected-checkpoint-sha256 and --reference-checkpoint')
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / 'results.json').exists():
        raise ValueError('Evaluation results already exist')
    import yaml
    cfg = yaml.safe_load((a.run / 'resolved_config.yaml').read_text())
    requested = yaml.safe_load((a.run / 'requested_config.yaml').read_text())
    os.environ.update(cfg['environment'])
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1')
    source = a.source or Path(requested['official_source'])
    sys.path[:0] = [str(source), *requested['dependency_roots']]
    contract = json.loads((a.run / 'data_contract.json').read_text())
    split = json.loads((a.run / 'episode_split.json').read_text())['ids']
    assert digest(a.run / 'episode_split.json') == contract['split_sha256']
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == contract['source_commit']
    patch = subprocess.check_output(['git', '-C', str(source), 'diff', '--binary'])
    if a.release_config:
        assert not patch, 'Official release evaluation requires unmodified upstream code'
    else:
        assert hashlib.sha256(patch).hexdigest() == contract['source_patch_sha256']
    install_episode_split(split)

    import numpy as np
    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Subset
    from latent_action_model.core.lam_lightinng import VJEPA_LAM
    from latent_action_model.data_loader.lerobot_dataset import LeRobotLAMDataset
    from latent_action_model.data_loader.collate import lam_collate
    from latent_action_model.data_loader.lerobot_datamodule import _lam_worker_init_fn
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required for this evaluation')
    started = time.monotonic()
    checkpoint_hash = digest(a.checkpoint)
    if a.expected_checkpoint_sha256:
        assert checkpoint_hash == a.expected_checkpoint_sha256
    # GB10 shares host RAM: hashing must not leave the entire 8 GB checkpoint
    # resident as clean page cache before allocating another inference model.
    with a.checkpoint.open('rb') as checkpoint_file:
        os.posix_fadvise(checkpoint_file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    saved = torch.load(a.checkpoint, weights_only=True, map_location='cpu', mmap=True)
    if a.release_config:
        mc = dict(yaml.safe_load(a.release_config.read_text())['model'])
        mc['vision_model_id'] = cfg['model']['vision_model_id']
        step = saved.get('global_step')
    else:
        assert saved['data_contract'] == contract
        assert saved['hyper_parameters']['state_output_scale'] == 2.0
        step = int(saved['global_step'])
        mc = dict(cfg['model'])
    mc.pop('optimizer')
    model = VJEPA_LAM(**mc).eval().requires_grad_(False)
    state = saved['state_dict']
    renamed = {}
    expected_keys = model.state_dict()
    for key in list(state):
        if key not in expected_keys and key.startswith('lam.vision_encoder.model.layer.'):
            target = key.replace('lam.vision_encoder.model.layer.',
                                 'lam.vision_encoder.model.model.layer.', 1)
            assert target in expected_keys and target not in state
            renamed[key] = target
            state[target] = state.pop(key)
    backbone_check = None
    if a.reference_checkpoint:
        reference = torch.load(a.reference_checkpoint, weights_only=True,
                               map_location='cpu', mmap=True)['state_dict']
        keys = [k for k in state if k.startswith('lam.vision_encoder.')]
        assert set(keys) == {k for k in reference if k.startswith('lam.vision_encoder.')}
        for key in keys:
            assert torch.equal(state[key], reference[key]), key
        backbone_check = {'tensors': len(keys), 'all_exact': True}
        del reference
    model.load_state_dict(state, strict=True)
    del state, expected_keys
    del saved
    with a.checkpoint.open('rb') as checkpoint_file:
        os.posix_fadvise(checkpoint_file.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    if a.release_hardlink:
        pin = json.loads((a.output / 'checkpoint-pin.json').read_text())
        assert a.checkpoint == Path(pin['temporary_hardlink'])
        assert a.checkpoint.name == 'source-checkpoint.ckpt'
        a.checkpoint.unlink()
    model = model.cuda()
    torch.cuda.reset_peak_memory_stats()
    lam = model.lam
    assert not lam.encoder.add_state
    if not a.release_config:
        assert lam.state_decoder.output_scale == 2.0
    if a.visual_only:
        # This head is not an input to the visual decoder. Its incompatible
        # released 14-D output must not be scored against 16-D RoboTwin labels.
        lam.state_decoder = None
    assert mc['vq_type'] == 'vae' and not mc.get('disable_vq', False)
    protocol = {'checkpoint_step': step, 'checkpoint_sha256': checkpoint_hash,
                'script_sha256': digest(__file__), 'data_contract': contract,
                'samples': a.samples, 'train_mean_samples': a.train_mean_samples,
                'precision': 'bf16 mixed, matching training validation',
                'selection': 'same official validation mixture indices 0..N-1 as periodic validation',
                'scope': 'Frozen model; future-informed posterior-mean z. Ablations are interventions, not separately trained baselines or policy success rates.',
                'groups': 'full horizon; end-effector displacement >=5cm in either arm; gripper change',
                'interval_scope': '95% paired episode-cluster bootstrap conditional on this checkpoint and these samples'}
    protocol.update(model_kind='official_release' if a.release_config else 'local',
                    visual_only=a.visual_only, model_max_state_dim=mc['max_state_dim'],
                    implementation_source=str(source),
                    implementation_patch_sha256=hashlib.sha256(patch).hexdigest(),
                    release_config_sha256=digest(a.release_config) if a.release_config else None,
                    renamed_checkpoint_keys=renamed, backbone_check=backbone_check,
                    split_scope='Held out for the local run; official training membership is unknown')
    write_json(a.output / 'protocol.json', protocol)
    print(json.dumps({'phase': 'loaded', 'step': step, 'checkpoint_sha256': checkpoint_hash}), flush=True)

    def loader_for(mode, n):
        dc = cfg['data']
        keys = ('data_root_dir', 'data_mix', 'num_frames', 'val_tail_ratio', 'video_backend',
                'image_hw', 'frame_dt_sec', 'human_frame_dt_sec')
        ds = LeRobotLAMDataset(**{k: dc[k] for k in keys if k in dc}, mode=mode)
        assert set(ds.mixture.datasets[0].trajectory_ids.tolist()) == set(split[mode])
        # Calibration uses training episodes, with deterministic evaluation images.
        for single in ds.mixture.datasets:
            for transform in single._lam_transforms_by_video_key.values():
                transform.eval()
        dl = DataLoader(Subset(ds, list(range(n))), batch_size=4, num_workers=2,
                        collate_fn=partial(lam_collate, max_state_dim=32),
                        worker_init_fn=_lam_worker_init_fn, pin_memory=True,
                        prefetch_factor=2, in_order=True)
        return ds, dl

    cache = {k: [] for k in ('x', 'y', 'z', 'current_z', 'states', 'delta', 'mask', 'emb')}
    rows = []
    native_outputs = []
    with torch.inference_mode():
        train_ds, dl = loader_for('train', a.train_mean_samples)
        means = []
        for i, batch in enumerate(dl):
            batch = model.transfer_batch_to_device(batch, torch.device('cuda'), 0)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = lam.get_latent_action(batch['videos'],
                    None if a.visual_only else batch['states'], batch['dec_videos'],
                    state_mask=None if a.visual_only else batch['state_mask'],
                    embodiment_ids=batch['embodiment_ids'])
            means.append(out['quantized'].float().cpu())
        train_mean = torch.cat(means).mean(0, keepdim=True).cuda()
        del dl, train_ds, means
        print(json.dumps({'phase': 'train_mean_encoded', 'samples': a.train_mean_samples}), flush=True)
        val_ds, dl = loader_for('val', a.samples)
        single = val_ds.mixture.datasets[0]
        for i, batch in enumerate(dl):
            eps = [int(v) for v in batch['trajectory_ids']]
            bases = [int(v) for v in batch['base_indices']]
            assert all(ep in set(split['val']) for ep in eps)
            batch = model.transfer_batch_to_device(batch, torch.device('cuda'), 0)
            assert torch.equal(batch['videos'], batch['dec_videos'])
            assert torch.equal(batch['delta_proprio'], batch['states'][:, 1] - batch['states'][:, 0])
            assert (batch['embodiment_ids'] == 1).all()
            assert (batch['state_mask'].any(1).sum(-1) == 16).all()
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out = lam.get_latent_action(batch['videos'],
                    None if a.visual_only else batch['states'], batch['dec_videos'],
                    state_mask=None if a.visual_only else batch['state_mask'],
                    embodiment_ids=batch['embodiment_ids'],
                    predict_future_frame=True)
                static_nodes = lam.encoder(out['dec_in'].expand(-1, 2, -1, -1),
                    None if a.visual_only else batch['states'][:, :1].expand(-1, 2, -1),
                    embodiment_id=batch['embodiment_ids'])
                if i == 0:
                    nodes = lam.encoder(torch.cat([out['dec_in'], out['tgt']], 1),
                                        None if a.visual_only else batch['states'],
                                        embodiment_id=batch['embodiment_ids'])
            with torch.autocast('cuda', enabled=False):
                current_z = lam.vq.inference(static_nodes.float())[0]
                if i == 0:
                    check_z = lam.vq.inference(nodes.float())[0]
                    torch.testing.assert_close(check_z, out['quantized'], rtol=0, atol=1e-5)
            values = dict(x=out['dec_in'], y=out['tgt'], z=out['quantized'], current_z=current_z,
                          states=batch['states'], delta=batch['delta_proprio'],
                          mask=batch['state_mask'].any(1), emb=batch['embodiment_ids'])
            for k, v in values.items():
                assert torch.isfinite(v).all()
                cache[k].append(v.cpu())
            native = [F.smooth_l1_loss(out['recon'].float(), out['tgt'].float(),
                      beta=.1, reduction='none').flatten(1).mean(1)]
            if not a.visual_only:
                native.append(((out['s_pred'].float() - batch['delta_proprio']).square()
                               * values['mask']).sum(1) / 16)
            native_outputs.append(torch.stack(native, 1).cpu())
            for ep, base in zip(eps, bases):
                length = int(single._all_trajectory_lengths[single._trajectory_id_to_index[ep]])
                rows.append({'episode': ep, 'base': base, 'future': min(base + 48, length - 1),
                             'full_horizon': base + 48 < length, 'episode_length': length})
            if i % 32 == 0:
                print(json.dumps({'phase': 'val_encoded', 'samples': len(rows)}), flush=True)
        del dl, val_ds
        cache = {k: torch.cat(v) for k, v in cache.items()}
        native_outputs = torch.cat(native_outputs)
        episodes = np.array([r['episode'] for r in rows])
        rng = np.random.default_rng(1942)
        for _ in range(10000):
            donors = rng.permutation(len(rows))
            if np.all(episodes[donors] != episodes):
                break
        else:
            raise RuntimeError('Cannot create a permutation with different-episode donors')
        for i, r in enumerate(rows):
            r['donor_index'] = int(donors[i])
            r['donor_episode'] = int(episodes[donors[i]])

        stats = json.loads((Path(contract['data_root']) / 'meta/stats_gr00t.json').read_text())['observation.state']
        xyz = [0, 1, 2, 8, 9, 10]
        quat = [3, 4, 5, 6, 11, 12, 13, 14]
        grip = [7, 15]
        xyz_scale = torch.tensor([(stats['max'][i] - stats['min'][i]) / 2 for i in xyz], device='cuda')

        def metrics(pred, sp, x, y, states, delta):
            pred, x, y = [v.float() for v in (pred, x, y)]
            err = (pred - y).square()
            change = (y - x).square().mean(-1).squeeze(1)
            top = change.argsort(dim=-1, stable=True)[:, -26:]
            visual = {'visual_mse': err.flatten(1).mean(1),
                      'visual_smooth_l1': F.smooth_l1_loss(pred, y, beta=.1, reduction='none').flatten(1).mean(1),
                      'visual_cosine': F.cosine_similarity(pred, y, dim=-1).flatten(1).mean(1),
                      'high_change_patch_mse': err.mean(-1).squeeze(1).gather(1, top).mean(1)}
            if a.visual_only:
                return visual
            sp = sp.float()
            se = (sp - delta).square()
            cm = ((sp[:, xyz] - delta[:, xyz]) * xyz_scale).reshape(-1, 2, 3).norm(dim=-1).mean(1) * 100
            q_pred = (states[:, 0, quat] + sp[:, quat]).reshape(-1, 2, 4)
            q_true = states[:, 1, quat].reshape(-1, 2, 4)
            q_pred = F.normalize(q_pred, dim=-1)
            q_true = F.normalize(q_true, dim=-1)
            angle = 2 * torch.acos((q_pred * q_true).sum(-1).abs().clamp(0, 1)) * (180 / torch.pi)
            return {**visual,
                    'state_mse': se[:, :16].mean(1), 'xyz_normalized_mse': se[:, xyz].mean(1),
                    'quaternion_component_mse': se[:, quat].mean(1), 'gripper_normalized_mse': se[:, grip].mean(1),
                    'position_error_cm': cm, 'rotation_error_degrees': angle.mean(1)}

        modes = ['real', 'zero', 'train_mean', 'other_episode', 'current_only', 'persistence']
        for start in range(0, len(rows), 4):
            sl = slice(start, start + 4)
            x, y, z, states, delta, emb = [cache[k][sl].cuda() for k in ('x', 'y', 'z', 'states', 'delta', 'emb')]
            codes = {'real': z, 'zero': torch.zeros_like(z), 'train_mean': train_mean.expand_as(z),
                     'other_episode': cache['z'][donors[start:start + 4]].cuda(),
                     'current_only': cache['current_z'][sl].cuda()}
            measures = {}
            for name in modes:
                if name == 'persistence':
                    pred, sp = x, torch.zeros_like(delta)
                else:
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        pred = lam.decoder(x, codes[name])
                        sp = None if a.visual_only else lam.state_decoder(codes[name], states[:, :1], emb)
                measures[name] = metrics(pred, sp, x, y, states, delta)
            replay_values = [measures['real']['visual_smooth_l1']]
            if not a.visual_only:
                replay_values.append(measures['real']['state_mse'])
            replay = torch.stack(replay_values, 1).cpu()
            torch.testing.assert_close(replay, native_outputs[sl], rtol=0, atol=1e-6)
            move = (delta[:, xyz] * xyz_scale).reshape(-1, 2, 3).norm(dim=-1).max(dim=-1).values
            for j in range(len(x)):
                row = rows[start + j]
                row['max_arm_displacement_cm'] = float(move[j]) * 100
                row['gripper_changed'] = bool((delta[j, grip].abs() > .1).any())
                row['metrics'] = {name: {k: float(v[j]) for k, v in m.items()} for name, m in measures.items()}
            if start % 128 == 0:
                print(json.dumps({'phase': 'ablated', 'samples': start + len(x)}), flush=True)

    names = list(rows[0]['metrics']['real'])
    groups = {'all': np.ones(len(rows), dtype=bool),
              'full_horizon': np.array([r['full_horizon'] for r in rows]),
              'moving_5cm': np.array([r['max_arm_displacement_cm'] >= 5 for r in rows]),
              'full_horizon_moving_5cm': np.array([r['full_horizon'] and r['max_arm_displacement_cm'] >= 5 for r in rows]),
              'gripper_changed': np.array([r['gripper_changed'] for r in rows])}
    arrays = {mode: {k: np.array([r['metrics'][mode][k] for r in rows]) for k in names} for mode in modes}
    groups_report = {}
    for name, mask in groups.items():
        means = {mode: {k: float(v[mask].mean()) for k, v in values.items()} for mode, values in arrays.items()} if mask.any() else {}
        groups_report[name] = {'samples': int(mask.sum()), 'unique_episodes': len(set(episodes[mask])), 'metrics': means}
    intervals = {}
    unique, inverse = np.unique(episodes, return_inverse=True)
    counts = np.bincount(inverse)
    draws = np.random.default_rng(142).integers(0, len(unique), size=(4000, len(unique)))
    for mode in modes[1:]:
        intervals[mode] = {}
        for metric in ('visual_mse', 'visual_smooth_l1', 'state_mse', 'high_change_patch_mse'):
            if metric not in names:
                continue
            diff = arrays[mode][metric] - arrays['real'][metric]
            sums = np.bincount(inverse, weights=diff)
            bs = sums[draws].sum(1) / counts[draws].sum(1)
            intervals[mode][metric] = {'control_minus_real': float(diff.mean()),
                'ci95': np.quantile(bs, [.025, .975]).tolist(),
                'real_relative_error_reduction_percent': float(100 * diff.mean() / arrays[mode][metric].mean()),
                'fraction_real_better': float((diff > 0).mean())}
    z = cache['z'].double().flatten(1)
    eig = torch.linalg.eigvalsh(torch.cov(z.T)).clamp_min(0)
    probs = eig / eig.sum().clamp_min(1e-30)
    geometry = {'mean_dimension_std': float(z.std(0).mean()),
                'mean_code_norm': float(z.norm(dim=1).mean()),
                'covariance_effective_rank': float((-(probs * probs.clamp_min(1e-30).log()).sum()).exp()),
                'mean_distance_to_current_only_z': float((z - cache['current_z'].double().flatten(1)).norm(dim=1).mean())}
    recorded = [json.loads(line) for line in (a.run / 'validation.jsonl').read_text().splitlines()]
    logged = next((v for v in recorded if v['step'] == step), None)
    replay_check = None
    if logged is not None and a.samples == 512:
        replay_check = {key: {'logged': logged['metrics'][src], 'replayed': groups_report['all']['metrics']['real'][key],
            'absolute_difference': abs(logged['metrics'][src] - groups_report['all']['metrics']['real'][key])}
            for key, src in [('visual_smooth_l1', 'val/recon_loss'), ('state_mse', 'val/state_loss')]
            if key in names}
        if any(v['absolute_difference'] > 1e-5 for v in replay_check.values()):
            raise RuntimeError(f'Validation replay mismatch: {replay_check}')
    reference_check = None
    if a.reference_results:
        reference = json.loads(a.reference_results.read_text())
        assert len(rows) == len(reference['rows'])
        identity = ('episode', 'base', 'future', 'full_horizon', 'episode_length',
                    'donor_index', 'donor_episode', 'gripper_changed')
        same_checkpoint = checkpoint_hash == reference['protocol']['checkpoint_sha256']
        max_difference = 0.0
        for row, prior in zip(rows, reference['rows']):
            assert all(row[k] == prior[k] for k in identity)
            for mode in (modes if same_checkpoint else ['persistence']):
                for metric in names:
                    difference = abs(row['metrics'][mode][metric] - prior['metrics'][mode][metric])
                    max_difference = max(max_difference, difference)
                    assert difference < 1e-6, (mode, metric, difference)
        reference_check = {'path': str(a.reference_results), 'sha256': digest(a.reference_results),
                           'same_pairs_and_donors': True, 'same_checkpoint': same_checkpoint,
                           'compared_modes': modes if same_checkpoint else ['persistence'],
                           'maximum_metric_difference': max_difference}
    report = {'complete': True, 'protocol': protocol, 'groups': groups_report,
              'control_minus_real': intervals, 'z_geometry': geometry,
              'validation_replay': replay_check, 'reference_check': reference_check, 'rows': rows,
              'elapsed_seconds': time.monotonic() - started,
              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(),
              'cuda_peak_reserved_bytes': torch.cuda.max_memory_reserved()}
    write_json(a.output / 'results.json', report)
    print(json.dumps({k: v for k, v in report.items() if k not in ('rows', 'protocol')}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
