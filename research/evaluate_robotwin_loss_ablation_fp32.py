"""Frozen, paired highest-FP32 evaluation of the four completed 3000-step arms.

Uses the exact step500 validation rows and recorded different-episode donors.
One frozen-DINO FP32 feature cache is shared by every arm and the release.
No optimizer is constructed. --prepare-only does not import torch or use CUDA.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from functools import partial
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.train_robotwin_stage1 import SSD_RUNS, install_episode_split, write_json
from research.summarize_robotwin_loss_ablation import ARMS, ARM_FACTORS, CONTRASTS

REFERENCE_STEP = 500
FINAL_STEP = 3000
SAMPLES = 512
RELEASE_SHA256 = '57c31dcba13c2998c4e866cdcfc427f14ae3c1b762a586778276e2100b48af88'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def checkpoint_ready(run):
    path = run / 'checkpoints/final.ckpt'
    summary = run / 'run_summary.json'
    return (path.is_file() and summary.is_file()
            and read_json(summary).get('phase') == 'complete'
            and read_json(summary).get('step') == FINAL_STEP)


def assert_no_failed_training(root):
    status_path = Path(root) / 'watch_status.json'
    if status_path.is_file():
        status = read_json(status_path)
        if status.get('failed_arms'):
            raise RuntimeError(f"Training watcher reported failed arms {status['failed_arms']} at {status.get('at')}; evaluation will not wait indefinitely")


def prepare(args):
    """Verify the protocol and pairing using only small CPU-side text files."""
    import yaml
    root = args.runs_root.resolve()
    out = args.output.resolve()
    cache_path = (args.cache or out / 'features512-highest-fp32.pt').resolve()
    for path in (root, out, cache_path):
        if not path.is_relative_to(SSD_RUNS.resolve()):
            raise ValueError(f'Run/output/cache must reside physically under SSD1 runs: {path}')
    mount = json.loads(subprocess.check_output(
        ['findmnt', '-J', '-T', str(SSD_RUNS.resolve()), '-o', 'TARGET,SOURCE,FSTYPE'], text=True))
    if mount['filesystems'][0]['target'] != '/media/sadasue/SSD1':
        raise RuntimeError('SSD1 is not the mounted filesystem serving the output directory')
    if args.batch_size not in (1, 2, 4):
        raise ValueError('Use B1, B2 or B4 inference, independently of the B4 training batch')
    if args.bootstrap < 1:
        raise ValueError('A positive episode-bootstrap draw count is required')
    runs = {name: root / name for name in ARMS}
    configs, requests, contracts, references = {}, {}, {}, {}
    for name, run in runs.items():
        configs[name] = yaml.safe_load((run / 'resolved_config.yaml').read_text())
        requests[name] = yaml.safe_load((run / 'requested_config.yaml').read_text())
        contracts[name] = read_json(run / 'data_contract.json')
        references[name] = read_json(run / f'z_validation_step{REFERENCE_STEP:06d}.json')
        cfg, reference = configs[name], references[name]
        actual = {'image_loss': cfg['model']['loss_type'], 'kl_beta': cfg['model']['vq_kwargs']['beta']}
        assert actual == ARM_FACTORS[name], (name, actual)
        assert cfg['trainer']['max_steps'] == FINAL_STEP and cfg['data']['batch_size'] == 4
        assert cfg['model']['state_output_scale'] == 2.0
        assert cfg['model']['warmup_steps'] == 1000
        assert requests[name]['seed'] == 42
        assert reference['step'] == REFERENCE_STEP and len(reference['rows']) == SAMPLES
        assert digest(run / 'episode_split.json') == contracts[name]['split_sha256']
        assert contracts[name] == contracts['baseline']
        # Only the two intended factors and the logging task name may differ.
        structural = deepcopy(cfg['model'])
        for key in ('loss_type', 'task_name'):
            structural.pop(key)
        structural['vq_kwargs'].pop('beta')
        if name == 'baseline':
            common_structure = structural
        else:
            assert structural == common_structure, name
        assert cfg['data'] == configs['baseline']['data']
    keys = ('episode', 'base', 'swap_valid', 'donor_episode', 'donor_base')
    rows = [{key: row[key] for key in keys} for row in references['baseline']['rows']]
    assert len({row['episode'] for row in rows}) == 415
    index = {(row['episode'], row['base']): i for i, row in enumerate(rows)}
    assert len(index) == SAMPLES, 'Expected unique validation pair identities'
    donors = [index[(row['donor_episode'], row['donor_base'])] for row in rows]
    assert sorted(donors) == list(range(SAMPLES))
    assert all(row['swap_valid'] and row['episode'] != row['donor_episode'] for row in rows)
    for name in ARMS:
        assert [{key: row[key] for key in keys} for row in references[name]['rows']] == rows
    source = Path(requests['baseline']['official_source']).resolve()
    official_source = source.parent / 'LaWAM'
    for path in (source, official_source):
        commit = subprocess.check_output(['git', '-C', str(path), 'rev-parse', 'HEAD'], text=True).strip()
        assert commit == contracts['baseline']['source_commit']
    patch = subprocess.check_output(['git', '-C', str(source), 'diff', '--binary'])
    assert hashlib.sha256(patch).hexdigest() == contracts['baseline']['source_patch_sha256']
    assert not subprocess.check_output(['git', '-C', str(official_source), 'diff', '--binary'])
    # The local source adds state_output_scale. All visual computation used below
    # must still be identical to the released source, including decoder classes.
    source_checks = {}
    core = 'latent_action_model/core/'
    for relative in ('utils/lam_encoder.py', 'utils/modules.py', 'utils/pos_embs.py', 'vq.py', 'vjepa_encoder.py'):
        file = core + relative
        assert (source / file).read_bytes() == (official_source / file).read_bytes(), file
        source_checks[file] = digest(source / file)
    preprocessing = 'latent_action_model/data_loader/video_aug.py'
    assert (source / preprocessing).read_bytes() == (official_source / preprocessing).read_bytes()
    source_checks[preprocessing] = digest(source / preprocessing)
    decoder = core + 'utils/lam_decoder.py'
    for name in ('modulate', 'AdaLNBlock', 'LAMDecoder_v2'):
        def definition(path):
            return next(node for node in ast.parse(path.read_text()).body
                        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name)
        local_ast = ast.dump(definition(source / decoder), include_attributes=False)
        public_ast = ast.dump(definition(official_source / decoder), include_attributes=False)
        assert local_ast == public_ast, name
        source_checks[decoder + ':' + name] = hashlib.sha256(local_ast.encode()).hexdigest()
    release = (args.release or SSD_RUNS.resolve().parent / 'models/lawam-lam-bd993da').resolve()
    assert (release / 'checkpoints/pytorch_model.pt').is_file()
    release_cfg = yaml.safe_load((release / 'dino_large_vae.yaml').read_text())['model']
    for key in ('latent_layer_to_use', 'norm_latents', 'norm_latents_type', 'image_hw', 'patch_size',
                'enc_add_state', 'enc_modal_mask', 'multi_input', 'vq_type'):
        assert release_cfg[key] == configs['baseline']['model'][key], key
    assert not common_structure['enc_add_state'] and not common_structure['multi_input']
    assert common_structure['num_frames'] == 2 and common_structure['latent_layer_to_use'] == -2
    assert common_structure['vq_type'] == 'vae' and not common_structure.get('disable_vq', False)
    dataset_info_path = Path(contracts['baseline']['data_root']) / 'meta/info.json'
    assert read_json(dataset_info_path)['fps'] == 30
    assert configs['baseline']['data']['frame_dt_sec'] == 1.6
    manifest = {
        'stage': 'prepared_not_executed', 'checkpoint_step_required': FINAL_STEP,
        'reference_step': REFERENCE_STEP, 'samples': SAMPLES, 'unique_episodes': 415,
        'validation_pair_and_donor_sha256': json_hash(rows),
        'reference_files': {name: {'path': str(runs[name] / f'z_validation_step{REFERENCE_STEP:06d}.json'),
                                  'sha256': digest(runs[name] / f'z_validation_step{REFERENCE_STEP:06d}.json')}
                            for name in ARMS},
        'config_sha256': {name: digest(runs[name] / 'resolved_config.yaml') for name in ARMS},
        'source': str(source), 'official_source': str(official_source),
        'source_commit': contracts['baseline']['source_commit'],
        'local_patch_sha256': hashlib.sha256(patch).hexdigest(),
        'verified_visual_source_equivalence': source_checks,
        'source_files_sha256': {name: digest(source / core / name) for name in
                               ('lam_model.py', 'lam_lightinng.py', 'utils/lam_decoder.py')},
        'release': str(release), 'release_expected_sha256': RELEASE_SHA256,
        'release_config_sha256': digest(release / 'dino_large_vae.yaml'),
        'cache': str(cache_path), 'output': str(out), 'mount': mount,
        'batch_size': args.batch_size, 'bootstrap_draws': args.bootstrap,
        'precision': 'highest FP32 DINO, encoder, VAE, decoder and state head; no autocast; TF32 disabled',
        'controls': ['real', 'zero', 'mean_global512', 'swapped_different_episode', 'current_only_IDM_x_x', 'persistence'],
        'official_scope': 'Actual released full weights, common visual evaluation only; 14-D state head not scored on 16-D labels.',
        'interval_scope': 'Paired target-episode cluster bootstrap, fixed trained weights, fixed global mean and donor permutation; no seed/precision/donor uncertainty or multiple-comparison correction.',
        'script_sha256': digest(__file__),
        'dataset_info_sha256': digest(dataset_info_path), 'frame_offset': 48,
        'ready_checkpoints': {name: checkpoint_ready(run) for name, run in runs.items()},
    }
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'prepared-protocol.json', manifest)
    return runs, configs, requests, contracts, rows, donors, release_cfg, manifest


def tensor_hash(tensor):
    import torch
    tensor = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(tuple(tensor.shape)).encode() + b'\0' + str(tensor.dtype).encode() + b'\0')
    h.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def module_hashes(model):
    import torch
    result = {}
    for name, module in model.named_children():
        h = hashlib.sha256()
        count = 0
        for key, value in module.state_dict().items():
            h.update(key.encode() + b'\0' + bytes.fromhex(tensor_hash(value)))
            count += 1
        result[name] = {'sha256': h.hexdigest(), 'state_tensors': count}
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(not tensor.is_floating_point() or tensor.dtype == torch.float32 for tensor in model.state_dict().values())
    return result


def load_model(checkpoint, cfg, contract, *, official=False):
    import torch
    from latent_action_model.core.lam_model import LatentLAMModel
    checkpoint_sha = digest(checkpoint)
    if official:
        assert checkpoint_sha == RELEASE_SHA256
    with checkpoint.open('rb') as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
    if not official:
        assert saved['global_step'] == FINAL_STEP
        assert saved['data_contract'] == contract
        hp = saved['hyper_parameters']
        for key, value in cfg.items():
            if key != 'optimizer':
                assert hp[key] == value, (key, hp.get(key), value)
    mc = dict(cfg)
    mc.pop('optimizer', None)
    model = LatentLAMModel(**mc).eval().requires_grad_(False)
    assert not model.encoder.add_state and model.vq_type == 'vae' and not model.disable_vq
    expected = model.state_dict()
    state, renamed = {}, {}
    for key, value in saved['state_dict'].items():
        assert key.startswith('lam.'), key
        target = key[len('lam.'):]
        if target not in expected and target.startswith('vision_encoder.model.layer.'):
            replacement = target.replace('vision_encoder.model.layer.', 'vision_encoder.model.model.layer.', 1)
            renamed[key] = 'lam.' + replacement
            target = replacement
        assert target not in state
        state[target] = value
    model.load_state_dict(state, strict=True)
    evidence = {'checkpoint': str(checkpoint), 'checkpoint_sha256': checkpoint_sha,
                'global_step': saved.get('global_step'), 'strictly_loaded_state_tensors': len(state),
                'renamed_checkpoint_keys': renamed,
                'state_output_scale': model.state_decoder.output_scale}
    del saved, state, expected
    gc.collect()
    with checkpoint.open('rb') as stream:
        os.posix_fadvise(stream.fileno(), 0, 0, os.POSIX_FADV_DONTNEED)
    return model, evidence


def build_cache(model, cfg, contract, reference_rows, donors, path, fingerprint, batch_size, run):
    import torch
    from torch.utils.data import DataLoader, Subset
    from latent_action_model.data_loader.lerobot_dataset import LeRobotLAMDataset
    from latent_action_model.data_loader.collate import lam_collate
    from latent_action_model.data_loader.lerobot_datamodule import _lam_worker_init_fn
    from latent_action_model.data_loader.video_aug import gpu_two_view_video_aug
    split = read_json(run / 'episode_split.json')['ids']
    install_episode_split(split)
    dc = cfg['data']
    keys = ('data_root_dir', 'data_mix', 'num_frames', 'val_tail_ratio', 'video_backend',
            'image_hw', 'frame_dt_sec', 'human_frame_dt_sec')
    ds = LeRobotLAMDataset(**{key: dc[key] for key in keys if key in dc}, mode='val')
    assert len(ds.mixture.datasets) == 1
    single = ds.mixture.datasets[0]
    assert set(single.trajectory_ids.tolist()) == set(split['val'])
    for transform in single._lam_transforms_by_video_key.values():
        transform.eval()
    loader = DataLoader(Subset(ds, range(SAMPLES)), batch_size=batch_size, num_workers=2,
                        collate_fn=partial(lam_collate, max_state_dim=32),
                        worker_init_fn=_lam_worker_init_fn, pin_memory=True,
                        prefetch_factor=2, in_order=True)
    values = {key: [] for key in ('x', 'y', 'states', 'delta', 'mask', 'emb')}
    rows, pixel_hashes = [], []
    parity_videos = None
    for batch in loader:
        # lam_collate returns raw uint8 clips. This is the exact upstream eval
        # preprocessing called by Lightning.transfer_batch_to_device; merely
        # aliasing dec_videos would skip ImageNet normalization.
        assert batch['videos'].dtype == torch.uint8 and batch['videos'].ndim == 5
        normalized, decoder_videos = gpu_two_view_video_aug(
            batch['videos'].cuda(), output_size=tuple(cfg['model']['image_hw']),
            training=False, dual_view_aug=cfg['model']['dual_view_aug'])
        batch['videos'], batch['dec_videos'] = normalized, decoder_videos
        assert torch.equal(batch['videos'], batch['dec_videos'])
        assert torch.equal(batch['delta_proprio'], batch['states'][:, 1] - batch['states'][:, 0])
        assert (batch['embodiment_ids'] == 1).all()
        mask = batch['state_mask'].any(1)
        assert mask[:, :16].all() and not mask[:, 16:].any()
        for i, (ep, base) in enumerate(zip(batch['trajectory_ids'], batch['base_indices'])):
            reference = reference_rows[len(rows)]
            ep, base = int(ep), int(base)
            assert (ep, base) == (reference['episode'], reference['base'])
            length = int(single._all_trajectory_lengths[single._trajectory_id_to_index[ep]])
            rows.append({**reference, 'donor_index': donors[len(rows)],
                         'future': min(base + 48, length - 1),
                         'full_horizon': base + 48 < length, 'episode_length': length})
            pixel_hashes.append(tensor_hash(batch['videos'][i]))
        videos = batch['videos'].cuda()
        features = model.vision_encoder.encode(torch.cat([videos, videos], dim=1), n=-2)
        assert features.dtype == torch.float32 and features.shape[1] == 4
        x, y = features[:, 2:3], features[:, -1:]
        torch.testing.assert_close(features[:, :2], torch.cat([x, y], 1), rtol=0, atol=1e-6)
        if len(rows) == len(videos):
            parity_videos = batch['videos'].cpu().clone()
            # Prove the reduced cached-feature route matches the full public API.
            native = model.get_latent_action(videos, batch['states'].cuda(), videos,
                state_mask=batch['state_mask'].cuda(), embodiment_ids=batch['embodiment_ids'].cuda(),
                predict_future_frame=True)
            nodes = model.encoder(torch.cat([x, y], 1), batch['states'].cuda(),
                                  embodiment_id=batch['embodiment_ids'].cuda())
            mu, _ = model.vq._encode(nodes)
            pred = model.decoder(x, mu)
            sp = model.state_decoder(mu, batch['states'][:, :1].cuda(), batch['embodiment_ids'].cuda())
            for left, right in ((x, native['dec_in']), (y, native['tgt']), (mu, native['quantized']),
                                (pred, native['recon']), (sp[:, :16], native['s_pred'][:, :16])):
                torch.testing.assert_close(left, right, rtol=0, atol=1e-6)
            del native, nodes, mu, pred, sp
        for key, value in dict(x=x, y=y, states=batch['states'], delta=batch['delta_proprio'],
                               mask=mask, emb=batch['embodiment_ids']).items():
            assert torch.isfinite(value).all()
            values[key].append(value.cpu())
        del features, x, y, videos
        if len(rows) % 128 == 0:
            print(json.dumps({'phase': 'FP32_DINO_cache', 'samples': len(rows)}), flush=True)
    cache = {key: torch.cat(value) for key, value in values.items()}
    stats_path = Path(contract['data_root']) / 'meta/stats_gr00t.json'
    stats = read_json(stats_path)['observation.state']
    xyz = [0, 1, 2, 8, 9, 10]
    scale = torch.tensor([(stats['max'][i] - stats['min'][i]) / 2 for i in xyz])
    movement = (cache['delta'][:, xyz] * scale).reshape(-1, 2, 3).norm(dim=-1).max(dim=-1).values
    change = (cache['y'] - cache['x']).square().mean(-1).squeeze(1)
    changed_order = change.mean(1).argsort(descending=True, stable=True).tolist()
    highest_quartile = set(changed_order[:SAMPLES // 4])
    for i, row in enumerate(rows):
        row.update(max_arm_displacement_cm=float(movement[i]) * 100,
                   moving_5cm=bool(movement[i] >= .05),
                   high_visual_change_top_quartile=i in highest_quartile,
                   gripper_changed=bool((cache['delta'][i, [7, 15]].abs() > .1).any()))
    cache['high_change_patch_indices'] = change.argsort(dim=-1, stable=True)[:, -math.ceil(change.shape[1] * .1):]
    cache['parity_videos'] = parity_videos
    cache.update(rows=rows, contract=contract, reference_identity_sha256=json_hash(reference_rows),
                 vision_encoder_fingerprint=fingerprint, precision='highest FP32; TF32 disabled',
                 pixel_sha256_per_pair=pixel_hashes, stats_sha256=digest(stats_path),
                 native_api_parity_first_batch=True, batch_size=batch_size,
                 frame_offset=48, feature_layer=-2,
                 preprocessing='upstream gpu_two_view_video_aug(training=False) on dataset eval uint8 clips')
    cache['tensor_sha256'] = {key: tensor_hash(value) for key, value in cache.items() if isinstance(value, torch.Tensor)}
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    torch.save(cache, temporary)
    temporary.replace(path)
    return cache


def validate_cache(cache, reference_rows, contract, vision_fingerprint):
    import torch
    assert cache['contract'] == contract
    assert cache['reference_identity_sha256'] == json_hash(reference_rows)
    assert cache['vision_encoder_fingerprint'] == vision_fingerprint
    assert cache['precision'] == 'highest FP32; TF32 disabled'
    assert cache['native_api_parity_first_batch']
    assert len(cache['rows']) == SAMPLES
    for row, reference in zip(cache['rows'], reference_rows):
        assert all(row[key] == value for key, value in reference.items())
        donor = cache['rows'][row['donor_index']]
        assert (donor['episode'], donor['base']) == (row['donor_episode'], row['donor_base'])
    for key, expected in cache['tensor_sha256'].items():
        assert tensor_hash(cache[key]) == expected, key
    assert cache['x'].dtype == torch.float32 and cache['y'].dtype == torch.float32
    assert torch.equal(cache['delta'], cache['states'][:, 1] - cache['states'][:, 0])


def native_route_parity(model, cache, *, official):
    """Check the cached route against the full released API on real images."""
    import torch
    videos = cache['parity_videos'].cuda()
    n = len(videos)
    states = None if official else cache['states'][:n].cuda()
    mask = None if official else cache['mask'][:n, None].expand(-1, 2, -1).cuda()
    emb = cache['emb'][:n].cuda()
    native = model.get_latent_action(videos, states, videos, state_mask=mask,
                                    embodiment_ids=emb, predict_future_frame=True)
    x, y = cache['x'][:n].cuda(), cache['y'][:n].cuda()
    nodes = model.encoder(torch.cat([x, y], 1))
    mu, _ = model.vq._encode(nodes)
    pred = model.decoder(x, mu)
    comparisons = {'current_features': (x, native['dec_in']), 'future_features': (y, native['tgt']),
                   'posterior_mu': (mu, native['quantized']), 'visual_prediction': (pred, native['recon'])}
    if not official:
        sp = model.state_decoder(mu, states[:, :1], emb)
        comparisons['state_prediction_valid16'] = (sp[:, :16], native['s_pred'][:, :16])
    for left, right in comparisons.values():
        assert left.dtype == right.dtype == torch.float32
        torch.testing.assert_close(left, right, rtol=0, atol=1e-6)
    return {'samples': n, 'all_close_atol_1e-6_rtol_0': True,
            'maximum_absolute_differences': {key: float((left - right).abs().max())
                                             for key, (left, right) in comparisons.items()}}


def evaluate_model(model, cache, batch_size, output, *, official):
    import torch
    mu_rows, lv_rows, node_rows, current_mu_rows, current_lv_rows = [], [], [], [], []
    for start in range(0, SAMPLES, batch_size):
        sl = slice(start, start + batch_size)
        x, y = cache['x'][sl].cuda(), cache['y'][sl].cuda()
        nodes = model.encoder(torch.cat([x, y], 1))
        mu, lv = model.vq._encode(nodes)
        current_nodes = model.encoder(torch.cat([x, x], 1))
        current_mu, current_lv = model.vq._encode(current_nodes)
        mu_rows.append(mu.cpu()); lv_rows.append(lv.cpu()); node_rows.append(nodes.cpu())
        current_mu_rows.append(current_mu.cpu()); current_lv_rows.append(current_lv.cpu())
        if (start + len(x)) % 128 == 0:
            print(json.dumps({'phase': 'FP32_encode', 'model': output.name, 'samples': start + len(x)}), flush=True)
    mu, lv, nodes = map(torch.cat, (mu_rows, lv_rows, node_rows))
    current_mu, current_lv = map(torch.cat, (current_mu_rows, current_lv_rows))
    assert mu.dtype == lv.dtype == nodes.dtype == torch.float32
    mean = mu.mean(0, keepdim=True)
    donors = torch.tensor([row['donor_index'] for row in cache['rows']])
    variants = {'real': mu, 'zero': torch.zeros_like(mu), 'mean': mean.expand_as(mu),
                'swapped': mu[donors], 'current_only': current_mu}
    rows = [dict(row, metrics={}) for row in cache['rows']]
    for start in range(0, SAMPLES, batch_size):
        sl = slice(start, start + batch_size)
        x, y, states, delta, emb, top = [cache[key][sl].cuda() for key in
                                        ('x', 'y', 'states', 'delta', 'emb', 'high_change_patch_indices')]
        real_pred = None
        for name, code in variants.items():
            z = code[sl].cuda()
            pred = model.decoder(x, z)
            if name == 'real':
                real_pred = pred
            err = (pred - y).square()
            metrics = {'visual_mse': err.flatten(1).mean(1),
                       'high_change_patch_mse': err.mean(-1).squeeze(1).gather(1, top).mean(1),
                       'output_difference_mse': (pred - real_pred).square().flatten(1).mean(1)}
            if not official:
                sp = model.state_decoder(z, states[:, :1], emb)
                metrics['state_mse'] = (sp[:, :16] - delta[:, :16]).square().mean(1)
            for key, value in metrics.items():
                assert torch.isfinite(value).all(), (output.name, name, key)
                for j, scalar in enumerate(value.cpu().tolist()):
                    rows[start + j]['metrics'][name + '_' + key] = scalar
        persistence = (x - y).square()
        for j in range(len(x)):
            rows[start + j]['metrics'].update(
                persistence_visual_mse=float(persistence[j].mean()),
                persistence_high_change_patch_mse=float(persistence[j].mean(-1).squeeze(0)[top[j]].mean()))
            if not official:
                rows[start + j]['metrics']['persistence_state_mse'] = float(delta[j, :16].square().mean())
        if (start + len(x)) % 128 == 0:
            print(json.dumps({'phase': 'FP32_interventions', 'model': output.name, 'samples': start + len(x)}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'mu': mu, 'logvar': lv, 'nodes': nodes, 'global512_mean': mean,
                'current_only_mu': current_mu, 'current_only_logvar': current_lv,
                'row_identity_sha256': cache['reference_identity_sha256']}, output / 'posterior-codes.pt')
    signal = float(mu.double().flatten(1).var(0, unbiased=False).sum())
    noise = float(lv.double().exp().flatten(1).sum(1).mean())
    geometry = {'mu_variance_sum': signal, 'noise_variance_sum': noise,
                'noise_fraction': noise / (noise + signal),
                'pre_norm_node_coordinate_std_mean': float(nodes.double().flatten(1).std(-1, unbiased=False).mean()),
                'mean_code_norm': float(mu.double().flatten(1).norm(dim=1).mean()),
                'mean_distance_to_current_only_mu': float((mu - current_mu).double().flatten(1).norm(dim=1).mean())}
    return {'rows': rows, 'geometry': geometry}


def statistics(reports, draws):
    """Identical episode resamples for every model and contrast in each subset."""
    import numpy as np
    reference = reports['baseline']['rows']
    groups = {
        'all': np.ones(SAMPLES, dtype=bool),
        'high_visual_change_top_quartile': np.array([row['high_visual_change_top_quartile'] for row in reference]),
        'moving_5cm': np.array([row['moving_5cm'] for row in reference]),
        'full_horizon_moving_5cm': np.array([row['moving_5cm'] and row['full_horizon'] for row in reference]),
    }
    episodes = np.array([row['episode'] for row in reference])
    arrays = {name: {key: np.array([row['metrics'][key] for row in report['rows']], dtype=np.float64)
                     for key in report['rows'][0]['metrics']} for name, report in reports.items()}
    target_metrics = ('visual_mse', 'state_mse', 'high_change_patch_mse')
    for metrics in arrays.values():
        for target in target_metrics:
            for control in ('mean', 'swapped', 'zero', 'current_only', 'persistence'):
                if 'real_' + target in metrics:
                    metrics[control + '_minus_real_' + target] = metrics[control + '_' + target] - metrics['real_' + target]
    result = {}
    for group, mask in groups.items():
        selected = episodes[mask]
        if not len(selected):
            result[group] = {'samples': 0, 'episodes': 0, 'arms': {}, 'factorial': {}}
            continue
        unique, inverse = np.unique(selected, return_inverse=True)
        counts = np.bincount(inverse)
        random_indices = np.random.default_rng(1942).integers(0, len(unique), size=(draws, len(unique)))
        denominator = counts[random_indices].sum(1)
        def estimate(values):
            values = values[mask]
            sums = np.bincount(inverse, weights=values)
            resamples = sums[random_indices].sum(1) / denominator
            return {'estimate': float(values.mean()), 'ci95': np.quantile(resamples, [.025, .975]).tolist()}
        arm_stats = {}
        for name, metrics in arrays.items():
            chosen = [key for key in metrics if any(key.endswith(target) for target in target_metrics)]
            arm_stats[name] = {key: estimate(metrics[key]) for key in chosen}
        effects = {}
        for name, coefficients in CONTRASTS.items():
            keys = [key for key in arrays['baseline'] if key.startswith(('real_', 'mean_minus_real_', 'swapped_minus_real_', 'current_only_minus_real_'))
                    and any(key.endswith(target) for target in target_metrics)]
            effects[name] = {'coefficients': coefficients, 'metrics': {
                key: estimate(sum(weight * arrays[arm][key] for arm, weight in coefficients.items())) for key in keys}}
        result[group] = {'samples': int(mask.sum()), 'episodes': len(unique), 'arms': arm_stats, 'factorial': effects}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--release', type=Path)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--bootstrap', type=int, default=4000)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--wait-for-checkpoints', action='store_true')
    args = parser.parse_args()
    runs, configs, requests, contracts, rows, donors, release_cfg, protocol = prepare(args)
    if args.prepare_only:
        print(json.dumps(protocol), flush=True)
        return
    if (args.output / 'results.json').exists():
        raise FileExistsError(args.output / 'results.json')
    pinned_keys = ('script_sha256', 'config_sha256', 'source_commit', 'local_patch_sha256',
                   'source_files_sha256', 'verified_visual_source_equivalence',
                   'validation_pair_and_donor_sha256', 'dataset_info_sha256', 'release_config_sha256')
    launch_pins = {key: protocol[key] for key in pinned_keys}
    write_json(args.output / 'launch-pins.json', launch_pins)
    while not all(checkpoint_ready(run) for run in runs.values()):
        assert_no_failed_training(args.runs_root)
        if not args.wait_for_checkpoints:
            raise RuntimeError('All four completed final3000 checkpoints are required; use --wait-for-checkpoints after GPU clearance')
        print(json.dumps({'phase': 'waiting_without_CUDA', 'ready': {name: checkpoint_ready(run) for name, run in runs.items()}}), flush=True)
        time.sleep(30)
    # A background waiter may have been prepared much earlier. Bind evidence
    # again immediately before loading any checkpoint or implementation.
    runs, configs, requests, contracts, rows, donors, release_cfg, protocol = prepare(args)
    assert {key: protocol[key] for key in pinned_keys} == launch_pins, 'Pinned script/config/source/data evidence changed while waiting'
    # Recheck the recorded full training pairing, using the same existing reader.
    from research.summarize_robotwin_loss_ablation import summarize
    pairing = summarize(runs, step=FINAL_STEP, draws=0)
    assert pairing['status'] == 'paired_snapshot', pairing['warnings']
    write_json(args.output / 'training-pairing-step3000.json', pairing)
    cfg, request = configs['baseline'], requests['baseline']
    os.environ.update(cfg['environment'])
    sys.path[:0] = [protocol['source'], *request['dependency_roots']]
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU fallback for this evaluation')
    while True:
        available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                             if line.startswith('MemAvailable:'))) * 1024
        free_cuda, total_cuda = torch.cuda.mem_get_info()
        if available >= 16 * 1024**3 and free_cuda >= 8 * 1024**3:
            break
        if not args.wait_for_checkpoints:
            raise RuntimeError('Need >=16GiB host available and >=8GiB CUDA free; defer without touching other jobs')
        print(json.dumps({'phase': 'waiting_for_memory_without_loading_model',
                          'host_available_bytes': available, 'cuda_free_bytes': free_cuda}), flush=True)
        time.sleep(30)
    protocol.update(stage='executing_frozen_evaluation',
                    available_host_memory_before=available, cuda_free_before=free_cuda,
                    torch_version=str(torch.__version__), cuda_version=torch.version.cuda,
                    actual_matmul_precision=torch.get_float32_matmul_precision(),
                    tf32_enabled=torch.backends.cuda.matmul.allow_tf32)
    write_json(args.output / 'protocol.json', protocol)
    cache_path = Path(protocol['cache'])
    cache = torch.load(cache_path, weights_only=True, map_location='cpu') if cache_path.exists() else None
    reports = {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for name in (*ARMS, 'official'):
            official = name == 'official'
            mc = deepcopy(release_cfg if official else configs[name]['model'])
            if official:
                mc['vision_model_id'] = cfg['model']['vision_model_id']
            checkpoint = (Path(protocol['release']) / 'checkpoints/pytorch_model.pt' if official
                          else runs[name] / 'checkpoints/final.ckpt')
            model, evidence = load_model(checkpoint, mc, contracts['baseline'], official=official)
            assert torch.get_float32_matmul_precision() == 'highest'
            assert not torch.backends.cuda.matmul.allow_tf32
            before = module_hashes(model)
            if name != 'baseline':
                assert before['vision_encoder'] == reports['baseline']['module_hashes_before']['vision_encoder']
            model.cuda()
            if cache is None:
                cache = build_cache(model, cfg, contracts['baseline'], rows, donors, cache_path,
                                    before['vision_encoder'], args.batch_size, runs['baseline'])
            validate_cache(cache, rows, contracts['baseline'], before['vision_encoder'])
            parity = native_route_parity(model, cache, official=official)
            if name == 'baseline':
                metadata = {key: value for key, value in cache.items() if not isinstance(value, torch.Tensor)}
                metadata.update(path=str(cache_path), sha256=digest(cache_path))
                write_json(args.output / 'feature-cache-manifest.json', metadata)
            out = args.output / name
            evaluation = evaluate_model(model, cache, args.batch_size, out, official=official)
            after = module_hashes(model)
            assert before == after
            report = {'complete': True, 'kind': name, **evidence, **evaluation,
                      'module_hashes_before': before, 'module_hashes_after': after, 'unchanged': True,
                      'all_parameters_frozen': True, 'no_parameter_gradients': True,
                      'native_api_cache_route_parity': parity,
                      'feature_cache_identity_sha256': cache['reference_identity_sha256'],
                      'precision': protocol['precision'], 'visual_only': official}
            write_json(out / 'results.json', report)
            reports[name] = report
            print(json.dumps({'phase': 'FP32_model_complete', 'model': name,
                              'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    groups = statistics(reports, args.bootstrap)
    report = {'complete': True, 'protocol': protocol, 'groups': groups,
              'models': {name: {key: value for key, value in model_report.items() if key != 'rows'}
                         for name, model_report in reports.items()},
              'elapsed_seconds': time.monotonic() - started,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
              'interpretation_limits': [
                  'Single-seed3000-update evidence; cannot prove convergence or universal causality.',
                  'Mean code estimated on these512validation samples is a diagnostic intervention.',
                  'Official visual reference uses actual released weights; its training membership is unknown.',
                  'FP32 DINO features differ from BF16 periodic-validation features; compare these arms and release within this common FP32 evaluation.',
                  'Higher latent variance or zero-code degradation alone does not demonstrate useful per-example future information.',
                  'Current-only replaces only the IDM future input with the current feature; decoder current input and teacher future target are fixed. This can be off distribution and is not a physical zero-action label.',
              ]}
    write_json(args.output / 'results.json', report)
    print(json.dumps({'complete': True, 'output': str(args.output / 'results.json'),
                      'elapsed_seconds': report['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
