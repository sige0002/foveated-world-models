"""Build CPU-only, verified replay/new-pair streams for matched continuations.

This writes metadata only. It never decodes images, constructs a model, or starts
training. ``dataset_index`` is the original upstream sampler index at epoch 0,
not an episode/frame index and not the row number in the shuffled manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SSD_RUNS = Path('/media/sadasue/SSD1/foveated-world-models/runs')
DEFAULT_SOURCE = SSD_RUNS / 'robotwin-loss-ablation-b4-20261114/kl_mse'
DEFAULT_OUTPUT = SSD_RUNS / 'robotwin-data-step-continuation-b4-20261114'


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def identity(row: dict) -> tuple[int, int]:
    return row['episode'], row['base']


def pair_hash(rows: list[dict]) -> str:
    return hashlib.sha256(json.dumps([identity(row) for row in rows],
                                    separators=(',', ':')).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', type=Path, default=DEFAULT_SOURCE)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seeds', type=int, nargs='+', default=[43, 44])
    args = parser.parse_args()
    source_run = args.source_run.resolve()
    output = args.output.resolve()
    if not output.is_relative_to(SSD_RUNS.resolve()):
        raise ValueError('Output must physically reside below SSD1 runs')
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError('Seeds must be distinct')
    output.mkdir(parents=True, exist_ok=True)
    manifest_dir = output / 'manifests'
    if manifest_dir.exists() or (output / 'manifest.json').exists():
        raise ValueError('Refusing to overwrite existing manifests')
    mount = subprocess.check_output(['findmnt', '-T', str(output), '-J'], text=True)
    import yaml
    requested_path = source_run / 'requested_config.yaml'
    resolved_path = source_run / 'resolved_config.yaml'
    split_path = source_run / 'episode_split.json'
    stream_path = source_run / 'training_stream.jsonl'
    checkpoint_path = source_run / 'checkpoints/final.ckpt'
    requested = yaml.safe_load(requested_path.read_text())
    resolved = yaml.safe_load(resolved_path.read_text())
    split = json.loads(split_path.read_text())['ids']
    contract = json.loads((source_run / 'data_contract.json').read_text())
    source = Path(requested['official_source']).resolve()
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    patch = subprocess.check_output(['git', '-C', str(source), 'diff', '--binary'])
    if head != requested['official_commit'] or head != contract['source_commit']:
        raise ValueError('Official source revision does not match original run')
    if hashlib.sha256(patch).hexdigest() != contract['source_patch_sha256']:
        raise ValueError('Source patch differs from original run')
    if digest(split_path) != contract['split_sha256']:
        raise ValueError('Original split hash mismatch')
    if resolved['data']['batch_size'] != 4 or resolved['trainer']['max_steps'] != 3000:
        raise ValueError('Expected original B4 / 3000-step run')
    if not resolved['data']['in_order'] or resolved['data']['debug_repeat_batch']:
        raise ValueError('Original sample order is not the expected ordered stream')
    if resolved['model']['loss_type'] != 'l2' or float(resolved['model']['vq_kwargs']['beta']) != 1e-5:
        raise ValueError('Expected the MSE + beta=1e-5 candidate')
    env = {**resolved['environment'], 'CUDA_VISIBLE_DEVICES': '',
           'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1'}
    os.environ.update(env)
    sys.path[:0] = [str(ROOT), str(source), *requested['dependency_roots']]
    import numpy as np
    import pyarrow.parquet as pq
    import torch
    from research.train_robotwin_stage1 import install_episode_split
    from latent_action_model.data_loader.lerobot_datamodule import LeRobotDataModule
    torch.set_num_threads(1)
    install_episode_split(split)
    dm = LeRobotDataModule(**resolved['data'])
    dm.setup('fit')
    train = dm.train_dataset
    train.mixture.set_epoch(0)
    if len(train.mixture.datasets) != 1 or train.mixture.seed != 42:
        raise ValueError('This protocol expects the audited single-dataset mixture')
    single = train.mixture.datasets[0]
    train_ids, val_ids = set(split['train']), set(split['val'])
    assert not train_ids & val_ids
    assert set(map(int, single.trajectory_ids)) == train_ids
    assert set(map(int, dm.val_dataset.mixture.datasets[0].trajectory_ids)) == val_ids
    dataset = Path(resolved['data']['data_root_dir']) / resolved['data']['data_mix']
    if dataset.resolve() != Path(contract['data_root']).resolve():
        raise ValueError('Data root differs from original run')
    episode_files = sorted((dataset / 'meta/episodes').rglob('*.parquet'))
    metadata = {}
    for path in episode_files:
        relative = str(path.relative_to(dataset))
        if digest(path) != contract['episode_metadata_sha256'][relative]:
            raise ValueError(f'Episode metadata changed: {relative}')
        for row in pq.read_table(path, columns=['episode_index', 'length', 'dataset_from_index']).to_pylist():
            episode = int(row['episode_index'])
            assert episode not in metadata
            metadata[episode] = row
    assert set(metadata) == train_ids | val_ids
    video_keys = single.modality_keys['video']
    non_wrist_keys = [key for key in video_keys if 'wrist' not in key.lower()]
    assert len(non_wrist_keys) == 1, 'Pair identity currently assumes one non-wrist camera'
    offsets = [int(v) for v in single._delta_indices[non_wrist_keys[0]]]
    assert offsets == [0, 48], offsets
    checked_video_paths = set()

    def reconstruct(index: int) -> dict:
        selected, episode, base = train.mixture.sample_step(index)
        assert selected is single
        episode, base = int(episode), int(base)
        assert episode in train_ids and episode not in val_ids
        info = metadata[episode]
        length = int(info['length'])
        assert 0 <= base < length
        selected_keys = train.mixture._select_video_keys_for_sample(single, index)
        # The upstream candidate selector may substitute another index when a
        # video is absent. Validate paths now, without opening any video.
        for key in selected_keys:
            path = Path(single.get_video_path(episode, key.replace('video.', '')))
            if path not in checked_video_paths:
                if not path.is_file():
                    raise FileNotFoundError(f'Missing video would change sampling: {path}')
                checked_video_paths.add(path)
        return {'dataset_index': index, 'episode': episode, 'base': base,
                'future': min(base + offsets[1], length - 1),
                'episode_length': length,
                'absolute_current': int(info['dataset_from_index']) + base,
                'absolute_future': int(info['dataset_from_index']) + min(base + offsets[1], length - 1)}

    batch_size, base_step, count = 4, 3000, 12000
    replay = [reconstruct(i) for i in range(count)]
    old_stream = [json.loads(line) for line in stream_path.read_text().splitlines() if line.strip()]
    if len(old_stream) != base_step:
        raise ValueError(f'Expected 3000 stream records, got {len(old_stream)}')
    for index, logged in enumerate(old_stream):
        if logged['step'] != index + 1 or pair_hash(replay[index * 4:(index + 1) * 4]) != logged['data_sha256']:
            raise ValueError(f'Reconstruction differs from actual old batch {index + 1}')
    old_identities = {identity(row) for row in replay}
    new, new_identities = [], set()
    candidate, rejected_old, rejected_duplicate = count, 0, 0
    while len(new) < count:
        row = reconstruct(candidate)
        candidate += 1
        key = identity(row)
        if key in old_identities:
            rejected_old += 1
            continue
        if key in new_identities:
            rejected_duplicate += 1
            continue
        new.append(row)
        new_identities.add(key)
    assert not new_identities & old_identities and len(new_identities) == count
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False, mmap=True)
    assert int(checkpoint['global_step']) == base_step
    assert len(checkpoint['optimizer_states']) == len(checkpoint['lr_schedulers']) == 1
    optimizer = checkpoint['optimizer_states'][0]
    optimizer_steps = Counter(int(value['step']) for value in optimizer['state'].values())
    assert set(optimizer_steps) == {base_step}
    assert all(float(group['lr']) == 3e-4 for group in optimizer['param_groups'])
    assert checkpoint['lr_schedulers'][0]['last_epoch'] == base_step
    checkpoint_info = {'path': str(checkpoint_path), 'sha256': digest(checkpoint_path),
                       'global_step': base_step, 'saved_epoch': int(checkpoint['epoch']),
                       'optimizer_state_count': len(optimizer['state']),
                       'optimizer_step_counts': dict(optimizer_steps),
                       'optimizer_group_parameter_counts': [len(g['params']) for g in optimizer['param_groups']],
                       'optimizer_group_learning_rates': [float(g['lr']) for g in optimizer['param_groups']],
                       'scheduler_last_epoch': checkpoint['lr_schedulers'][0]['last_epoch']}
    del checkpoint, optimizer

    def describe(rows: list[dict]) -> dict:
        return {'draws': len(rows), 'unique_pairs': len({identity(row) for row in rows}),
                'unique_episodes': len({row['episode'] for row in rows}),
                'unique_endpoint_frames': len({row[key] for row in rows for key in ('absolute_current', 'absolute_future')}),
                'horizon_lt48_count': sum(row['future'] - row['base'] < 48 for row in rows),
                'horizon_zero_count': sum(row['future'] == row['base'] for row in rows)}

    old_episodes = {row['episode'] for row in replay}
    new_episodes = {row['episode'] for row in new}
    old_frames = {row[key] for row in replay for key in ('absolute_current', 'absolute_future')}
    new_frames = {row[key] for row in new for key in ('absolute_current', 'absolute_future')}
    overlap = {'pair_identity_intersection': len(old_identities & new_identities),
               'episode_intersection': len(old_episodes & new_episodes),
               'endpoint_frame_intersection': len(old_frames & new_frames),
               'new_draws_sharing_any_endpoint_with_old': sum(
                   row['absolute_current'] in old_frames or row['absolute_future'] in old_frames for row in new),
               'train_validation_episode_intersection': len(train_ids & val_ids)}
    manifest_dir.mkdir()
    arms = {}
    for seed in args.seeds:
        permutation = np.random.default_rng(seed).permutation(count).tolist()
        permutation_sha = hashlib.sha256(json.dumps(permutation, separators=(',', ':')).encode()).hexdigest()
        for condition, pool in [('replay', replay), ('new', new)]:
            name = f'seed{seed}_{condition}'
            rows = [pool[i] for i in permutation]
            assert Counter(map(identity, rows)) == Counter(map(identity, pool))
            arm = {'schema_version': 1, 'arm': name, 'seed': seed, 'condition': condition,
                   'source_run': str(source_run), 'checkpoint_sha256': checkpoint_info['sha256'],
                   'base_step': base_step, 'additional_steps': 3000, 'batch_size': batch_size,
                   'mixture_epoch': 0, 'mixture_seed': 42,
                   'permutation_sha256': permutation_sha,
                   'data_sha256_by_batch': [pair_hash(rows[i:i + batch_size]) for i in range(0, count, batch_size)],
                   'rows': rows}
            path = manifest_dir / f'{name}.json'
            write_json(path, arm)
            arms[name] = {'path': str(path), 'sha256': digest(path), 'seed': seed,
                          'condition': condition, 'rows': count,
                          'permutation_sha256': permutation_sha}
    source_files = [requested_path, resolved_path, split_path, stream_path,
                    source_run / 'data_contract.json', source_run / 'train_robotwin_stage1.py',
                    Path(__file__).resolve(), ROOT / 'research/train_robotwin_stage1.py',
                    *episode_files, dataset / 'qfix/audit.json']
    source_files += sorted((source / 'latent_action_model').rglob('*.py'))
    source_files += sorted((source / 'starVLA/dataloader/gr00t_lerobot').rglob('*.py'))
    file_hashes = {str(path): digest(path) for path in source_files}
    summary = {'schema_version': 1,
               'protocol': 'robotwin_same_checkpoint_replay_vs_new_pairs_20261114',
               'source_run': str(source_run), 'checkpoint': checkpoint_info,
               'base_step': base_step, 'additional_steps': 3000, 'batch_size': batch_size,
               'learning_rate': 3e-4, 'warmup_restarts': False,
               'mixture_epoch': 0, 'mixture_seed': 42, 'seeds': args.seeds,
               'source_commit': head, 'source_patch_sha256': hashlib.sha256(patch).hexdigest(),
               'original_batches_verified': len(old_stream),
               'replay': describe(replay), 'new': describe(new), 'overlap': overlap,
               'new_pool_selection': {'candidate_start': count, 'candidate_stop_exclusive': candidate,
                                      'rejected_prior_pairs': rejected_old,
                                      'rejected_duplicate_new_pairs': rejected_duplicate},
               'video_paths_verified_without_decoding': len(checked_video_paths),
               'fixed_validation': {'split_path': str(split_path), 'split_sha256': digest(split_path),
                                    'samples': 512, 'sampler_indices': [0, 511], 'mixture_epoch': 0},
               'arms': arms, 'input_file_sha256': file_hashes,
               'mount': json.loads(mount),
               'interpretation': [
                   'New means unseen (episode, base frame) pairs, not new episodes or disjoint frames.',
                   'Replay retains the original 12000-draw multiset, including duplicates; new pairs are unique.',
                   'Seeds replicate continuation order/augmentation/posterior sampling from one common trained checkpoint.',
                   'Both seeds use the same replay pool and the same new pool; this is not independent pretraining replication.',
                   'Same per-seed permutation is applied to pool positions in replay and new.',
                   'Use original sampler indices with mixture epoch 0; validate observed episode/base and fail on substitution.',
                   'Common augmentation parameters and epsilon require runner RNG control; augmented pixels will differ.',
                   'Checkpoint epoch metadata must not reset sampler identities or restart warmup.'
               ], 'cuda_initialized': torch.cuda.is_initialized()}
    assert not summary['cuda_initialized']
    shutil.copy2(__file__, output / Path(__file__).name)
    write_json(output / 'manifest.json', summary)
    print(json.dumps({key: summary[key] for key in ['original_batches_verified', 'replay', 'new', 'overlap',
                                                  'new_pool_selection', 'cuda_initialized']}, indent=2), flush=True)
    print(f'Manifest: {output / "manifest.json"}', flush=True)


if __name__ == '__main__':
    main()
