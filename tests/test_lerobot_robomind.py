"""Regression coverage for raw LeRobot state semantics and held-out statistics."""
import json
from pathlib import Path
import sys

import av
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torchvision.transforms.v2 import InterpolationMode, Resize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor/lawam'))
from train_stage1 import check_resume_contract, load_config
from vendor.dataset.lerobot_robomind import (
    CAMERA, EEF, GRIPPER, LeRobotSource, LeRobotRoboMINDDataset,
    lerobot_collate, minmax, transform_image,
)
from vendor.dataset.prepare_robomind import group_for_row, group_splits


@pytest.fixture
def raw_root(tmp_path):
    (tmp_path / 'meta/episodes').mkdir(parents=True)
    (tmp_path / 'data/chunk-000').mkdir(parents=True)
    info = dict(codebase_version='v3.0', data_path='data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet',
                video_path='videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4',
                features={EEF: dict(shape=[12], names={'motors': [f'{arm}_{part}'
                    for arm in ('left', 'right') for part in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')]}),
                          GRIPPER: dict(shape=[2], names={'motors': ['left_gripper', 'right_gripper']})})
    (tmp_path / 'meta/info.json').write_text(json.dumps(info))
    metadata = [dict(episode_index=i, length=20, tasks=['test'], **{
        'data/chunk_index': 0, 'data/file_index': 0,
        f'videos/{CAMERA}/chunk_index': 0, f'videos/{CAMERA}/file_index': i,
        f'videos/{CAMERA}/from_timestamp': 0., f'videos/{CAMERA}/to_timestamp': 2.}) for i in range(10)]
    splits = group_splits(metadata)
    rows = []
    for row in metadata:
        i = row['episode_index']
        heldout = splits[group_for_row(row)] != 'train'
        for j in range(20):
            eef = np.arange(12, dtype=np.float32) + (1000 if heldout else i) + j / 10
            eef[3:6] = [.1, -.2, .3]  # Preserve these RPY values, not rotation-matrix columns.
            rows.append({'episode_index': i, 'timestamp': j / 10, EEF: eef.tolist(), GRIPPER: [.25, .75]})
        path = tmp_path / info['video_path'].format(video_key=CAMERA, chunk_index=0, file_index=i)
        path.parent.mkdir(parents=True, exist_ok=True)
        with av.open(str(path), 'w') as output:
            stream = output.add_stream('libx264', rate=10)
            stream.width, stream.height, stream.pix_fmt = 32, 18, 'yuv420p'
            for j in range(20):
                frame = av.VideoFrame.from_ndarray(np.full((18, 32, 3), 40 + j * 8, np.uint8), format='rgb24')
                for packet in stream.encode(frame):
                    output.mux(packet)
            for packet in stream.encode():
                output.mux(packet)
    pq.write_table(pa.Table.from_pylist(metadata), tmp_path / 'meta/episodes/file-000.parquet')
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / 'data/chunk-000/file-000.parquet')
    return tmp_path


def test_raw_pose_train_statistics_and_batch(raw_root):
    source = LeRobotSource(raw_root)
    assert source.maximum[0] < 20  # Held-out extreme values must not enter statistics.
    np.testing.assert_allclose(source.arrays[0]['states'][0, 3:6], [.1, -.2, .3])
    np.testing.assert_allclose(source.arrays[0]['states'][0, [6, 13]], [.25, .75])
    for split in ('train', 'validation', 'test'):
        ds = LeRobotRoboMINDDataset(source, split)
        sample = ds[0]
        assert sample['frames'].dtype == torch.uint8
        assert sample['frames'].shape == (2, 3, 256, 256)
        assert sample['frames'].min() > 0  # No introduced black borders.
        assert sample['frames'][1].float().mean() > sample['frames'][0].float().mean() + 70
        batch = lerobot_collate([sample])
        assert batch['state_mask'][..., :14].all()
        assert not batch['state_mask'][..., 14:].any()
        torch.testing.assert_close(batch['delta_proprio'], batch['states'][:, 1] - batch['states'][:, 0])
        if split != 'train':
            assert sample['proprio'][0, 0] > 1  # Official normalization does not clip.


def test_crop_resize_matches_reference():
    frame = torch.arange(3 * 18 * 32).remainder(256).to(torch.uint8).reshape(3, 18, 32)
    expected = Resize((16, 16), interpolation=InterpolationMode.BILINEAR, antialias=True)(frame[:, :, 4:28])
    torch.testing.assert_close(transform_image(frame, (16, 16)), expected, rtol=0, atol=0)
    np.testing.assert_array_equal(minmax(np.array([[8., 5.]], np.float32),
                                       np.array([2., 5.]), np.array([4., 5.])), [[5., 0.]])


def test_resume_rejects_legacy_or_changed_contract():
    contract = {'state_schema': '14d', 'normalization': [1, 2]}
    check_resume_contract({'data_contract': contract}, contract)
    for checkpoint in ({}, {'data_contract': {'state_schema': '20d'}}):
        with pytest.raises(ValueError, match='data contract'):
            check_resume_contract(checkpoint, contract)


def test_default_schedule_and_legacy_explicit():
    config = load_config(ROOT / 'configs/stage1_robomind.yaml')
    assert config['data']['format'] == 'lerobot_v3'
    assert 'train_manifest' not in config['data']
    assert config['model']['warmup_steps'] == 5000
    assert config['trainer']['max_steps'] == 10000
    assert config['data']['batch_size'] == 2 and config['data']['horizon_seconds'] == 1.2
    assert load_config(ROOT / 'configs/stage1_robomind_legacy.yaml')['data']['format'] == 'robomind_video'
