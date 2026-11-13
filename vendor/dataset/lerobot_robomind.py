"""Read RoboMIND's LeRobot v3 Parquet/MP4 directly, without converted state files.

This is a LaWAM batch adapter for the on-disk LeRobot format, not a replacement
dataset format or the upstream StarVLA dataset class. Source files are read-only.
"""
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import resize

from .prepare_robomind import CAMERA, group_for_row, group_splits
from .temporal_data import timestamp_pairs

EEF = 'observation.states.end_effector'
GRIPPER = 'observation.states.gripper'
STATE_SCHEMA = 'robomind_fr3_dual_xyz_m_rpy_rad_gripper_per_arm_14d_train_minmax_v1'


def minmax(values, minimum, maximum):
    result = np.zeros_like(values, dtype=np.float32)
    active = minimum != maximum
    result[..., active] = 2 * ((values[..., active] - minimum[active]) /
                              (maximum[active] - minimum[active])) - 1
    return result


def transform_image(frame, image_hw):
    """Match upstream uint8 -> max-aspect center crop -> bilinear AA resize."""
    h, w = frame.shape[-2:]
    if w / h > 4 / 3:
        width = max(1, int(np.floor(h * (4 / 3))))
        left = (w - width) // 2
        frame = frame[..., left:left + width].contiguous()
    return resize(frame, list(image_hw), interpolation=InterpolationMode.BILINEAR, antialias=True)


def read_frame(container, stream, target, start, end, tolerance=.02):
    """Nearest actual video PTS within the episode, with an explicit sync bound."""
    container.seek(int(np.floor(target / float(stream.time_base))), stream=stream, backward=True)
    best, distance = None, float('inf')
    for frame in container.decode(stream):
        if frame.pts is None:
            raise ValueError('Video frame has no timestamp')
        time = float(frame.pts * stream.time_base)
        if start - 1e-6 <= time < end - 1e-6 and abs(time - target) < distance:
            best, distance = frame, abs(time - target)
        if time >= target:
            break
    if best is None or distance > tolerance or best.is_corrupt:
        raise ValueError(f'Missing/corrupt video observation at {target}; nearest distance={distance}')
    array = best.to_ndarray(format='rgb24')
    return torch.from_numpy(array.copy()).permute(2, 0, 1)


class LeRobotSource:
    """Shared read-only tables and training-only normalization statistics."""
    def __init__(self, root):
        self.root = Path(root).expanduser().resolve()
        self.path = self.root / 'meta/info.json'
        self.info = json.loads(self.path.read_text())
        features = self.info['features']
        expected_names = [f'{arm}_{part}' for arm in ('left', 'right')
                          for part in ('x', 'y', 'z', 'roll', 'pitch', 'yaw')]
        if (self.info['codebase_version'] != 'v3.0' or
                features[EEF]['shape'] != [12] or features[GRIPPER]['shape'] != [2] or
                features[EEF]['names'] != {'motors': expected_names} or
                features[GRIPPER]['names'] != {'motors': ['left_gripper', 'right_gripper']}):
            raise ValueError('Expected RoboMIND LeRobot v3 XYZ/RPY source schema')
        self.metadata = pq.read_table(self.root / 'meta/episodes').to_pylist()
        if len({r['episode_index'] for r in self.metadata}) != len(self.metadata):
            raise ValueError('Duplicate episode metadata')
        self.splits = group_splits(self.metadata)
        self.arrays = {}
        tables = {}
        fingerprints = {}
        for row in self.metadata:
            relative = self.info['data_path'].format(chunk_index=int(row['data/chunk_index']),
                                                     file_index=int(row['data/file_index']))
            if relative not in tables:
                path = self.root / relative
                table = pq.read_table(path, columns=['episode_index', 'timestamp', EEF, GRIPPER])
                ids = np.asarray(table['episode_index'])
                # Each episode is contiguous in LeRobot; never sort/mend source observations.
                cuts = np.r_[0, np.flatnonzero(np.diff(ids)) + 1, len(ids)]
                spans = {int(ids[a]): (a, b) for a, b in zip(cuts[:-1], cuts[1:])}
                if len(spans) != len(cuts) - 1:
                    raise ValueError('Noncontiguous episode rows')
                tables[relative] = (table, spans)
                fingerprints[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
            table, spans = tables[relative]
            episode = int(row['episode_index'])
            a, b = spans[episode]
            part = table.slice(a, b - a)
            times = np.asarray(part['timestamp'], dtype=np.float64)
            eef = np.asarray(part[EEF].to_pylist(), dtype=np.float32)
            grip = np.asarray(part[GRIPPER].to_pylist(), dtype=np.float32)
            states = np.column_stack((eef[:, :6], grip[:, 0], eef[:, 6:], grip[:, 1]))
            if (len(times) != row['length'] or len(times) < 2 or
                    not np.isfinite(times).all() or np.any(np.diff(times) <= 0) or
                    not np.isfinite(states).all() or np.any((grip < 0) | (grip > 1))):
                raise ValueError(f'Invalid raw episode {episode}; no silent filling or dropping')
            self.arrays[episode] = dict(timestamps=times, states=states)
        train_ids = [int(r['episode_index']) for r in self.metadata
                     if self.splits[group_for_row(r)] == 'train']
        self.minimum = np.min([self.arrays[i]['states'].min(axis=0) for i in train_ids], axis=0)
        self.maximum = np.max([self.arrays[i]['states'].max(axis=0) for i in train_ids], axis=0)
        self.contract = dict(version=1, format='lerobot_v3', camera=CAMERA, state_schema=STATE_SCHEMA,
                             normalization=dict(mode='min_max', fit_split='train', clip=False,
                                                min=self.minimum.tolist(), max=self.maximum.tolist()),
                             image_transform='uint8_max_aspect_4_3_center_crop_bilinear_antialias',
                             missing_observations='error', source_parquet_sha256=fingerprints,
                             info_sha256=hashlib.sha256(self.path.read_bytes()).hexdigest(),
                             episodes={s: sorted(int(r['episode_index']) for r in self.metadata
                                                 if self.splits[group_for_row(r)] == s)
                                       for s in ('train', 'validation', 'test')})
        # Include video offsets and membership, which define the state/image alignment.
        self.contract['episode_metadata_sha256'] = {
            str(p.relative_to(self.root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((self.root / 'meta/episodes').rglob('*.parquet'))}


class LeRobotRoboMINDDataset(Dataset):
    def __init__(self, source, split, *, image_hw=(256, 256), max_state_dim=32,
                 num_embodiments=32, horizon_seconds=1.2, tolerance_seconds=None):
        if split not in ('train', 'validation', 'test'):
            raise ValueError(f'Unknown split {split}')
        if max_state_dim < 14 or num_embodiments <= 1 or len(image_hw) != 2 or min(image_hw) <= 0:
            raise ValueError('Invalid model capacity/image dimensions; cannot truncate states')
        self.source, self.path, self.split = source, source.path, split
        self.image_hw = tuple(image_hw)
        self.rows = [r for r in source.metadata if source.splits[group_for_row(r)] == split]
        self.episodes = {str(r['episode_index']) for r in self.rows}
        self.groups = {group_for_row(r) for r in self.rows}
        self.schemas = {1: (STATE_SCHEMA, 14)}
        self.pairs = []
        self.videos = []
        for i, row in enumerate(self.rows):
            path = source.root / source.info['video_path'].format(
                video_key=CAMERA, chunk_index=int(row[f'videos/{CAMERA}/chunk_index']),
                file_index=int(row[f'videos/{CAMERA}/file_index']))
            if not path.is_file():
                raise FileNotFoundError(path)
            self.videos.append(path)
            times = self._arrays(i)['timestamps']
            self.pairs.extend((i, a, b, tail) for a, b, tail in timestamp_pairs(
                times, horizon=horizon_seconds, tolerance=tolerance_seconds))
        if not self.pairs:
            raise ValueError(f'No usable pairs for {split}')
        self.contract = dict(source.contract, image_hw=list(image_hw), max_state_dim=max_state_dim,
                             horizon_seconds=horizon_seconds, tolerance_seconds=tolerance_seconds)

    def _arrays(self, i):
        return self.source.arrays[int(self.rows[i]['episode_index'])]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        i, a, b, tail = self.pairs[index]
        row, data = self.rows[i], self._arrays(i)
        times = data['timestamps'][[a, b]]
        start, end = (float(row[f'videos/{CAMERA}/{key}_timestamp']) for key in ('from', 'to'))
        with av.open(str(self.videos[i])) as container:
            stream = container.streams.video[0]
            stream.codec_context.thread_count = 1
            frames = [transform_image(read_frame(container, stream, start + t, start, end), self.image_hw)
                      for t in times]
        states = minmax(data['states'][[a, b]], self.source.minimum, self.source.maximum)
        return dict(frames=torch.stack(frames), proprio=torch.from_numpy(states), embodiment_id=1,
                    dataset_name='RoboMIND-franka_fr3_dual', trajectory_id=str(row['episode_index']),
                    base_index=a, timestamps=torch.from_numpy(times.copy()), dt=float(times[1] - times[0]),
                    is_tail=tail, split_group=group_for_row(row))


def lerobot_collate(batch, *, max_state_dim=32):
    from latent_action_model.data_loader.collate import lam_collate
    if max_state_dim < 14:
        raise ValueError('State truncation is forbidden')
    result = lam_collate(batch, max_state_dim=max_state_dim)
    result['timestamps'] = torch.stack([s['timestamps'] for s in batch])
    result['dt'] = torch.tensor([s['dt'] for s in batch], dtype=torch.float64)
    result['is_tail'] = torch.tensor([s['is_tail'] for s in batch])
    result['split_group'] = [s['split_group'] for s in batch]
    return result
