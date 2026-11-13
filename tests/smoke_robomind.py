"""Real RoboMIND + frozen DINO: loader checks, two steps, save and resume."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import av
import torch
import yaml
from torchvision.transforms import v2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'vendor/lawam'))
from vendor.dataset.robomind import RoboMINDDataset, check_disjoint
from vendor.dataset.lerobot_robomind import LeRobotSource, LeRobotRoboMINDDataset, lerobot_collate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/data2/yuki_sadasue/datasets/RoboMIND-lerobot/franka_fr3_dual'))
    parser.add_argument('--legacy-root', type=Path, help='Optional old manifests: verify every episode/timestamp/pair is unchanged')
    parser.add_argument('--report', type=Path, help='Optional JSON result destination')
    args = parser.parse_args()
    source = LeRobotSource(args.root)
    datasets = {s: LeRobotRoboMINDDataset(source, s)
                for s in ('train','validation','test')}
    check_disjoint(*datasets.values())
    report = {}
    for name, ds in datasets.items():
        old = None
        if args.legacy_root:
            old = RoboMINDDataset(args.legacy_root / f'{name}.jsonl', require_observed_endpoints=name!='train')
        for index in np.linspace(0, len(ds)-1, min(8,len(ds)), dtype=int):
            batch = lerobot_collate([ds[int(index)]])
            assert torch.isfinite(batch['states']).all()
            assert batch['videos'].shape == (1,2,3,256,256)
            assert batch['state_mask'][..., :14].all() and not batch['state_mask'][..., 14:].any()
            if old is not None:
                # Independent reference: exact previously audited MP4 timestamps,
                # followed by the pinned official max-aspect crop and v2 resize.
                i, a, b, _ = old.pairs[int(index)]
                with av.open(old.rows[i]['video_file']) as container:
                    stream = container.streams.video[0]
                    for endpoint, j in enumerate((a, b)):
                        target = float(old._arrays(i)['video_pts'][j])
                        container.seek(int(round(target / float(stream.time_base))), stream=stream, backward=True)
                        for frame in container.decode(stream):
                            if frame.pts is not None and abs(float(frame.pts * stream.time_base) - target) < 1e-5:
                                raw = torch.from_numpy(frame.to_ndarray(format='rgb24')).permute(2, 0, 1)
                                break
                        else:
                            raise AssertionError(f'Missing reference frame {target}')
                        h, w = raw.shape[-2:]
                        if w / h > 4 / 3:
                            cw = int(np.floor(h * (4 / 3))); left = (w - cw) // 2
                            raw = raw[..., left:left + cw].contiguous()
                        expected = v2.Resize((256, 256), interpolation=v2.InterpolationMode.BILINEAR, antialias=True)(raw)
                        torch.testing.assert_close(batch['videos'][0, endpoint], expected, rtol=0, atol=0)
        if old is not None:
            assert ds.episodes == old.episodes and ds.groups == old.groups and ds.pairs == old.pairs
            for i, row in enumerate(ds.rows):
                assert str(row['episode_index']) == old.rows[i]['episode_id']
                np.testing.assert_array_equal(ds._arrays(i)['timestamps'], old._arrays(i)['timestamps'])
                assert old._arrays(i)['original_valid'].all()
        report[name] = dict(pairs=len(ds), tail_pairs=int(sum(p[3] for p in ds.pairs)),
                            episodes=len(ds.episodes), groups=len(ds.groups))
    with tempfile.TemporaryDirectory(prefix='lawam-robomind-') as directory:
        temp = Path(directory)
        from train_stage1 import load_config
        config = load_config(ROOT/'configs/stage1_smoke.yaml')
        config['data']['root'] = str(args.root.resolve())
        (temp/'smoke.yaml').write_text(yaml.safe_dump(config))
        command = [sys.executable,str(ROOT/'train_stage1.py'),'--config',str(temp/'smoke.yaml'),
                   '--output',str(temp/'run')]
        subprocess.run(command,check=True,cwd=ROOT)
        checkpoint=temp/'run/checkpoints/final.ckpt'
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['global_step']==2 and saved['optimizer_states']
        assert saved['data_contract']['state_schema'] == datasets['train'].contract['state_schema']
        assert all(torch.isfinite(v).all() for v in saved['state_dict'].values() if v.is_floating_point())
        report['initial_training'] = json.loads((temp/'run/run_summary.json').read_text())
        del saved
        config=yaml.safe_load((temp/'run/resolved_config.yaml').read_text())
        config['trainer']['max_steps']=3
        (temp/'resume.yaml').write_text(yaml.safe_dump(config))
        command[command.index('--config')+1]=str(temp/'resume.yaml')
        subprocess.run(command+['--resume',str(checkpoint)],check=True,cwd=ROOT)
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['global_step']==3
        report['resumed_training'] = json.loads((temp/'run/run_summary.json').read_text())
        report['data_contract'] = saved['data_contract']
    report['smoke']='PASS: sampled loaders; real DINO; tiny IDM/LaWM; two steps, checkpoint, resume to step 3'
    report['all_legacy_pairs_and_timestamps_equal'] = bool(args.legacy_root)
    report['reference_video_endpoints_equal'] = 48 if args.legacy_root else None
    report['checkpoint_scope'] = 'Temporary smoke checkpoints deleted after verification; no full training launched.'
    report['source_sha256'] = {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() for p in
                             ('train_stage1.py', 'vendor/dataset/lerobot_robomind.py', 'tests/smoke_robomind.py')}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
