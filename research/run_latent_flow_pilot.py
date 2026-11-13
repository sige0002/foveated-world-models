"""Small real-data comparison; exploratory, not a publication benchmark.

Run from the repository root. No model downloads. Uses the same small IDM and
split for all variants. An existing output directory is rejected to avoid mixing runs.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from train_stage1 import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--vision-model', required=True)
    parser.add_argument('--prepared-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--accelerator', choices=['cpu', 'gpu'], default='cpu')
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--val-batches', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--variants', nargs='+',
                        choices=['direct', 'residual', 'refinement', 'flow_current', 'flow_noise'],
                        default=['direct', 'residual', 'refinement', 'flow_current', 'flow_noise'])
    args = parser.parse_args()
    if args.steps < 1 or args.val_batches < 1:
        parser.error('steps and val-batches must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    results = {}
    variants = [('direct', 'direct', 'current', 0.1),
                ('residual', 'refinement', 'current', 0.1),
                ('refinement', 'refinement', 'current', 0.1),
                ('flow_current', 'flow', 'current', 0.1),
                ('flow_noise', 'flow', 'noise', 1.0)]
    for name, decoder_type, source, noise in variants:
        if name not in args.variants:
            continue
        config = load_config(ROOT/'configs/stage1_smoke.yaml')
        config['seed'] = args.seed
        config['model'].update(vision_model_id=str(Path(args.vision_model).resolve()),
                               decoder_type=decoder_type, flow_source=source,
                               flow_noise_scale=noise, flow_steps=1 if name == 'residual' else 4,
                               image_aug=False, dual_view_aug=False)
        # This historical pilot explicitly consumes prepared NPZs; never silently
        # switch its matched comparisons when the default Stage-1 loader changes.
        config['data'].update(format='robomind_video',
                              train_manifest=str((args.prepared_root/'train.jsonl').resolve()),
                              val_manifest=str((args.prepared_root/'validation.jsonl').resolve()))
        config['trainer'].update(accelerator=args.accelerator, max_steps=args.steps,
                                 val_check_interval=args.steps, num_sanity_val_steps=0,
                                 limit_val_batches=args.val_batches)
        path = args.output/f'{name}.yaml'
        path.write_text(yaml.safe_dump(config))
        env = dict(os.environ, OMP_NUM_THREADS='4')
        if args.accelerator == 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = ''
        with (args.output/f'{name}.log').open('w') as log:
            subprocess.run([sys.executable, str(ROOT/'train_stage1.py'), '--config', str(path.resolve()),
                            '--output', str((args.output/name).resolve())],
                           cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
        results[name] = json.loads((args.output/name/'run_summary.json').read_text())
        (args.output/'comparison.json').write_text(json.dumps(results, indent=2))
        print(name, json.dumps(results[name]['metrics']), flush=True)


if __name__ == '__main__':
    main()
