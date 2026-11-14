"""Bounded frozen B1 x two-pair runtime test, never the final512 evaluation."""
import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from research import evaluate_robotwin_loss_ablation_fp32 as evaluator


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--reference-rows', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.resolve().is_relative_to(evaluator.SSD_RUNS.resolve())
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    cfg = yaml.safe_load((args.run / 'resolved_config.yaml').read_text())
    request = yaml.safe_load((args.run / 'requested_config.yaml').read_text())
    contract = evaluator.read_json(args.run / 'data_contract.json')
    os.environ.update(cfg['environment'])
    sys.path[:0] = [request['official_source'], *request['dependency_roots']]
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    assert torch.cuda.is_available()
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    free_cuda, _ = torch.cuda.mem_get_info()
    if available < 16 * 1024**3 or free_cuda < 8 * 1024**3:
        raise RuntimeError('Not enough free memory for the bounded smoke; no model loaded')
    # Overrides are local to this test subprocess. Production evaluator still
    # requires step3000 and512pairs; it is not edited or invoked via its CLI.
    evaluator.FINAL_STEP = 10000
    evaluator.SAMPLES = 2
    selected = evaluator.read_json(args.reference_rows)['rows'][:2]
    assert selected[0]['episode'] != selected[1]['episode']
    reference = [{'episode': row['episode'], 'base': row['base'], 'swap_valid': True,
                  'donor_episode': selected[1 - i]['episode'], 'donor_base': selected[1 - i]['base']}
                 for i, row in enumerate(selected)]
    release = evaluator.SSD_RUNS.parent / 'models/lawam-lam-bd993da'
    release_cfg = yaml.safe_load((release / 'dino_large_vae.yaml').read_text())['model']
    reports = {}
    started = time.monotonic()
    cache = None
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for name in ('local10000', 'official'):
            official = name == 'official'
            mc = deepcopy(release_cfg if official else cfg['model'])
            mc['vision_model_id'] = cfg['model']['vision_model_id']
            checkpoint = release / 'checkpoints/pytorch_model.pt' if official else args.run / 'checkpoints/last.ckpt'
            model, evidence = evaluator.load_model(checkpoint, mc, contract, official=official)
            before = evaluator.module_hashes(model)
            model.cuda()
            if cache is None:
                cache = evaluator.build_cache(model, cfg, contract, reference, [1, 0],
                    args.output / 'two-pair-smoke-cache.pt', before['vision_encoder'], 1, args.run)
            evaluator.validate_cache(cache, reference, contract, before['vision_encoder'])
            parity = evaluator.native_route_parity(model, cache, official=official)
            result = evaluator.evaluate_model(model, cache, 1, args.output / name, official=official)
            after = evaluator.module_hashes(model)
            assert before == after
            reports[name] = {**evidence, **result, 'before': before, 'after': after,
                             'unchanged': True, 'native_route_parity': parity}
            print(json.dumps({'phase': 'smoke_model_completed', 'model': name,
                              'loaded_tensors': evidence['strictly_loaded_state_tensors'],
                              'parity': parity, 'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated()}), flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
    evaluator.write_json(args.output / 'results.json', {
        'complete': True, 'scope': 'Runtime smoke only; two pairs, B1, original10000 and official; no optimizer/update. Production step3000/512 constants untouched.',
        'evaluator_sha256': evaluator.digest(evaluator.__file__),
        'smoke_script_sha256': evaluator.digest(__file__), 'reports': reports,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'elapsed_seconds': time.monotonic() - started})
    print(json.dumps({'complete': True, 'output': str(args.output / 'results.json')}), flush=True)


if __name__ == '__main__':
    main()
