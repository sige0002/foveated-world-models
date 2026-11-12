"""Evaluate fixed correction checkpoints without fitting on research-unused data."""
import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from research.train_cached_decoders import evaluate, file_hash, make_decoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--constant-checkpoints', type=Path, nargs='*', default=[])
    parser.add_argument('--constant-cache', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--expected-steps', type=int, default=400,
                        help='Prespecified fixed endpoint; defaults to the original400 protocol')
    parser.add_argument('--zero-noise', action='store_true',
                        help='Exploratory deterministic inference on fixed stochastic weights')
    args = parser.parse_args()
    if args.expected_steps < 1:
        parser.error('Expected steps must be positive')
    if args.output.exists():
        parser.error('Output must be new')
    torch.set_num_threads(4)
    manifest = json.loads((args.cache/'cache_manifest.json').read_text())
    data_path = args.cache/'validation.pt'
    if file_hash(data_path) != manifest['cache_sha256']['validation']:
        raise ValueError('Evaluation cache hash mismatch')
    data = torch.load(data_path, weights_only=True)
    if len({r['episode'] for r in data['rows']}) != len(data['rows']):
        raise ValueError('Expected one pair per episode')
    constant_data = None
    if args.constant_checkpoints:
        if args.constant_cache is None:
            parser.error('Constant checkpoints require their training cache')
        cm = json.loads((args.constant_cache/'cache_manifest.json').read_text())
        cp = args.constant_cache/'train.pt'
        if file_hash(cp) != cm['cache_sha256']['train']:
            raise ValueError('Constant training cache hash mismatch')
        codes = torch.load(cp, weights_only=True)['codes']
        if not torch.equal(codes, codes[:1].expand_as(codes)):
            raise ValueError('Control training codes must all be identical')
        constant_data = dict(data, codes=codes[:1].expand_as(data['codes']).clone())
    results = []
    for path in args.checkpoints + args.constant_checkpoints:
        saved = torch.load(path, weights_only=True)
        if saved['steps'] != args.expected_steps:
            raise ValueError(f'Expected prespecified final {args.expected_steps}-step model')
        model = make_decoder(saved['variant'], saved['dim'], saved['layers'],
                             data['current'].shape[-1], data['codes'].shape[-1],
                             math.isqrt(data['current'].shape[-2]))
        model.load_state_dict(saved['state_dict'], strict=True)
        if args.zero_noise:
            if not hasattr(model, 'stochastic'):
                raise ValueError('Zero-noise evaluation requires an explicit stochastic flag')
            model.stochastic = False
        control = path in args.constant_checkpoints
        evaluation = evaluate(model, constant_data if control else data)
        result = dict(checkpoint=str(path), checkpoint_sha256=file_hash(path),
                      variant=saved['variant'], seed=saved['seed'], steps=saved['steps'],
                      constant_code_control=control, **evaluation)
        results.append(result)
        print(path, evaluation['metrics'], flush=True)
    report = dict(cache=str(args.cache), cache_manifest=manifest,
                  inference_mode='zero-noise deterministic; not guaranteed conditional mean' if args.zero_noise else 'standard',
                  evaluation_scope='Cached evaluation; see cache_manifest for split provenance. True-future posterior codes.',
                  checkpoint_selection=f'Fixed final step {args.expected_steps}, no evaluation fitting or checkpoint selection',
                  script_sha256=file_hash(Path(__file__)), results=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
