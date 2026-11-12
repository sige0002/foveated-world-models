"""Cache a frozen decoder's current-only prior alongside an immutable IDM-code cache."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor/lawam')]
import torch
import yaml
from research.train_cached_decoders import file_hash
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2


@torch.inference_mode()
def predict_prior(model, current, code_dim, batch_size=2):
    # No target, posterior code, or future-derived quantity is an input here.
    values = []
    for h in current.split(batch_size):
        z = h.new_zeros((len(h), 1, code_dim))
        values.append(model(h, z))
    return torch.cat(values)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codes', type=Path, required=True)
    p.add_argument('--prior-cache', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        p.error('New output directory required')
    torch.set_num_threads(4)
    code_meta = json.loads((args.codes / 'cache_manifest.json').read_text())
    prior_meta = json.loads((args.prior_cache / 'cache_manifest.json').read_text())
    config = yaml.safe_load((args.prior_cache / 'resolved_config.yaml').read_text())['model']
    weights_path = args.prior_cache / 'reference_weights.pt'
    weights = torch.load(weights_path, weights_only=True, mmap=True)
    assert weights['global_step'] == prior_meta['checkpoint_step']
    args.output.mkdir(parents=True)
    metrics, hashes = {}, {}
    model = None
    for split in ('train', 'validation'):
        for root, meta in ((args.codes, code_meta), (args.prior_cache, prior_meta)):
            assert file_hash(root / f'{split}.pt') == meta['cache_sha256'][split]
        data = torch.load(args.codes / f'{split}.pt', weights_only=True)
        prior_data = torch.load(args.prior_cache / f'{split}.pt', weights_only=True)
        key = lambda d: [(r['episode'], r['index'], r['dt']) for r in d['rows']]
        assert key(data) == key(prior_data)
        for name in ('current', 'future'):
            assert torch.equal(data[name], prior_data[name])
        if model is None:
            grid = math.isqrt(data['current'].shape[-2])
            model = LAMDecoder_v2(context_dim=config['dim'], input_dim=data['current'].shape[-1],
                code_dim=config['code_dim'], num_queries=config['num_queries'],
                num_layers=config['dec_layers'], num_heads=config['num_heads'], dropout=0.,
                grid_hw=(grid, grid), ffn_expansion_factor=config['ffn_expansion_factor'],
                last_ln=config.get('decoder_last_ln', True)).eval().requires_grad_(False)
            prefix = 'lam.decoder.'
            model.load_state_dict({k[len(prefix):]:v for k,v in weights['state_dict'].items()
                                   if k.startswith(prefix)}, strict=True)
            del weights
        data['baseline'] = predict_prior(model, data['current'], config['code_dim'])
        assert torch.isfinite(data['baseline']).all()
        errors = (data['baseline'] - data['future']).square().flatten(1).mean(1)
        metrics[split] = {'mse': errors.mean().item(), 'per_example': errors.tolist()}
        if split == 'validation':
            expected = torch.tensor([r['reference_zero_z_mse'] for r in prior_data['rows']])
            assert (errors - expected).abs().max() < 1e-4
        path = args.output / f'{split}.pt'
        torch.save(data, path)
        hashes[split] = file_hash(path)
        print(split, metrics[split]['mse'], flush=True)
    report = dict(code_meta, cache_sha256=hashes, parent_code_cache=str(args.codes),
                  parent_code_cache_sha256=code_meta['cache_sha256'],
                  prior_checkpoint_step=prior_meta['checkpoint_step'],
                  prior_checkpoint_sha256=prior_meta['checkpoint_sha256'],
                  prior_weights_sha256=file_hash(weights_path),
                  prior_source='Frozen decoder(current, zero-code); no true future or posterior input',
                  prior_metrics=metrics, extraction_code_sha256=file_hash(Path(__file__)))
    (args.output / 'cache_manifest.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
