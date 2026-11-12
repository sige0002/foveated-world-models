"""Reuse an immutable current-only prior for exactly matching feature pairs."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from research.train_cached_decoders import file_hash


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--codes', type=Path, required=True)
    p.add_argument('--prior', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output must be new')
    torch.set_num_threads(1)
    cm = json.loads((a.codes/'cache_manifest.json').read_text())
    pm = json.loads((a.prior/'cache_manifest.json').read_text())
    data, prior_metrics = {}, {}
    for split in ('train', 'validation'):
        for root, meta in ((a.codes, cm), (a.prior, pm)):
            if file_hash(root/f'{split}.pt') != meta['cache_sha256'][split]:
                raise ValueError('Input cache hash mismatch')
        codes = torch.load(a.codes/f'{split}.pt', weights_only=True)
        prior = torch.load(a.prior/f'{split}.pt', weights_only=True)
        if codes['rows'] != prior['rows']:
            raise ValueError('Pair identities or metadata differ')
        for key in ('current', 'future'):
            if not torch.equal(codes[key], prior[key]):
                raise ValueError('Features differ; cached prior is not reusable')
        if prior['baseline'].shape != codes['current'].shape or not torch.isfinite(prior['baseline']).all():
            raise ValueError('Invalid prior')
        data[split] = dict(codes, baseline=prior['baseline'])
        errors = (prior['baseline']-codes['future']).square().flatten(1).mean(1)
        prior_metrics[split] = dict(mse=float(errors.mean()), per_example=errors.tolist())
    a.output.mkdir(parents=True)
    hashes = {}
    for split, values in data.items():
        torch.save(values, a.output/f'{split}.pt')
        hashes[split] = file_hash(a.output/f'{split}.pt')
    out = dict(cm, cache_sha256=hashes, parent_code_cache=str(a.codes),
               parent_code_cache_sha256=cm['cache_sha256'])
    if 'prior_checkpoint_step' in pm:
        for key in ('prior_checkpoint_step', 'prior_checkpoint_sha256', 'prior_weights_sha256', 'prior_source'):
            out[key] = pm[key]
    else:
        if 'baseline=corresponding decoder(current,zero code)' not in pm.get('conditioning_mean', ''):
            raise ValueError('Original-model audit must explicitly identify its zero-code baseline')
        out.update(prior_checkpoint_step=pm['checkpoint_step'],
                   prior_checkpoint_sha256=pm['named_checkpoint_sha256'],
                   prior_weights_sha256=pm['snapshot_sha256'],
                   prior_source='Audited frozen decoder(current,zero code); no future input')
    out['prior_metrics'] = prior_metrics
    out['baseline_reuse'] = dict(parent=str(a.prior), parent_cache_sha256=pm['cache_sha256'],
                                 parent_manifest_sha256=file_hash(a.prior/'cache_manifest.json'),
                                 script_sha256=file_hash(Path(__file__)),
                                 features_and_rows_exactly_equal=True)
    (a.output/'cache_manifest.json').write_text(json.dumps(out, indent=2)+'\n')
    print(a.output, hashes)


if __name__ == '__main__':
    main()
