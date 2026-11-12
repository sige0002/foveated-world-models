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
    p.add_argument('--split-map', nargs='+', default=['train=train', 'validation=validation'],
                   help='code split=prior split; output keeps the code split name')
    a = p.parse_args()
    if a.output.exists():
        p.error('Output must be new')
    mappings = [item.split('=', 1) for item in a.split_map]
    allowed = {'train', 'validation', 'observed96'}
    if (any(len(item) != 2 or not set(item) <= allowed for item in mappings)
            or len({item[0] for item in mappings}) != len(mappings)):
        p.error('Expected unique split mappings using train, validation or observed96')
    torch.set_num_threads(1)
    cm = json.loads((a.codes/'cache_manifest.json').read_text())
    pm = json.loads((a.prior/'cache_manifest.json').read_text())
    data, prior_metrics = {}, {}
    for split, prior_split in mappings:
        for root, meta, name in ((a.codes, cm, split), (a.prior, pm, prior_split)):
            if file_hash(root/f'{name}.pt') != meta['cache_sha256'][name]:
                raise ValueError('Input cache hash mismatch')
        codes = torch.load(a.codes/f'{split}.pt', weights_only=True)
        prior = torch.load(a.prior/f'{prior_split}.pt', weights_only=True)
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
    # These fields described the old baseline, not the copied IDM codes.
    out.pop('decoder_precision_alignment', None)
    out['baseline_source'] = f"{out['prior_source']}; checkpoint step {out['prior_checkpoint_step']}"
    out['row_metadata_note'] = ('Source rows are preserved; historical *_mse fields may concern older '
                                'checkpoints. Use the newly computed prior_metrics for this baseline.')
    out['baseline_reuse'] = dict(parent=str(a.prior), parent_cache_sha256=pm['cache_sha256'],
                                 parent_manifest_sha256=file_hash(a.prior/'cache_manifest.json'),
                                 script_sha256=file_hash(Path(__file__)),
                                 split_map=dict(mappings),
                                 features_and_rows_exactly_equal=True)
    (a.output/'cache_manifest.json').write_text(json.dumps(out, indent=2)+'\n')
    print(a.output, hashes)


if __name__ == '__main__':
    main()
