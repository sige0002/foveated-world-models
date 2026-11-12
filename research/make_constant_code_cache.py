"""Replace only codes with a verified pre-existing train-constant vector."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):
            h.update(block)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error('Output must be new')
    torch.set_num_threads(1)
    meta = json.loads((a.cache/'cache_manifest.json').read_text())
    ref = json.loads((a.reference/'cache_manifest.json').read_text())
    ref_path = a.reference/'train.pt'
    assert sha(ref_path) == ref['cache_sha256']['train']
    rc = torch.load(ref_path,weights_only=True)['codes']
    vector = rc[:1].clone()
    assert torch.isfinite(vector).all() and torch.equal(rc,vector.expand_as(rc))
    a.output.mkdir()
    hashes = {}
    for split in ('train','validation'):
        src = a.cache/f'{split}.pt'
        assert sha(src) == meta['cache_sha256'][split]
        data = torch.load(src,weights_only=True)
        assert data['codes'].shape[1:] == vector.shape[1:]
        output = dict(data,codes=vector.expand_as(data['codes']).clone())
        dst = a.output/f'{split}.pt'
        torch.save(output,dst)
        check = torch.load(dst,weights_only=True)
        for k,v in data.items():
            if k != 'codes':
                assert torch.equal(check[k],v) if isinstance(v,torch.Tensor) else check[k] == v
        assert torch.equal(check['codes'],vector.expand_as(check['codes']))
        hashes[split] = sha(dst)
    manifest = dict(meta,cache_sha256=hashes,
                    parent_code_cache=str(a.cache),parent_code_cache_sha256=meta['cache_sha256'],
                    code_source='Same fixed constant vector in every train and validation row; no sample-specific future code',
                    conditioning_mean='Constant code control; no posterior conditioning',
                    constant_source=str(a.reference),constant_source_train_sha256=sha(ref_path),
                    constant_script_sha256=sha(Path(__file__)),
                    noncode_fields_exactly_preserved=True)
    if 'export' in manifest:
        manifest['parent_code_export'] = manifest.pop('export')
    (a.output/'cache_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(a.output,hashes)


if __name__ == '__main__':
    main()
