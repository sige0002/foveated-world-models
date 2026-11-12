"""Export a completed frozen-query head's posterior means as replacement cache codes.

True future is used through Q(x,y): this is Stage1 posterior reconstruction,
not an executable current-only prior prediction.
"""
import argparse,json,hashlib
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
import torch
from research.idm.train_reference_heads import Model,representation
from research.train_cached_decoders import file_hash

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--queries',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output must be new')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('medium')
    manifest=json.loads((a.cache/'cache_manifest.json').read_text())
    query_manifest=json.loads((a.queries/'manifest.json').read_text())
    report=json.loads(a.comparison.read_text());saved=torch.load(a.checkpoint,weights_only=True)
    matches=[r for r in report['results'] if r['seed']==saved['seed'] and r['variant']==saved['variant']]
    if len(matches)!=1 or saved['steps']!=report['settings']['steps']:raise ValueError('Not a completed reported checkpoint')
    model=Model().eval();model.load_state_dict(saved['state_dict'],strict=True)
    cal=torch.load(a.queries/'train_calibration.pt',weights_only=True)
    def selected_hash(state,prefix):
        digest=hashlib.sha256()
        for key in sorted(state):
            if key.startswith(prefix):
                value=state[key].detach().cpu().contiguous()
                digest.update(key.encode());digest.update(str(value.dtype).encode());digest.update(str(tuple(value.shape)).encode());digest.update(value.numpy().tobytes())
        return digest.hexdigest()
    encoder_snapshot=torch.load(a.cache/'reference_weights.pt',weights_only=True,mmap=True)
    encoder_hash=selected_hash(encoder_snapshot['state_dict'],'lam.encoder.')
    del encoder_snapshot
    head_hash=selected_hash(saved['state_dict'],'head.')
    a.output.mkdir(parents=True);hashes={};verification={}
    for split in ('train','validation'):
        parent_hash=file_hash(a.cache/f'{split}.pt')
        if parent_hash!=manifest['cache_sha256'][split] or parent_hash!=query_manifest['parent_cache_sha256'][split]:raise ValueError('Parent hash mismatch')
        data=torch.load(a.cache/f'{split}.pt',weights_only=True);q=torch.load(a.queries/f'{split}.pt',weights_only=True)
        if q['rows']!=data['rows'] or not torch.equal(q['indices'],torch.arange(len(data['codes']))):raise ValueError('Row mismatch')
        with torch.inference_mode():mu,_=model.posterior(representation(q,cal,saved['variant']))
        if mu.shape!=data['codes'].shape or not torch.isfinite(mu).all():raise ValueError('Code shape/finite mismatch')
        exported=dict(data,codes=mu)
        torch.save(exported,a.output/f'{split}.pt');check=torch.load(a.output/f'{split}.pt',weights_only=True)
        for key in ('current','future'):assert torch.equal(check[key],data[key])
        assert check['rows']==data['rows']
        hashes[split]=file_hash(a.output/f'{split}.pt');verification[split]={'pairs':len(mu),'current_future_rows_exactly_preserved':True}
    out=dict(manifest,cache_sha256=hashes,code_source='learned frozen-query posterior mean: uses real future; NOT executable prior prediction',
        conditioning_mean='Gaussian posterior mu from frozen Q(current,true_future)-Q(current,current), train-only calibrated; for Stage1 posterior reconstruction, not executable prior' if saved['variant']=='reference' else 'Gaussian posterior mu from frozen true-future query and specified train-only calibration; not executable prior',
        export=dict(encoder_state_sha256=encoder_hash,head_state_sha256=head_hash,parent_cache=str(a.cache.resolve()),parent_cache_sha256=manifest['cache_sha256'],
        head_checkpoint=str(a.checkpoint.resolve()),head_checkpoint_sha256=file_hash(a.checkpoint),
        comparison_sha256=file_hash(a.comparison),query_manifest_sha256=file_hash(a.queries/'manifest.json'),
        calibration_sha256=file_hash(a.queries/'train_calibration.pt'),variant=saved['variant'],seed=saved['seed'],steps=saved['steps'],verification=verification))
    (a.output/'cache_manifest.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps(out['export'],indent=2))
if __name__=='__main__':main()
