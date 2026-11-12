"""Reuse audited nested512 features/queries with immutable reference heads43/44."""
import json
from pathlib import Path
import shutil
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from torch import nn
from research.idm.cache_reference_queries import sha256
from research.idm.train_reference_heads import representation
from research.idm.cache_unused_episodes import state_hash


@torch.inference_mode()
def main():
    torch.set_num_threads(2);torch.set_float32_matmul_precision('medium')
    base=ROOT/'runs/reference-idm42-nested512-prior10000'
    bm=json.loads((base/'cache_manifest.json').read_text())
    assert sha256(base/'train.pt')==bm['cache_sha256']['train']
    audit=json.loads((ROOT/'research/idm/nested512-independent-verification.json').read_text())
    assert audit['all_checks_passed']
    data=torch.load(base/'train.pt',weights_only=True)
    assert len(data['rows'])==512
    added=torch.load(base/'additional_queries.pt',weights_only=True)
    assert added['rows']==data['rows'][128:]
    oldq=torch.load(ROOT/'research/idm/reference-query-full/train.pt',weights_only=True)
    assert oldq['rows']==data['rows'][:128]
    calpath=ROOT/'research/idm/reference-query-full/train_calibration.pt'
    cal=torch.load(calpath,weights_only=True)
    for seed in (43,44):
        parent=ROOT/f'runs/reference-idm{seed}-with-prior10000'
        dest=ROOT/f'runs/reference-idm{seed}-nested512-prior10000'
        if dest.exists():raise FileExistsError(dest)
        pm=json.loads((parent/'cache_manifest.json').read_text());ex=pm['export']
        assert sha256(calpath)==ex['calibration_sha256']
        headpath=Path(ex['head_checkpoint'])
        assert sha256(headpath)==ex['head_checkpoint_sha256']
        saved=torch.load(headpath,weights_only=True)
        assert saved['seed']==seed and saved['steps']==2000 and saved['variant']=='reference'
        assert state_hash(saved['state_dict'],'head.')==ex['head_state_sha256']
        head=nn.Sequential(nn.Linear(1024,256),nn.GELU(),nn.Linear(256,64)).eval()
        head.load_state_dict({k[5:]:v for k,v in saved['state_dict'].items() if k.startswith('head.')})
        assert sha256(parent/'train.pt')==pm['cache_sha256']['train']
        old=torch.load(parent/'train.pt',weights_only=True)
        for k in ('current','future','baseline'):
            assert torch.equal(data[k][:128],old[k])
        assert old['rows']==data['rows'][:128]
        reproduced=head(representation(oldq,cal,'reference')).chunk(2,-1)[0]
        assert torch.equal(reproduced,old['codes'])
        x=representation(added,cal,'reference')
        extra=torch.cat([head(chunk).chunk(2,-1)[0] for chunk in x.split(128)])
        exported=dict(data,codes=torch.cat((old['codes'],extra)))
        assert torch.isfinite(exported['codes']).all()
        assert sha256(parent/'validation.pt')==pm['cache_sha256']['validation']
        validation=torch.load(parent/'validation.pt',weights_only=True)
        baseval=torch.load(base/'validation.pt',weights_only=True)
        assert validation['rows']==baseval['rows']
        for k in ('current','future','baseline'):
            assert torch.equal(validation[k],baseval[k])
        dest.mkdir()
        torch.save(exported,dest/'train.pt')
        shutil.copyfile(parent/'validation.pt',dest/'validation.pt')
        report=dict(bm,cache_sha256={s:sha256(dest/f'{s}.pt') for s in ('train','validation')},
            export=ex,head_seed=seed,parent_code_cache=str(base),parent_code_cache_sha256=bm['cache_sha256'],
            nested_head_export=dict(seed=seed,old128_exact=True,new384_head_batch=128,
                validation_byte_hash_preserved=True,noncode_fields_preserved=True,
                original_head_parent=str(parent),original_head_parent_manifest_sha256=sha256(parent/'cache_manifest.json'),
                added_queries_sha256=sha256(base/'additional_queries.pt'),old_queries_sha256=sha256(ROOT/'research/idm/reference-query-full/train.pt'),
                script_sha256=sha256(Path(__file__)),precision='medium CPU2 B128'),
            provenance_note='export describes immutable head originally trained on128; nested_head_export describes this512-code export. Shared encoder2900/prior10000; not independent end-to-end seeds.')
        (dest/'cache_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print('Exported',seed,dest,flush=True)


if __name__=='__main__':main()
