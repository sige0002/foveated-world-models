"""All-arm real-data initialization audit; only an explicitly isolated two-step toy continuation."""
import copy
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from research.train_bounded_innovation import load,preflight,write
from research.bounded_innovation import InnovationModel,kl_summary
from research.idm.train_local_repair import state_hash
from research.train_cached_decoders import file_hash


def main():
    source=Path(sys.argv[1]);output=Path(sys.argv[2])
    if output.exists():raise RuntimeError('New output required')
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(42)
    data,decoder,inputs,cal,T,hashes=load(source)
    decoder.eval();old={}
    with torch.no_grad():
        for s,d in data.items():
            old[s]=torch.cat([decoder(d['current'][i:i+8],d['codes'][i:i+8],baseline=d['baseline'][i:i+8]) for i in range(0,len(d['rows']),8)])
    results={};init={};isolated={}
    for seed in [42,43,44]:
        for arm in ['future','recoding','noise']:
            m=InnovationModel(copy.deepcopy(decoder),arm,seed).eval()
            x=inputs['future' if arm=='future' else 'recoding']
            if seed==42:
                r=preflight(m,data,x,old);r['comparison_rows']='all512/64/96'
            else:
                tiny={s:{k:(v[:8] if isinstance(v,(torch.Tensor,list)) else v) for k,v in d.items()} for s,d in data.items()}
                r=preflight(m,tiny,{s:v[:8] for s,v in x.items()},{s:v[:8] for s,v in old.items()})
                r['comparison_rows']='first8 per split; KL below covers all rows'
            with torch.no_grad():r['all_row_kl']={s:kl_summary(m.mean(v)) for s,v in x.items()}
            r['head_hash']=state_hash(m.head);r['decoder_hash']=state_hash(m.decoder)
            results[f'{arm}{seed}']=r
            if seed==42 and arm=='future':
                # Copy is never saved as a fit start. Source/preflight model unchanged.
                anchor=state_hash(m);test=copy.deepcopy(m)
                opt=torch.optim.SGD(test.parameters(),lr=.01)
                d=data['train'];sl=slice(0,8);checks=[]
                for step in [1,2]:
                    opt.zero_grad()
                    pred,mu=test(d['current'][sl],d['codes'][sl],x['train'][sl],torch.zeros(8,1,8),baseline=d['baseline'][sl])
                    (pred-d['future'][sl]).square().mean().backward()
                    h=sum(float(v.grad.square().sum()) for v in test.head.parameters() if v.grad is not None)**.5
                    c=float(test.decoder.action_proj.extra.weight.grad.norm())
                    checks.append({'step':step,'head_gradient_norm':h,'extra_column_gradient_norm':c})
                    if step==1:
                        assert c>0;opt.step()
                    else:assert h>0
                assert state_hash(m)==anchor
                isolated={'checks':checks,'epsilon':'zero to remove noise-correlation explanation',
                          'private_copy_only':True,'source_initial_state_unchanged':True,
                          'note':'One private SGD update and two backward checks; not a400-step fit or its initial weights.'}
            print(seed,arm,'initial PASS',flush=True)
    for seed in [42,43,44]:
        assert len({results[f'{a}{seed}']['head_hash'] for a in ['future','recoding','noise']})==1
    assert len({r['decoder_hash'] for r in results.values()})==1
    for p in [ROOT/'research/bounded_innovation.py',ROOT/'research/train_bounded_innovation.py',Path(__file__)]:hashes[str(p)]=file_hash(p)
    write(output,{'results':results,'isolated_gradient_audit':isolated,'source_sha256':hashes,
                  'same_seed_head_exact':True,'all_decoder_initial_exact':True,'full_training_launched':False})

if __name__=='__main__':main()
