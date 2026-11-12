"""Compare production and adapted heads on identical cached Q; no encoder/decoder run."""
import copy,hashlib,json,sys
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.train_reference_heads import representation
from research.train_cached_decoders import file_hash

def metrics(outputs):
    ref=outputs['float64'];result={}
    for mode,(mu,lv,other_mu,other_lv) in outputs.items():
        m,l,om,ol=[t.double() for t in (mu,lv,other_mu,other_lv)]
        variance=float(m.var(0,unbiased=False).sum());error=float((m-ref[0]).square().sum(-1).mean())
        result[mode]=dict(mu_between_example_variance=variance,future_swap_mu_energy=float((om-m).square().sum(-1).mean()),
            future_swap_logvar_energy=float((ol-l).square().sum(-1).mean()),
            mu_max_abs_vs_float64=float((m-ref[0]).abs().max()),mu_error_energy_vs_float64=error,
            mu_error_energy_over_float64_variance=error/max(float(ref[0].var(0,unbiased=False).sum()),1e-30))
    return result

@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    qpath=ROOT/'research/idm/reference-query-full/validation.pt';q=torch.load(qpath,weights_only=True)
    calpath=ROOT/'research/idm/reference-query-full/train_calibration.pt';cal=torch.load(calpath,weights_only=True)
    weights_path=ROOT/'runs/frozen-idm-step2900/reference_weights.pt'
    saved=torch.load(weights_path,weights_only=True,mmap=True)['state_dict']
    keys=['lam.encoder.out_proj.weight','lam.encoder.out_proj.bias','lam.vq.pre_norm.weight','lam.vq.pre_norm.bias','lam.vq.mu.weight','lam.vq.mu.bias','lam.vq.logvar.weight','lam.vq.logvar.bias']
    values=[saved[k].clone() for k in keys];del saved
    digest=hashlib.sha256()
    for k,v in zip(keys,values):digest.update(k.encode());digest.update(v.numpy().tobytes())
    outputs={}
    for mode,dtype in [('medium',torch.float32),('highest',torch.float32),('float64',torch.float64)]:
        torch.set_float32_matmul_precision('medium' if mode=='medium' else 'highest')
        W,b,g,bn,M,bm,V,bv=[v.to(dtype) for v in values]
        def production(x):
            n=F.layer_norm(F.linear(x,W,b),(32,),g,bn,1e-5)
            return F.linear(n,M,bm),F.linear(n,V,bv).clamp(-10,10)
        # Match original extraction's B=1 numerical route.
        ms,ls,oms,ols=[],[],[],[]
        for i in range(len(q['query'])):
            m,l=production(q['query'][i:i+1].to(dtype));om,ol=production(q['future_swapped'][i:i+1].to(dtype))
            ms.append(m);ls.append(l);oms.append(om);ols.append(ol)
        outputs[mode]=tuple(torch.cat(v).double() for v in (ms,ls,oms,ols))
    cached=torch.load(ROOT/'runs/frozen-idm-step2900/validation.pt',weights_only=True)['codes']
    repro=float((outputs['medium'][0]-cached.double()).abs().max());assert repro==0
    result=dict(production=metrics(outputs),production_medium_mu_reproduction_max_error=repro,adapted=[],
        provenance=dict(query_sha256=file_hash(qpath),calibration_sha256=file_hash(calpath),production_head_tensors_sha256=digest.hexdigest(),
            script_sha256=file_hash(Path(__file__)),torch_version=torch.__version__,threads=4,device='CPU',production_batch_size=1,adapted_batch_size=64),
        limitation='Only saved step2900 Q and head computations. Q already contains numerical effects of its original encoder evaluation. No full-encoder precision change, decoder evaluation, training, or step6501 Q analysis performed.')
    for seed in (42,43,44):
        for variant in ('raw','centered','reference'):
            path=ROOT/f'research/idm/reference-comparison2000/{variant}-{seed}.pt'
            s=torch.load(path,weights_only=True)['state_dict']
            head=nn.Sequential(nn.Linear(1024,256),nn.GELU(),nn.Linear(256,64)).eval()
            head.load_state_dict({k[len('head.'):]:v for k,v in s.items() if k.startswith('head.')})
            x=representation(q,cal,variant);changed=representation(dict(q,query=q['future_swapped']),cal,variant)
            out={}
            for mode,dtype in [('medium',torch.float32),('highest',torch.float32),('float64',torch.float64)]:
                torch.set_float32_matmul_precision('medium' if mode=='medium' else 'highest')
                h=copy.deepcopy(head).to(dtype)
                m,l=h(x.to(dtype)).chunk(2,-1);om,ol=h(changed.to(dtype)).chunk(2,-1)
                out[mode]=(m.double(),l.clamp(-10,10).double(),om.double(),ol.clamp(-10,10).double())
            result['adapted'].append(dict(seed=seed,variant=variant,checkpoint_sha256=file_hash(path),metrics=metrics(out)))
    path=ROOT/'research/idm/head-precision-audit-step2900.json'
    if path.exists():raise FileExistsError(path)
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
