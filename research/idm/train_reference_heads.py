"""Matched frozen-query Gaussian heads and residual decoders; CPU only."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.train_cached_decoders import make_decoder,file_hash

class Model(nn.Module):
    def __init__(self):
        super().__init__();self.head=nn.Sequential(nn.Linear(1024,256),nn.GELU(),nn.Linear(256,64))
        self.decoder=make_decoder('residual',128,2,768,32,16)
    def posterior(self,x):
        mu,lv=self.head(x).chunk(2,-1);return mu,lv.clamp(-10,10)

def weight_hash(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items():h.update(k.encode());h.update(v.detach().numpy().tobytes())
    return h.hexdigest()

def representation(data,cal,kind):
    q=data['query']
    if kind=='raw':return q/cal['scales']['raw']
    if kind=='centered':return (q-cal['center'])/cal['scales']['centered']
    if kind=='reference':return (q-data['reference'])/cal['scales']['reference']
    return (cal['center']/cal['scales']['raw']).expand_as(q)

@torch.inference_mode()
def evaluate(model,data,x,batch=8,future_x=None):
    model.eval();rows=[];gen=torch.Generator().manual_seed(1407)
    mus,lvs=model.posterior(x)
    for start in range(0,len(x),batch):
        sl=slice(start,start+batch);h,y=data['current'][sl],data['future'][sl]
        mu,lv=mus[sl],lvs[sl];j=(torch.arange(start,min(start+batch,len(x)))+1)%len(x)
        eps=torch.randn(mu.shape,generator=gen)
        z=mu+(.5*lv).exp()*eps;wrong=mus[j]+(.5*lvs[j]).exp()*eps
        mean_pred=model.decoder(h,mu);sample=model.decoder(h,z);sw=model.decoder(h,wrong)
        values={'mean_mse':(mean_pred-y).square().flatten(1).mean(1),
                'sample_mse':(sample-y).square().flatten(1).mean(1),
                'swapped_mse':(sw-y).square().flatten(1).mean(1),
                'swap_prediction_change':(sw-sample).square().flatten(1).mean(1)}
        for i in range(len(h)):rows.append(dict(data['rows'][start+i],**{k:float(v[i]) for k,v in values.items()}))
    metrics={k:sum(r[k] for r in rows)/len(rows) for k in values}
    metrics.update(mu_variance=float(mus.var(0,unbiased=False).sum()),logvar_variance=float(lvs.var(0,unbiased=False).sum()))
    if future_x is not None:
        fm,fl=model.posterior(future_x)
        m,l,fm,fl=[v.double() for v in (mus,lvs,fm,fl)]
        def kl(a,b,c,d):
            delta=b-d
            return .5*(delta.expm1()-delta+(a-c).square()*(-d).exp()).sum(-1)
        metrics['future_swapped_symmetric_posterior_kl']=float((kl(m,l,fm,fl)+kl(fm,fl,m,l)).mean()/2)
    return dict(metrics=metrics,rows=rows)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--queries',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--steps',type=int,default=400)
    p.add_argument('--seeds',type=int,nargs='+',default=[42]);p.add_argument('--beta',type=float,default=5e-5)
    a=p.parse_args()
    if a.output.exists() or a.steps<1:p.error('New output and positive steps required')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('medium')
    meta=json.loads((a.queries/'manifest.json').read_text())
    data={s:torch.load(a.cache/f'{s}.pt',weights_only=True) for s in ('train','validation')}
    queries={s:torch.load(a.queries/f'{s}.pt',weights_only=True) for s in data}
    for s in data:
        assert file_hash(a.cache/f'{s}.pt')==meta['parent_cache_sha256'][s]
        assert torch.equal(queries[s]['indices'],torch.arange(len(data[s]['current'])))
        assert queries[s]['rows']==data[s]['rows']
    assert not {r['episode'] for r in data['train']['rows']} & {r['episode'] for r in data['validation']['rows']}
    cal=torch.load(a.queries/'train_calibration.pt',weights_only=True)
    a.output.mkdir(parents=True)
    report=dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},query_manifest=meta,
        objective='MSE + beta * mean(sum_dimension KL(q||N(0,I))). beta matches Stage1 coefficient; MSE differs from production SmoothL1. Frozen encoder, small fresh decoder; not production quality.',results=[])
    for seed in a.seeds:
        init_hash=None;batch_hash=None;eps_hash=None
        for kind in ('raw','centered','reference','constant'):
            torch.manual_seed(seed);model=Model();ih=weight_hash(model)
            if init_hash is None:init_hash=ih
            assert ih==init_hash
            opt=torch.optim.AdamW([dict(params=[v for v in model.parameters() if v.ndim>=2],weight_decay=.01),
                                  dict(params=[v for v in model.parameters() if v.ndim<2],weight_decay=0.)],lr=3e-4)
            rng=torch.Generator().manual_seed(seed+1000);noise=torch.Generator().manual_seed(seed+2000)
            bh=hashlib.sha256();eh=hashlib.sha256();history=[];started=time.perf_counter()
            x={s:representation(queries[s],cal,kind) for s in data}
            for step in range(1,a.steps+1):
                model.train();idx=torch.randint(len(x['train']),(8,),generator=rng);bh.update(idx.numpy().tobytes())
                mu,lv=model.posterior(x['train'][idx]);eps=torch.randn(mu.shape,generator=noise);eh.update(eps.numpy().tobytes())
                pred=model.decoder(data['train']['current'][idx],mu+(.5*lv).exp()*eps)
                mse=F.mse_loss(pred,data['train']['future'][idx]);kl=.5*(mu.square()+lv.exp()-1-lv).sum(-1).mean()
                loss=mse+a.beta*kl
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
                opt.zero_grad(set_to_none=True);loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                if not torch.isfinite(norm):raise RuntimeError('Nonfinite gradients')
                opt.step()
                if step%100==0 or step==a.steps:
                    history.append(dict(step=step,mse=float(mse.detach()),kl=float(kl.detach())))
                    print(seed,kind,history[-1],flush=True)
            if batch_hash is None:batch_hash=bh.hexdigest();eps_hash=eh.hexdigest()
            assert bh.hexdigest()==batch_hash and eh.hexdigest()==eps_hash
            changed=dict(queries['validation'],query=queries['validation']['future_swapped'])
            future_x=representation(changed,cal,kind)
            val=evaluate(model,data['validation'],x['validation'],future_x=future_x);train=evaluate(model,data['train'],x['train'])
            result=dict(seed=seed,variant=kind,initial_weight_sha256=ih,minibatch_sha256=bh.hexdigest(),
                noise_sha256=eh.hexdigest(),history=history,validation=val,train_metrics=train['metrics'],
                parameters=sum(v.numel() for v in model.parameters()),seconds=time.perf_counter()-started)
            report['results'].append(result)
            torch.save(dict(state_dict=model.state_dict(),seed=seed,variant=kind,steps=a.steps),a.output/f'{kind}-{seed}.pt')
            (a.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print('RESULT',seed,kind,val['metrics'],flush=True)
if __name__=='__main__':main()
