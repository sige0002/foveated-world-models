"""Prepared A/B/C experiment; never launches training merely by importing.

A: direct MSE + KL. B: add flow loss through shared Gaussian head.
C: same as B, detach only the latent at the flow branch input.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.train_cached_decoders import make_decoder,file_hash
from research.idm.train_reference_heads import representation,weight_hash,evaluate

class JointModel(nn.Module):
    def __init__(self,query_dim=1024,code_dim=32,feature_dim=768,dim=128,layers=2,grid=16,decoder_family="wide"):
        super().__init__()
        self.head=nn.Sequential(nn.Linear(query_dim,256),nn.GELU(),nn.Linear(256,2*code_dim))
        # All arms allocate both branches to preserve identical initialization.
        if decoder_family not in ('wide','narrow'):raise ValueError('Unknown decoder family')
        self.decoder_family=decoder_family
        direct_kind,flow_kind=('wide_residual','wide_flow') if decoder_family=='wide' else ('residual','flow')
        self.decoder=make_decoder(direct_kind,dim,layers,feature_dim,code_dim,grid)
        self.flow=make_decoder(flow_kind,dim,layers,feature_dim,code_dim,grid)
    def posterior(self,x):
        mu,lv=self.head(x).chunk(2,-1);return mu,lv.clamp(-10,10)
    def losses(self,query,current,target,eps,flow_noise,flow_time,arm):
        if arm not in ('A','B','C'):raise ValueError('Unknown arm')
        mu,lv=self.posterior(query);z=mu+(.5*lv).exp()*eps
        direct=F.mse_loss(self.decoder(current,z),target)
        kl=.5*(mu.square()+lv.exp()-1-lv).sum(-1).mean()
        flow=None
        if arm!='A':
            condition=z if arm=='B' else z.detach()
            flow=self.flow.training_output(current,condition,target,noise=flow_noise,time=flow_time).velocity_loss
        return dict(direct=direct,kl=kl,flow=flow)

def gradient_stats(parts,head):
    parameters=list(head.parameters())
    vectors=[]
    for key in ('direct','flow'):
        if parts[key] is None:vectors.append(None);continue
        gs=torch.autograd.grad(parts[key],parameters,retain_graph=True,allow_unused=True)
        vectors.append(torch.cat([(g if g is not None else torch.zeros_like(p)).flatten() for p,g in zip(parameters,gs)]))
    a,b=vectors
    out={'direct_head_norm':float(a.norm())}
    if b is None:out.update(flow_head_norm=None,cosine=None)
    else:
        out['flow_head_norm']=float(b.norm());den=a.norm()*b.norm()
        out['cosine']=float(a.dot(b)/den) if float(den)>0 else None
    return out

def clip_separately(model):
    # Avoid flow-branch gradient norm indirectly rescaling head/direct updates.
    return {name:float(torch.nn.utils.clip_grad_norm_(getattr(model,name).parameters(),1.)) for name in ('head','decoder','flow')}

@torch.inference_mode()
def flow_eval(model,data,x,batch=8):
    model.eval();mu,lv=model.posterior(x);gen=torch.Generator().manual_seed(1907);rows=[]
    for start in range(0,len(x),batch):
        h,y=data['current'][start:start+batch],data['future'][start:start+batch]
        z=mu[start:start+batch];noise=torch.randn(h.shape,generator=gen)
        # This path accepts no future target; all 4 Euler states are generated.
        pred=model.flow(h,z,noise=noise)
        mse=(pred-y).square().flatten(1).mean(1)
        rows.extend(float(v) for v in mse)
    return dict(generated_mse=sum(rows)/len(rows),rows=rows,code='posterior mean (true-future IDM); flow state itself generated',flow_steps=model.flow.steps)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('cache','queries','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--steps',type=int,default=400);p.add_argument('--seeds',type=int,nargs='+',default=[42])
    p.add_argument('--decoder-family',choices=['wide','narrow'],default='wide',help='Wide full-feature readout primary; narrow fixed-rank control')
    p.add_argument('--beta',type=float,default=5e-5);p.add_argument('--flow-weight',type=float,default=1.)
    a=p.parse_args()
    if a.output.exists() or a.steps<1 or a.beta<0 or a.flow_weight<0:p.error('New output, positive steps and nonnegative weights required')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('medium')
    meta=json.loads((a.queries/'manifest.json').read_text())
    data={s:torch.load(a.cache/f'{s}.pt',weights_only=True) for s in ('train','validation')}
    queries={s:torch.load(a.queries/f'{s}.pt',weights_only=True) for s in data}
    for s in data:
        assert file_hash(a.cache/f'{s}.pt')==meta['parent_cache_sha256'][s]
        assert queries[s]['rows']==data[s]['rows']
        assert torch.equal(queries[s]['indices'],torch.arange(len(data[s]['current'])))
    assert not {r['episode'] for r in data['train']['rows']} & {r['episode'] for r in data['validation']['rows']}
    cal=torch.load(a.queries/'train_calibration.pt',weights_only=True)
    x={s:representation(queries[s],cal,'reference') for s in data}
    a.output.mkdir(parents=True)
    report=dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},
        source_cache_hashes=meta['parent_cache_sha256'],query_manifest_sha256=file_hash(a.queries/'manifest.json'),
        code_sha256=file_hash(Path(__file__)),implementation_sha256={str(path.relative_to(ROOT)):file_hash(path) for path in [ROOT/'research/train_cached_decoders.py',ROOT/'research/wide_head_flow.py',ROOT/'research/idm/train_reference_heads.py',ROOT/'vendor/lawam/latent_action_model/core/utils/latent_flow.py']},results=[],
        scope='Frozen Q encoder; only Gaussian head/decoders train. A differs in active compute. B/C isolate flow-to-head gradient, not interpolation causality. Production uses SmoothL1; this pilot uses MSE.')
    for seed in a.seeds:
        expected_init=None;expected_schedule=None;expected_direct_final=None
        for arm in ('A','B','C'):
            torch.manual_seed(seed);model=JointModel(decoder_family=a.decoder_family);initial=weight_hash(model)
            if expected_init is None:expected_init=initial
            assert initial==expected_init
            groups=[dict(params=[p for p in model.parameters() if p.ndim>=2],weight_decay=.01),dict(params=[p for p in model.parameters() if p.ndim<2],weight_decay=0.)]
            opt=torch.optim.AdamW(groups,lr=3e-4)
            generators=[torch.Generator().manual_seed(seed+offset) for offset in (1000,2000,3000,4000)]
            schedule=hashlib.sha256();history=[];started=time.perf_counter()
            for step in range(1,a.steps+1):
                model.train();idx=torch.randint(len(x['train']),(8,),generator=generators[0])
                h,y=data['train']['current'][idx],data['train']['future'][idx]
                eps=torch.randn((8,1,32),generator=generators[1]);noise=torch.randn(h.shape,generator=generators[2]);t=torch.rand(8,generator=generators[3])
                for value in (idx,eps,noise,t):schedule.update(value.numpy().tobytes())
                parts=model.losses(x['train'][idx],h,y,eps,noise,t,arm)
                loss=parts['direct']+a.beta*parts['kl']
                if parts['flow'] is not None:loss=loss+a.flow_weight*parts['flow']
                if not torch.isfinite(loss):raise RuntimeError('Nonfinite objective')
                log=step==1 or step%100==0 or step==a.steps
                stats=gradient_stats(parts,model.head) if log else None
                opt.zero_grad(set_to_none=True);loss.backward();norms=clip_separately(model)
                if not all(torch.isfinite(torch.tensor(v)) for v in norms.values()):raise RuntimeError('Nonfinite gradient')
                opt.step()
                if log:
                    history.append(dict(step=step,losses={k:float(v.detach()) if v is not None else None for k,v in parts.items()},head_gradients=stats,preclip_norms=norms))
                    print(seed,arm,history[-1],flush=True)
            if expected_schedule is None:expected_schedule=schedule.hexdigest()
            assert schedule.hexdigest()==expected_schedule
            final=(weight_hash(model.head),weight_hash(model.decoder))
            if arm=='A':expected_direct_final=final
            if arm=='C' and final!=expected_direct_final:raise RuntimeError('A/C head/direct trajectories diverged; examine unintended coupling')
            result=dict(seed=seed,arm=arm,initial_sha256=initial,schedule_sha256=schedule.hexdigest(),history=history,
                head_sha256=final[0],direct_sha256=final[1],elapsed_seconds=time.perf_counter()-started,
                parameters_allocated=sum(p.numel() for p in model.parameters()),
                parameters_active=sum(p.numel() for p in model.head.parameters())+sum(p.numel() for p in model.decoder.parameters())+(sum(p.numel() for p in model.flow.parameters()) if arm!='A' else 0),
                direct_validation=evaluate(model,data['validation'],x['validation']),
                flow_validation=flow_eval(model,data['validation'],x['validation']) if arm!='A' else None)
            report['results'].append(result)
            torch.save(dict(state_dict=model.state_dict(),seed=seed,arm=arm,steps=a.steps),a.output/f'{arm}-{seed}.pt')
            (a.output/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
if __name__=='__main__':main()
