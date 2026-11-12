"""Prepared CPU joint Gaussian-head/prior-residual comparison; main starts training."""
import argparse,hashlib,json,math,sys,time
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.train_cached_decoders import make_decoder,file_hash
from research.idm.postprojection_representations import VARIANTS,representation
from research.idm.postprojection_heads import shared_initial_heads
from research.idm.train_reference_heads import weight_hash
class Model(nn.Module):
 def __init__(self,seed,feature_dim=768,grid=16,dim=128,layers=2):
  super().__init__();self.head=shared_initial_heads(seed)['direction']
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed+3000);self.decoder=make_decoder('prior_residual',dim,layers,feature_dim,32,grid)
 def posterior(self,x):
  mu,lv=self.head(x).chunk(2,-1);return mu,lv.clamp(-10,10)
def schedule(seed,steps,batch_size,n):
 indices=torch.Generator().manual_seed(seed+1000);noise=torch.Generator().manual_seed(seed+2000)
 for _ in range(steps):yield torch.randint(n,(batch_size,),generator=indices),torch.randn(batch_size,1,32,generator=noise)
def objective(model,x,h,y,baseline,eps,beta=5e-5):
 mu,lv=model.posterior(x.detach());z=mu+(.5*lv).exp()*eps
 pred=model.decoder(h.detach(),z,baseline=baseline.detach())
 mse=F.mse_loss(pred,y.detach());kl=.5*(mu.square()+lv.exp()-1-lv).sum(-1).mean()
 return mse+beta*kl,mse,kl
def add_constant_control(features):
 """Return new feature mappings; never mutate cached three-control tensors."""
 return {split:dict(values,constant=torch.zeros_like(values["direction"])) for split,values in features.items()}

def clip_gradients(model):
 norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
 if not torch.isfinite(norm):raise RuntimeError("Nonfinite gradient norm")
 return float(norm)

def head_gradient_diagnostics(model,mse,kl,beta):
 params=tuple(model.head.parameters())
 def norm(loss):
  gradients=torch.autograd.grad(loss,params,retain_graph=True,allow_unused=True)
  return float(torch.sqrt(sum((g.detach().double().square().sum() for g in gradients if g is not None),torch.tensor(0.,dtype=torch.float64))))
 mse_norm,kl_norm=norm(mse),norm(kl)
 return dict(head_mse_gradient_norm=mse_norm,head_raw_kl_gradient_norm=kl_norm,head_weighted_kl_gradient_norm=beta*kl_norm)

@torch.inference_mode()
def evaluate(model,data,x,batch=8):
 model.eval();mu,lv=model.posterior(x);eps=torch.randn(mu.shape,generator=torch.Generator().manual_seed(1407));rows=[]
 for start in range(0,len(x),batch):
  sl=slice(start,start+batch);h,y,b=[data[k][sl] for k in ('current','future','baseline')];m,l=mu[sl],lv[sl]
  mean=model.decoder(h,m,baseline=b);sample=model.decoder(h,m+(.5*l).exp()*eps[sl],baseline=b)
  errors={k:(v-y).square().flatten(1).mean(1) for k,v in [('mean_mse',mean),('sample_mse',sample),('prior_mse',b)]}
  for i in range(len(h)):rows.append(dict(data['rows'][start+i],**{k:float(v[i]) for k,v in errors.items()}))
 return dict(metrics={k:sum(r[k] for r in rows)/len(rows) for k in errors},rows=rows,eval_eps_sha256=hashlib.sha256(eps.numpy().tobytes()).hexdigest())
def load_inputs(cache,queries):
 meta=json.loads((queries/'manifest.json').read_text());source=json.loads((cache/'cache_manifest.json').read_text())
 if file_hash(cache/'cache_manifest.json')!=meta['source_manifest_sha256']:raise ValueError('Source manifest changed')
 cal=torch.load(queries/'train_calibration.pt',weights_only=True)
 if cal['fit_split']!='train' or cal['fit_count']!=128:raise ValueError('Expected fixed train128 calibration')
 for name,digest in meta['output_sha256'].items():
  if file_hash(queries/name)!=digest:raise ValueError('Representation artifact hash mismatch')
 data={};features={}
 for split,expected in [('train',128),('validation',64)]:
  if file_hash(cache/f'{split}.pt')!=meta['source_cache_sha256'][split] or meta['source_cache_sha256'][split]!=source['cache_sha256'][split]:raise ValueError('Source data hash mismatch')
  d=torch.load(cache/f'{split}.pt',weights_only=True);q=torch.load(queries/f'{split}.pt',weights_only=True)
  if len(d['rows'])!=expected or d['rows']!=q['rows']:raise ValueError('Rows/count mismatch')
  for k in ('current','future','baseline'):
   if d[k].requires_grad or not torch.isfinite(d[k]).all():raise ValueError('Nonfixed/nonfinite input')
  data[split]=d;features[split]={v:representation(q['projection'],cal,v).detach() for v in VARIANTS}
  for v in VARIANTS:
   if not torch.equal(features[split][v],q['representations'][v]):raise ValueError('Representation reproduction failed')
 if {r['episode'] for r in data['train']['rows']} & {r['episode'] for r in data['validation']['rows']}:raise ValueError('Episode overlap')
 return data,features,meta

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,default=ROOT/'runs/reference-idm-with-prior6501');p.add_argument('--queries',type=Path,default=ROOT/'research/idm/mature-postprojection-cache');p.add_argument('--output',type=Path,required=True);p.add_argument('--threads',type=int,default=2);p.add_argument('--seeds',type=int,nargs='+',default=[42]);p.add_argument('--steps',type=int,default=400);p.add_argument('--batch-size',type=int,default=8);p.add_argument('--lr',type=float,default=3e-4);p.add_argument('--beta',type=float,default=5e-5);p.add_argument('--include-constant-control',action='store_true');a=p.parse_args()
 if min(a.threads,a.steps,a.batch_size)<=0 or a.lr<=0 or a.beta<0:p.error('Invalid settings')
 if a.output.exists():raise FileExistsError(a.output)
 torch.set_num_threads(a.threads);torch.set_float32_matmul_precision('highest')
 data,x,meta=load_inputs(a.cache,a.queries)
 variants=(*VARIANTS,'constant') if a.include_constant_control else VARIANTS
 if a.include_constant_control:x=add_constant_control(x)
 tokens=data['train']['current'].shape[-2];grid=math.isqrt(tokens)
 if grid*grid!=tokens:raise ValueError('Non-square grid')
 files=[Path(__file__),ROOT/'research/idm/postprojection_representations.py',ROOT/'research/idm/postprojection_heads.py',ROOT/'research/train_cached_decoders.py',ROOT/'research/prior_correction.py',ROOT/'research/wide_head_flow.py',ROOT/'vendor/lawam/latent_action_model/core/utils/lam_decoder.py',ROOT/'vendor/lawam/latent_action_model/core/utils/latent_flow.py']
 report=dict(settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()},precision='float32 highest CPU',optimizer=dict(name='AdamW',weight_decay_matrix=.01,weight_decay_bias_norm=0.,clip_global_norm=1.),architecture='34→256→64 Gaussian head;32mu/32logvar clamp[-10,10]; prior_residual dim128 layers2',objective='MSE + beta*mean(sum_code_dimension KL(q||N(0,I))); sampled posterior in training',query_manifest=meta,query_manifest_sha256=file_hash(a.queries/'manifest.json'),source_hashes={str(f):file_hash(f) for f in files},limitations='New trained head on postprojection representations; differs from production input path/loss/decoder, not a same-conditions improvement of production head. Validation64 only;96 not evaluated. Frozen future-conditioned posterior, not execution-time prior.',results=[])
 a.output.mkdir(parents=True)
 for seed in a.seeds:
  expected=None
  for variant in variants:
   model=Model(seed,feature_dim=data['train']['current'].shape[-1],grid=grid);initial=dict(full=weight_hash(model),head=weight_hash(model.head),decoder=weight_hash(model.decoder))
   opt=torch.optim.AdamW([dict(params=[v for v in model.parameters() if v.ndim>=2],weight_decay=.01),dict(params=[v for v in model.parameters() if v.ndim<2],weight_decay=0.)],lr=a.lr)
   bh=hashlib.sha256();eh=hashlib.sha256();history=[];gradient_history=[];started=time.perf_counter()
   for step,(idx,eps) in enumerate(schedule(seed,a.steps,a.batch_size,128),1):
    model.train();bh.update(idx.numpy().tobytes());eh.update(eps.numpy().tobytes());opt.zero_grad(set_to_none=True)
    loss,mse,kl=objective(model,x['train'][variant][idx],*[data['train'][k][idx] for k in ('current','future','baseline')],eps,a.beta)
    if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
    if step in (1,10,100,400):gradient_history.append(dict(step=step,**head_gradient_diagnostics(model,mse,kl,a.beta)))
    loss.backward();gradient_norm=clip_gradients(model);opt.step()
    if step%100==0 or step==a.steps:history.append(dict(step=step,loss=float(loss.detach()),mse=float(mse.detach()),kl=float(kl.detach()),gradient_norm_before_clipping=gradient_norm));print(seed,variant,step,flush=True)
   fairness=dict(initial=initial,batch_sha256=bh.hexdigest(),posterior_eps_sha256=eh.hexdigest())
   if expected is None:expected=fairness
   if fairness!=expected:raise RuntimeError('Matched initialization/schedule failed')
   evaluation=evaluate(model,data['validation'],x['validation'][variant],a.batch_size)
   train_evaluation=evaluate(model,data['train'],x['train'][variant],a.batch_size)
   path=a.output/f'{variant}-{seed}.pt';torch.save(dict(state_dict=model.state_dict(),variant=variant,seed=seed,steps=a.steps,config=report['settings'],architecture=report['architecture'],calibration_sha256=meta['output_sha256']['train_calibration.pt'],query_manifest_sha256=report['query_manifest_sha256'],source_cache_sha256=meta['source_cache_sha256']),path)
   result=dict(seed=seed,variant=variant,fairness=fairness,history=history,gradient_diagnostics=gradient_history,validation=evaluation,train=train_evaluation,checkpoint=str(path),checkpoint_sha256=file_hash(path),final_weight_sha256=weight_hash(model),seconds=time.perf_counter()-started)
   report['results'].append(result);(a.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
 for split,digest in meta['source_cache_sha256'].items():
  if file_hash(a.cache/f'{split}.pt')!=digest:raise RuntimeError('Source mutated during training')
if __name__=='__main__':main()
