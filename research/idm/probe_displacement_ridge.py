"""Prespecified train-only-CV linear utility of frozen mean codes; CPU2."""
import json,hashlib,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.cache_reference_queries import sha256
ALPHAS=[.01,.1,1.,10.,100.,1000.];VARIANTS=['zero','original6501','reference42']
FLOOR=torch.tensor([.001,.001,.001,.01,.001,.001,.001,.01],dtype=torch.float64)
def scaler(x,floor):return x.mean(0),x.std(0,unbiased=False).clamp_min(floor)
def fit(x,y,alpha):
 xm,xs=scaler(x,1e-6);ym,ys=scaler(y,FLOOR);xx=(x-xm)/xs;yy=(y-ym)/ys
 # Dual solution implements sum squared errors + alpha times weight norm squared.
 dual=torch.linalg.solve(xx@xx.T+alpha*torch.eye(len(xx),dtype=x.dtype),yy)
 return dict(weight=xx.T@dual,x_mean=xm,x_scale=xs,y_mean=ym,y_scale=ys)
def predict(model,x):return (((x-model['x_mean'])/model['x_scale'])@model['weight'])*model['y_scale']+model['y_mean']
def fold_indices(episodes):
 unique=sorted(set(episodes));rng=np.random.default_rng(20261113);rng.shuffle(unique);parts=np.array_split(np.array(unique),5)
 return [torch.tensor([i for i,e in enumerate(episodes) if e in set(part.tolist())]) for part in parts]
def verify_rows(cache,labels):
 if len(cache['rows'])!=len(labels['rows']):raise ValueError('Row count mismatch')
 for row,label in zip(cache['rows'],labels['rows']):
  for k,v in row.items():
   if label.get(k)!=v:raise ValueError(f'Row mismatch {k}')
def load():
 lb=ROOT/'research/idm/observed-displacement-labels';lm=json.loads((lb/'manifest.json').read_text());data={};sources={}
 for split in ('train','validation','observed96'):
  lpath=lb/f'{split}.pt';assert sha256(lpath)==lm['output_sha256'][lpath.name];labels=torch.load(lpath,weights_only=True)
  src=lm['source_caches'][split];cdir=Path(src['cache']);cpath=cdir/f"{src['split']}.pt";assert sha256(cpath)==src['cache_sha256'];cache=torch.load(cpath,weights_only=True);verify_rows(cache,labels)
  assert labels['delta8_mask'].all() and labels['current_state_observed_mask'].all()
  if split=='observed96':original=cache['original_codes']
  else:
   opath=ROOT/f'runs/frozen-idm-step6501/{split}.pt';om=json.loads((opath.parent/'cache_manifest.json').read_text());assert sha256(opath)==om['cache_sha256'][split]
   old=torch.load(opath,weights_only=True)
   assert torch.equal(old['current'],cache['current']) and torch.equal(old['future'],cache['future'])
   assert [(r['episode'],r['index'],r['dt']) for r in old['rows']]==[(r['episode'],r['index'],r['dt']) for r in cache['rows']]
   original=old['codes'];sources[str(opath)]=sha256(opath)
  n=len(labels['rows']);common=torch.cat((cache['current'].mean((1,2)),labels['current_state20']),-1).double()
  codes=dict(zero=torch.zeros(n,32),original6501=original.reshape(n,32),reference42=cache['codes'].reshape(n,32))
  data[split]=dict(x={k:torch.cat((common,z.double()),-1) for k,z in codes.items()},y=labels['delta8'].double(),rows=cache['rows'])
  assert common.shape==(n,788) and all(x.shape==(n,820) and torch.isfinite(x).all() for x in data[split]['x'].values()) and torch.isfinite(data[split]['y']).all()
  sources.update({str(lpath):sha256(lpath),str(cpath):sha256(cpath),str(cdir/'cache_manifest.json'):sha256(cdir/'cache_manifest.json')})
 sources[str(lb/'manifest.json')]=sha256(lb/'manifest.json')
 return data,sources
@torch.inference_mode()
def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');out=ROOT/'research/idm/displacement-ridge-probe'
 if out.exists():raise FileExistsError(out)
 data,sources=load();episodes=[str(r['episode']) for r in data['train']['rows']];folds=fold_indices(episodes)
 assert sorted(torch.cat(folds).tolist())==list(range(128))
 for split in ('validation','observed96'):assert not set(episodes)&{str(r['episode']) for r in data[split]['rows']}
 protocol=dict(seed=20261113,folds=[f.tolist() for f in folds],fold_episode_ids=[[episodes[i] for i in f] for f in folds],alphas=ALPHAS,variants=VARIANTS,feature_dim=820,current='DINO mean over temporal and patch axes768 + current prepared state20',code='32 posterior mean, zero/original6501/reference42',feature_std_floor=1e-6,target_std_floors=FLOOR.tolist(),standardization='Population mean/std fit separately on each fold training subset; fulltrain refit for final evaluation',ridge='sum squared normalized target error + alpha*sum squared coefficients; centered intercept unpenalized; float64 dual solve',selection='Lowest mean per-example normalized8D squared error over pooled heldout train folds; alpha ascending first wins exact ties',masks='All label and current-state coordinates must be observed; otherwise abort instead of implicit imputation',evaluation='validation64 and previously research-observed96, fixed fulltrain target normalization; no test tuning',threads=2,dtype='float64',torch_version=str(torch.__version__),source_sha256=sources,script_sha256=sha256(Path(__file__)),limitations='Linear utility for fixed representations, not conditional MI/action identification. Pooled current may be insufficient, and Z may encode nonlinear current information. Mu probe differs from noisy Gaussian single-sample decoding. observed96 already used in visual research and is not an independent final test.')
 out.mkdir();(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 # Protocol exists before any hyperparameter fitting or evaluation.
 results={};models={};allidx=torch.arange(128);train=data['train'];ym,ys=scaler(train['y'],FLOOR)
 for variant in VARIANTS:
  cv=[]
  for alpha in ALPHAS:
   losses=torch.empty(128,dtype=torch.float64)
   for held in folds:
    keep=allidx[~torch.isin(allidx,held)];m=fit(train['x'][variant][keep],train['y'][keep],alpha)
    losses[held]=((predict(m,train['x'][variant][held])-train['y'][held])/m['y_scale']).square().mean(-1)
   cv.append(dict(alpha=alpha,normalized_mse=float(losses.mean()),per_example=losses.tolist()))
  alpha=min(cv,key=lambda v:v['normalized_mse'])['alpha'];m=fit(train['x'][variant],train['y'],alpha);models[variant]=m;evaluations={}
  for split,d in data.items():
   pred=predict(m,d['x'][variant]);squared=(pred-d['y']).square();norm=squared/ys.square()
   evaluations[split]=dict(normalized_mse=float(norm.mean()),physical_coordinate_mse=squared.mean(0).tolist(),normalized_coordinate_mse=norm.mean(0).tolist(),rows=[dict(r,prediction=pred[i].tolist(),target=d['y'][i].tolist(),physical_coordinate_squared_error=squared[i].tolist(),normalized_mse=float(norm[i].mean())) for i,r in enumerate(d['rows'])])
  results[variant]=dict(selected_alpha=alpha,cv=cv,evaluation=evaluations)
  print(variant,alpha,{s:v['normalized_mse'] for s,v in evaluations.items()},flush=True)
 for path,digest in sources.items():assert sha256(Path(path))==digest
 torch.save(dict(models=models,fulltrain_target_mean=ym,fulltrain_target_scale=ys),out/'fitted_models.pt')
 report=dict(protocol_sha256=sha256(out/'protocol.json'),models_sha256=sha256(out/'fitted_models.pt'),results=results,physical_mse_units=['m^2','m^2','m^2','gripper_source_units^2','m^2','m^2','m^2','gripper_source_units^2'])
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
