"""Prespecified spatial/nonlinear current control for frozen displacement probes."""
import json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.probe_displacement_ridge import load,fold_indices,scaler,FLOOR,VARIANTS,sha256
ALPHAS=[1e-4,1e-3,.01,.1,1.,10.,100.,1000.,10000.]
CONFIGS={'spatial_linear':None,'spatial_rbf025':.25,'spatial_rbf1':1.,'spatial_rbf4':4.}
def standardize(train,query,floor=1e-6):
 mean=train.mean(0);scale=train.std(0,unbiased=False).clamp_min(floor)
 return (train-mean)/scale,(query-mean)/scale,dict(mean=mean,scale=scale)
def geometry(images,states,qi,qs):
 a,b,ic=standardize(images,qi);s,t,sc=standardize(states,qs)
 d=a.shape[1];g=((a@a.T)/d+(s@s.T)/s.shape[1]).double()/2
 cross=((b@a.T)/d+(t@s.T)/s.shape[1]).double()/2
 qnorm=(b.square().mean(-1)+t.square().mean(-1)).double()/2
 return g,cross,qnorm,dict(image=ic,state=sc)
def current_kernel(g,cross,qnorm,multiplier):
 if multiplier is None:return g,cross,None
 dist=(g.diag()[:,None]+g.diag()[None,:]-2*g).clamp_min(0)
 positive=dist[torch.triu(torch.ones_like(dist,dtype=torch.bool),diagonal=1)&(dist>0)]
 median=float(positive.median()) if len(positive) else 1.
 qdist=(qnorm[:,None]+g.diag()[None,:]-2*cross).clamp_min(0)
 scale=2*median*multiplier**2
 return (-dist/scale).exp(),(-qdist/scale).exp(),median
def normalize_kernel(k,cross):
 means=k.mean(0);grand=k.mean();kc=k-k.mean(1,keepdim=True)-means[None,:]+grand
 qc=cross-cross.mean(1,keepdim=True)-means[None,:]+grand
 scale=float(kc.diag().mean())
 if scale<=1e-12:return torch.zeros_like(k),torch.zeros_like(cross),dict(mean=means,grand=grand,scale=0.)
 return kc/scale,qc/scale,dict(mean=means,grand=grand,scale=scale)
def combined_kernel(g,cross,qnorm,ztrain,zquery,multiplier):
 c,cq,median=current_kernel(g,cross,qnorm,multiplier);c,cq,cn=normalize_kernel(c,cq)
 z,zq,zc=standardize(ztrain,zquery);zk,zqk,zn=normalize_kernel(z@z.T/32,zq@z.T/32)
 k,q,kn=normalize_kernel(c+zk,cq+zqk)
 return k,q,dict(current_normalization=cn,code_normalization=zn,sum_normalization=kn,code_scaler=zc,median_current_distance=median)
def solve_many(k,cross,targets,alphas):
 eig,u=torch.linalg.eigh((k+k.T)/2)
 if float(eig.min()) < -1e-4:raise ValueError('Unexpected non-PSD kernel')
 eig=eig.clamp_min(0);projection=u.T@targets;qu=cross@u
 return [qu@(projection/(eig[:,None]+a)) for a in alphas]
@torch.inference_mode()
def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');out=ROOT/'research/idm/spatial-current-displacement-probe'
 if out.exists():raise FileExistsError(out)
 data,sources=load();images={};states={};codes={}
 labels=json.loads((ROOT/'research/idm/observed-displacement-labels/manifest.json').read_text())
 for split,d in data.items():
  src=labels['source_caches'][split];cached=torch.load(Path(src['cache'])/f"{src['split']}.pt",weights_only=True)
  images[split]=cached['current'].flatten(1).float();states[split]=d['x']['zero'][:,768:788].float();codes[split]={v:d['x'][v][:,-32:] for v in VARIANTS}
  assert images[split].shape[1]==256*768
 episodes=[str(r['episode']) for r in data['train']['rows']];folds=fold_indices(episodes);oldbase=ROOT/'research/idm/displacement-ridge-boundary-audit';old=json.loads((oldbase/'results.json').read_text());oldprotocol=json.loads((oldbase/'protocol.json').read_text());assert [f.tolist() for f in folds]==oldprotocol['folds']
 protocol=dict(alphas=ALPHAS,current_configs=CONFIGS,variants=VARIANTS,folds=[f.tolist() for f in folds],seed=20261113,input='All256x768 current DINO coordinates flattened with fixed patch order, plus current20D state; original/ref posterior means or zero code',current_linear_geometry='Average of image dot/d_image and state dot/20 after per-coordinate train-only mean/std(floor1e-6)',rbf='exp(-squared_current_distance/(2*train_median_positive_pair_distance*bandwidth_multiplier^2))',code_kernel='Linear dot/32 after train-only per-coordinate standardization',normalization='Center each component using training Gram means; normalize each nonzero component by training mean diagonal; sum current+code and recenter/renormalize to training mean diagonal1; zero code remainszero, total kernel scale1 in every group',targets='Train-fold population mean/std, floors .001m xyz/.01 gripper, refit on fulltrain only after CV',selection='Common5episode CV; minimize pooled heldout normalized8D MSE, per configuration alpha and then configuration jointly per variant; order in configs/alphas wins exact ties; save every selection before any validation/observed96 prediction',kernel_regularization='K_centered_unit_trace + alpha I; float64 eigensolve, tiny negative eigenvalues clippedzero; intercept targetmean unpenalized',current_geometry_dtype='Float32 highest CPU Gram construction, float64 kernels/solve/targets',threads=2,source_sha256=sources,original_results_sha256=sha256(oldbase/'results.json'),script_sha256=sha256(Path(__file__)),limits='Already research-observed64/96, no independent finaltest. This tests specified richer current representations, not all possible current information. Additive linear z kernel is fixed; no world-model training/GPU. No claim of conditional MI or action identity.')
 out.mkdir();(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');started=time.perf_counter();cv={v:{c:torch.empty(len(ALPHAS),128,dtype=torch.float64) for c in CONFIGS} for v in VARIANTS};allidx=torch.arange(128);foldinfo=[]
 for fi,held in enumerate(folds):
  keep=allidx[~torch.isin(allidx,held)];g,cross,norm,_=geometry(images['train'][keep],states['train'][keep],images['train'][held],states['train'][held]);ym,ys=scaler(data['train']['y'][keep],FLOOR);target=(data['train']['y'][keep]-ym)/ys
  for name,mult in CONFIGS.items():
   for variant in VARIANTS:
    k,q,kn=combined_kernel(g,cross,norm,codes['train'][variant][keep],codes['train'][variant][held],mult)
    pred=solve_many(k,q,target,ALPHAS)
    for j,value in enumerate(pred):cv[variant][name][j,held]=(value-(data['train']['y'][held]-ym)/ys).square().mean(-1)
    foldinfo.append(dict(fold=fi,config=name,variant=variant,kernel_mean_diagonal=float(k.diag().mean()),median_current_distance=kn['median_current_distance']))
  print('CV fold',fi+1,'/5 complete',flush=True)
 selection={}
 for variant in VARIANTS:
  candidates=[]
  for name in CONFIGS:
   for j,alpha in enumerate(ALPHAS):candidates.append(dict(config=name,alpha=alpha,cv_normalized_mse=float(cv[variant][name][j].mean()),per_example=cv[variant][name][j].tolist()))
  best=min(candidates,key=lambda v:v['cv_normalized_mse']);byconfig={c:min([v for v in candidates if v['config']==c],key=lambda v:v['cv_normalized_mse']) for c in CONFIGS}
  selection[variant]=dict(selected=best,selected_by_config=byconfig,candidates=candidates,alpha_at_boundary=best['alpha'] in (ALPHAS[0],ALPHAS[-1]))
 (out/'train_selection.json').write_text(json.dumps(dict(selection=selection,fold_normalization=foldinfo),indent=2)+'\n')
 # All fitting choices locked before these evaluation predictions.
 splitnames=['train','validation','observed96'];sizes=[len(data[s]['rows']) for s in splitnames];qi=torch.cat([images[s] for s in splitnames]);qs=torch.cat([states[s] for s in splitnames]);g,cross,norm,fullscalers=geometry(images['train'],states['train'],qi,qs);ym,ys=scaler(data['train']['y'],FLOOR);target=(data['train']['y']-ym)/ys;results={};model_records={}
 for variant in VARIANTS:
  configurations={};zq=torch.cat([codes[s][variant] for s in splitnames])
  for name,mult in CONFIGS.items():
   alpha=selection[variant]['selected_by_config'][name]['alpha'];k,q,kn=combined_kernel(g,cross,norm,codes['train'][variant],zq,mult);pred=solve_many(k,q,target,[alpha])[0]*ys+ym;metrics={};start=0
   for split,n in zip(splitnames,sizes):
    prediction=pred[start:start+n];sq=(prediction-data[split]['y']).square();normalized=sq/ys.square();metrics[split]=dict(normalized_mse=float(normalized.mean()),physical_coordinate_mse=sq.mean(0).tolist(),rows=[dict(row,prediction=prediction[i].tolist(),target=data[split]['y'][i].tolist(),normalized_mse=float(normalized[i].mean()),physical_coordinate_squared_error=sq[i].tolist()) for i,row in enumerate(data[split]['rows'])]);start+=n
   configurations[name]=dict(alpha=alpha,evaluation=metrics)
   model_records[variant+'_'+name]=dict(alpha=alpha,kernel=k,normalization=kn)
  chosen=selection[variant]['selected']['config'];results[variant]=dict(selected_config=chosen,selected_alpha=selection[variant]['selected']['alpha'],selected_evaluation=configurations[chosen]['evaluation'],all_prespecified_configurations=configurations,old_pooled_metrics={s:old['results'][variant]['evaluation'][s]['normalized_mse'] for s in splitnames});print(variant,chosen,selection[variant]['selected']['alpha'],{s:v['normalized_mse'] for s,v in results[variant]['selected_evaluation'].items()},flush=True)
 for path,digest in sources.items():assert sha256(Path(path))==digest
 torch.save(dict(fulltrain_scalers=fullscalers,target_mean=ym,target_scale=ys,models=model_records),out/'kernel_fit_artifacts.pt')
 report=dict(protocol_sha256=sha256(out/'protocol.json'),selection_sha256=sha256(out/'train_selection.json'),fit_artifacts_sha256=sha256(out/'kernel_fit_artifacts.pt'),seconds=time.perf_counter()-started,results=results)
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
