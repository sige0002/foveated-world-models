"""Train512 frozen-mu additive versus state-dependent coefficient kernel probe."""
import json,sys,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.probe_spatial_current import geometry,current_kernel,normalize_kernel,standardize,solve_many,CONFIGS
from research.idm.probe_nested512_displacement import ALPHAS,gripper_metrics
from research.idm.probe_displacement_ridge import scaler,FLOOR,sha256
from research.idm.train_reference_heads import representation
VARIANTS=['zero','additive','interaction']
def combine(g,cross,norm,z,zq,mult,variant):
 current,cquery,median=current_kernel(g,cross,norm,mult);current,cquery,cn=normalize_kernel(current,cquery)
 if variant=='zero':z=torch.zeros_like(z);zq=torch.zeros_like(zq)
 a,b,zc=standardize(z,zq);code,cq,zn=normalize_kernel(a@a.T/32,b@a.T/32);pn=None
 total,query=current+code,cquery+cq
 if variant=='interaction':
  product,pquery,pn=normalize_kernel(current*code,cquery*cq);total=total+product;query=query+pquery
 k,q,kn=normalize_kernel(total,query)
 return k,q,dict(current=cn,code=zn,product=pn,total=kn,code_scaler=zc,median_distance=median)
@torch.inference_mode()
def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');out=ROOT/'research/idm/interaction-probe512'
 if out.exists():raise FileExistsError(out)
 prior=ROOT/'research/idm/nested512-displacement-probe';pp=json.loads((prior/'protocol.json').read_text());pr=json.loads((prior/'results.json').read_text());ps=json.loads((prior/'train_selection.json').read_text());pf=torch.load(prior/'fit_artifacts.pt',weights_only=True);sources=dict(pp['source_sha256'])
 for path,digest in sources.items():assert sha256(Path(path))==digest
 sources[str(prior/'results.json')]=sha256(prior/'results.json');sources[str(prior/'train_selection.json')]=sha256(prior/'train_selection.json');sources[str(prior/'train_labels.pt')]=sha256(prior/'train_labels.pt');assert sources[str(prior/'train_labels.pt')]==pr['labels_sha256']
 labels512=torch.load(prior/'train_labels.pt',weights_only=True);cache=ROOT/'runs/reference-idm42-nested512-prior10000';meta=json.loads((cache/'cache_manifest.json').read_text());data={s:torch.load(cache/f'{s}.pt',weights_only=True,mmap=True) for s in ('train','validation')};data['observed96']=torch.load(ROOT/'runs/research-unused-96-prior6501/validation.pt',weights_only=True,mmap=True)
 torch.set_float32_matmul_precision('highest');images={};states={};ysplit={};codes={}
 for s,d in data.items():
  l=labels512 if s=='train' else torch.load(ROOT/f'research/idm/observed-displacement-labels/{s}.pt',weights_only=True);assert l['delta8_mask'].all()
  for row,lr in zip(d['rows'],l['rows']):assert all(row[k]==lr[k] for k in ('episode','index','dt'))
  images[s]=d['current'].flatten(1).float();states[s]=l['current_state20'].float();ysplit[s]=l['delta8'].double();codes[s]=d['codes'].reshape(len(d['rows']),32).double()
 folds=[torch.tensor(f) for f in pp['folds']];assert pp['alphas']==ALPHAS and pp['current_configs']==CONFIGS;bins=pp['bin_thresholds'];common_scale=pf['common_target_scale128'];out.mkdir();protocol=dict(variants=VARIANTS,alphas=ALPHAS,current_configs=CONFIGS,folds=pp['folds'],kernel='Train-center and mean-diagonal normalize current/code separately; interaction adds similarly centered/normalized Schur product of these centered current/code kernels; normalize total to mean diagonal1. Zero uses zero32 and reproduces current-only.',selection='Same512 five episode folds; all candidates and alphas train-only; save all selections before observed64/96 evaluation; no post-evaluation changes.',representation='Immutable reference42 mu32, no refit or new encoder extraction',mathematics='Known tensor-product RKHS/varying-coefficient model; additive f0(C)+w^T z versus additional state-dependent linear coefficients. Centering changes intercept/main effects but preserves PSD; no novelty claim.',target_scaling=pp['target_training_scaling'],common_evaluation_scale='Exact old128 target scale from prior512 fit artifact',right_gripper_bins=bins,source_sha256=sources,script_sha256=sha256(Path(__file__)),shared_kernel_source_sha256=sha256(ROOT/'research/idm/probe_spatial_current.py'),threads=2,interpretation='Observed-data readout sensitivity, not information absence/presence or action identity. Equal kernel strength but interaction changes capacity/regularization. No new world-model training or GPU.')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');started=time.perf_counter();allidx=torch.arange(512);cv={v:{c:torch.empty(len(ALPHAS),512,dtype=torch.float64) for c in CONFIGS} for v in VARIANTS}
 for fi,held in enumerate(folds):
  keep=allidx[~torch.isin(allidx,held)];g,cross,norm,_=geometry(images['train'][keep],states['train'][keep],images['train'][held],states['train'][held]);ym,ys=scaler(ysplit['train'][keep],FLOOR);target=(ysplit['train'][keep]-ym)/ys
  for name,mult in CONFIGS.items():
   for variant in VARIANTS:
    k,q,_=combine(g,cross,norm,codes['train'][keep],codes['train'][held],mult,variant);assert abs(float(k.diag().mean())-1)<1e-12;pred=solve_many(k,q,target,ALPHAS)
    for j,p in enumerate(pred):cv[variant][name][j,held]=(p-(ysplit['train'][held]-ym)/ys).square().mean(-1)
  print('CV fold',fi+1,'complete',flush=True)
 selection={}
 for variant in VARIANTS:
  candidates=[dict(config=c,alpha=a,cv_normalized_mse=float(cv[variant][c][j].mean()),per_example=cv[variant][c][j].tolist()) for c in CONFIGS for j,a in enumerate(ALPHAS)]
  if variant in ('zero','additive'):
   for previous in ps['zero' if variant=='zero' else 'reference42']['candidates']:
    matched=next(v for v in candidates if v['config']==previous['config'] and v['alpha']==previous['alpha']);assert abs(matched['cv_normalized_mse']-previous['cv_normalized_mse'])<1e-12
  best=min(candidates,key=lambda v:v['cv_normalized_mse']);selection[variant]=dict(selected=best,candidates=candidates,alpha_at_boundary=best['alpha'] in (ALPHAS[0],ALPHAS[-1]),selected_by_config={c:min([v for v in candidates if v['config']==c],key=lambda v:v['cv_normalized_mse']) for c in CONFIGS})
 (out/'train_selection.json').write_text(json.dumps(selection,indent=2)+'\n');names=['train','validation','observed96'];qi=torch.cat([images[s] for s in names]);qs=torch.cat([states[s] for s in names]);g,cross,norm,scalers=geometry(images['train'],states['train'],qi,qs);allcode=torch.cat([codes[s] for s in names]);ym,ys=scaler(ysplit['train'],FLOOR);target=(ysplit['train']-ym)/ys;results={};models={}
 for variant in VARIANTS:
  configurations={}
  for name,mult in CONFIGS.items():
   alpha=selection[variant]['selected_by_config'][name]['alpha'];k,q,normalization=combine(g,cross,norm,codes['train'],allcode,mult,variant);prediction=solve_many(k,q,target,[alpha])[0]*ys+ym;start=0;evaluation={}
   for s in names:
    n=len(data[s]['rows']);pred=prediction[start:start+n];y=ysplit[s];sq=(pred-y).square();common=sq/common_scale.square();own=sq/ys.square();evaluation[s]=dict(normalized_mse_old128_scale=float(common.mean()),normalized_mse_train512_scale=float(own.mean()),physical_coordinate_mse=sq.mean(0).tolist(),right_gripper=gripper_metrics(y,pred,bins['static'],bins['small_large']),rows=[dict(row,prediction=pred[i].tolist(),target=y[i].tolist(),normalized_mse_old128_scale=float(common[i].mean()),physical_coordinate_squared_error=sq[i].tolist()) for i,row in enumerate(data[s]['rows'])]);start+=n
   configurations[name]=dict(alpha=alpha,evaluation=evaluation);models[variant+'_'+name]=dict(kernel=k,normalization=normalization,alpha=alpha)
  chosen=selection[variant]['selected']['config'];results[variant]=dict(selected_config=chosen,selected_alpha=selection[variant]['selected']['alpha'],selected_evaluation=configurations[chosen]['evaluation'],all_prespecified_configurations=configurations);print(variant,chosen,selection[variant]['selected']['alpha'],{s:v['normalized_mse_old128_scale'] for s,v in results[variant]['selected_evaluation'].items()},flush=True)
 for variant,previous in [('zero','zero'),('additive','reference42')]:
  for s in names:assert abs(results[variant]['selected_evaluation'][s]['normalized_mse_old128_scale']-pr['results'][previous]['selected_evaluation'][s]['normalized_mse_old128_scale'])<1e-12
 for path,digest in sources.items():assert sha256(Path(path))==digest
 torch.save(dict(current_scalers=scalers,target_mean512=ym,target_scale512=ys,common_target_scale128=common_scale,models=models),out/'fit_artifacts.pt');report=dict(protocol_sha256=sha256(out/'protocol.json'),selection_sha256=sha256(out/'train_selection.json'),fit_artifacts_sha256=sha256(out/'fit_artifacts.pt'),additive_and_zero_previous_cv_and_eval_reproduced=True,seconds=time.perf_counter()-started,results=results);(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
