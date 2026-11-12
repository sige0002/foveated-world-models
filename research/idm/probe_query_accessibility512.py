"""Matched-readout access to frozen query differences versus learned mu, train512."""
import json,sys,time
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.probe_spatial_current import geometry,current_kernel,normalize_kernel,standardize,solve_many,CONFIGS
from research.idm.probe_nested512_displacement import ALPHAS,gripper_metrics
from research.idm.probe_displacement_ridge import scaler,FLOOR,sha256
from research.idm.train_reference_heads import representation
VARIANTS=['mu32','query1024','query_pca32']
def pca_fit(train,rank=32):
 x,_,cal=standardize(train,train);_,singular,vh=torch.linalg.svd(x,full_matrices=False);components=vh[:rank].T
 return dict(mean=cal['mean'],scale=cal['scale'],components=components,explained_variance_fraction=float(singular[:rank].square().sum()/singular.square().sum()))
def pca_apply(x,pca):return ((x-pca['mean'])/pca['scale'])@pca['components']
def combine(g,cross,norm,z,zq,mult):
 current,cquery,median=current_kernel(g,cross,norm,mult);current,cquery,cn=normalize_kernel(current,cquery)
 a,b,zc=standardize(z,zq);d=z.shape[-1];code,cq,zn=normalize_kernel(a@a.T/d,b@a.T/d);k,q,kn=normalize_kernel(current+code,cquery+cq)
 return k,q,dict(current=cn,code=zn,total=kn,code_scaler=zc,median_distance=median)
@torch.inference_mode()
def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');out=ROOT/'research/idm/query-accessibility512'
 if out.exists():raise FileExistsError(out)
 prior=ROOT/'research/idm/nested512-displacement-probe';pp=json.loads((prior/'protocol.json').read_text());pr=json.loads((prior/'results.json').read_text());ps=json.loads((prior/'train_selection.json').read_text());pf=torch.load(prior/'fit_artifacts.pt',weights_only=True);sources=dict(pp['source_sha256'])
 for path,digest in sources.items():assert sha256(Path(path))==digest
 sources[str(prior/'results.json')]=sha256(prior/'results.json');sources[str(prior/'train_selection.json')]=sha256(prior/'train_selection.json');sources[str(prior/'train_labels.pt')]=sha256(prior/'train_labels.pt');assert sources[str(prior/'train_labels.pt')]==pr['labels_sha256']
 labels512=torch.load(prior/'train_labels.pt',weights_only=True);cache=ROOT/'runs/reference-idm42-nested512-prior10000';meta=json.loads((cache/'cache_manifest.json').read_text());data={s:torch.load(cache/f'{s}.pt',weights_only=True,mmap=True) for s in ('train','validation')};data['observed96']=torch.load(ROOT/'runs/research-unused-96-prior6501/validation.pt',weights_only=True,mmap=True)
 qpaths=dict(old_train=ROOT/'research/idm/reference-query-full/train.pt',added=cache/'additional_queries.pt',validation=ROOT/'research/idm/reference-query-full/validation.pt',observed96=ROOT/'research/idm/observed96-reference-queries2900/validation.pt');qfiles={k:torch.load(path,weights_only=True) for k,path in qpaths.items()}
 assert sha256(qpaths['old_train'])==meta['extraction_source_sha256'][str(qpaths['old_train'])]
 audit=json.loads((ROOT/'research/idm/nested512-independent-verification.json').read_text());assert sha256(qpaths['added'])==audit['additional_queries_sha256']
 q96m=json.loads((qpaths['observed96'].parent/'manifest.json').read_text());assert sha256(qpaths['observed96'])==q96m['query_sha256']
 qdata=dict(train={k:torch.cat((qfiles['old_train'][k],qfiles['added'][k])) for k in ('query','reference')},validation=qfiles['validation'],observed96=qfiles['observed96']);qdata['train']['rows']=qfiles['old_train']['rows']+qfiles['added']['rows']
 for s,d in data.items():
  assert qdata[s]['rows']==d['rows']
  assert len(qdata[s]['query'])==len(d['rows'])
 for path in qpaths.values():sources[str(path)]=sha256(path)
 headpath=Path(meta['export']['head_checkpoint']);calpath=ROOT/'research/idm/reference-query-full/train_calibration.pt';assert sha256(headpath)==meta['export']['head_checkpoint_sha256'] and sha256(calpath)==meta['export']['calibration_sha256'];sources[str(headpath)]=sha256(headpath);sources[str(calpath)]=sha256(calpath);headck=torch.load(headpath,weights_only=True);head=nn.Sequential(nn.Linear(1024,256),nn.GELU(),nn.Linear(256,64)).eval().requires_grad_(False);head.load_state_dict({k[5:]:v for k,v in headck['state_dict'].items() if k.startswith('head.')},strict=True);cal=torch.load(calpath,weights_only=True);del headck
 torch.set_float32_matmul_precision('medium');mu_errors={}
 for s,d in data.items():
  x=representation(qdata[s],cal,'reference');batch=128 if s=='train' else 64;mu=torch.cat([head(v).chunk(2,-1)[0] for v in x.split(batch)]);mu_errors[s]=float((mu-d['codes']).abs().max());print('mu reproduction',s,mu_errors[s],flush=True);assert torch.equal(mu,d['codes']) if s!='observed96' else mu_errors[s]<2e-4,s
 torch.set_float32_matmul_precision('highest');images={};states={};ysplit={};codes={};queries={}
 for s,d in data.items():
  l=labels512 if s=='train' else torch.load(ROOT/f'research/idm/observed-displacement-labels/{s}.pt',weights_only=True);assert l['delta8_mask'].all()
  for row,lr in zip(d['rows'],l['rows']):assert all(row[k]==lr[k] for k in ('episode','index','dt'))
  images[s]=d['current'].flatten(1).float();states[s]=l['current_state20'].float();ysplit[s]=l['delta8'].double();codes[s]=d['codes'].reshape(len(d['rows']),32).double();queries[s]=(qdata[s]['query']-qdata[s]['reference']).reshape(len(d['rows']),1024).double()
 folds=[torch.tensor(f) for f in pp['folds']];assert pp['alphas']==ALPHAS and pp['current_configs']==CONFIGS;bins=pp['bin_thresholds'];common_scale=pf['common_target_scale128'];out.mkdir();protocol=dict(variants=VARIANTS,alphas=ALPHAS,current_configs=CONFIGS,folds=pp['folds'],code_dimensions=dict(mu32=32,query1024=1024,query_pca32=32),pca='Exact float64 SVD of train-fold per-coordinate standardized Qxy-Qxx; top32 scores; downstream code-kernel train-standardization effectively whitens retained scores. PCA fit without labels; full512 PCA only after selection.',kernel='Same spatial current kernels and additive linear code kernel; each component and total training-centered/unit-mean-diagonal. Divide raw code Gram by its dimension before normalization; total strength1 for all.',selection='All candidates/alphas train-only5episode CV; all choices saved before observed64/96 evaluation; no further grid extension.',target_scaling=pp['target_training_scaling'],common_evaluation_scale='Exact old128 target scale from prior512 fit artifact',right_gripper_bins=bins,source_sha256=sources,mu_reproduction_max_error=mu_errors,mu_reproduction_note='Train512/dev64 exact; observed96 CPU2 medium/B64 differs <=2e-4 from original CPU4 export (original recorded exact). Existing cached mu used unchanged. Numerical audit in sibling query-accessibility512-head-reproduction.json.',script_sha256=sha256(Path(__file__)),shared_kernel_source_sha256=sha256(ROOT/'research/idm/probe_spatial_current.py'),threads=2,interpretation='Finite-readout accessibility comparison, not information-theoretic proof that mu lost information. Raw1024 has different dimension; PCA32 matches output dimension but linear subspace/nonlinear learned mu/training objective differ. Already observed evaluation sets, no independent finaltest. No new labels, images or world-model training.')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');started=time.perf_counter();allidx=torch.arange(512);cv={v:{c:torch.empty(len(ALPHAS),512,dtype=torch.float64) for c in CONFIGS} for v in VARIANTS};pcas=[];norm_checks=[]
 for fi,held in enumerate(folds):
  keep=allidx[~torch.isin(allidx,held)];g,cross,norm,_=geometry(images['train'][keep],states['train'][keep],images['train'][held],states['train'][held]);pca=pca_fit(queries['train'][keep]);pcas.append(pca);traincode=dict(mu32=codes['train'][keep],query1024=queries['train'][keep],query_pca32=pca_apply(queries['train'][keep],pca));heldcode=dict(mu32=codes['train'][held],query1024=queries['train'][held],query_pca32=pca_apply(queries['train'][held],pca));ym,ys=scaler(ysplit['train'][keep],FLOOR);target=(ysplit['train'][keep]-ym)/ys
  for name,mult in CONFIGS.items():
   for variant in VARIANTS:
    k,q,_=combine(g,cross,norm,traincode[variant],heldcode[variant],mult);assert abs(float(k.diag().mean())-1)<1e-12;pred=solve_many(k,q,target,ALPHAS)
    for j,p in enumerate(pred):cv[variant][name][j,held]=(p-(ysplit['train'][held]-ym)/ys).square().mean(-1)
  print('CV fold',fi+1,'complete',flush=True)
 selection={}
 for variant in VARIANTS:
  candidates=[dict(config=c,alpha=a,cv_normalized_mse=float(cv[variant][c][j].mean()),per_example=cv[variant][c][j].tolist()) for c in CONFIGS for j,a in enumerate(ALPHAS)]
  if variant=='mu32':
   for previous in ps['reference42']['candidates']:
    matched=next(v for v in candidates if v['config']==previous['config'] and v['alpha']==previous['alpha']);assert abs(matched['cv_normalized_mse']-previous['cv_normalized_mse'])<1e-12
  best=min(candidates,key=lambda v:v['cv_normalized_mse']);selection[variant]=dict(selected=best,candidates=candidates,alpha_at_boundary=best['alpha'] in (ALPHAS[0],ALPHAS[-1]),selected_by_config={c:min([v for v in candidates if v['config']==c],key=lambda v:v['cv_normalized_mse']) for c in CONFIGS})
 (out/'train_selection.json').write_text(json.dumps(selection,indent=2)+'\n');fullpca=pca_fit(queries['train']);names=['train','validation','observed96'];qi=torch.cat([images[s] for s in names]);qs=torch.cat([states[s] for s in names]);g,cross,norm,scalers=geometry(images['train'],states['train'],qi,qs);allq=torch.cat([queries[s] for s in names]);trcode=dict(mu32=codes['train'],query1024=queries['train'],query_pca32=pca_apply(queries['train'],fullpca));allcode=dict(mu32=torch.cat([codes[s] for s in names]),query1024=allq,query_pca32=pca_apply(allq,fullpca));ym,ys=scaler(ysplit['train'],FLOOR);target=(ysplit['train']-ym)/ys;results={};models={}
 for variant in VARIANTS:
  configurations={}
  for name,mult in CONFIGS.items():
   alpha=selection[variant]['selected_by_config'][name]['alpha'];k,q,normalization=combine(g,cross,norm,trcode[variant],allcode[variant],mult);prediction=solve_many(k,q,target,[alpha])[0]*ys+ym;start=0;evaluation={}
   for s in names:
    n=len(data[s]['rows']);pred=prediction[start:start+n];y=ysplit[s];sq=(pred-y).square();common=sq/common_scale.square();own=sq/ys.square();evaluation[s]=dict(normalized_mse_old128_scale=float(common.mean()),normalized_mse_train512_scale=float(own.mean()),physical_coordinate_mse=sq.mean(0).tolist(),right_gripper=gripper_metrics(y,pred,bins['static'],bins['small_large']),rows=[dict(row,prediction=pred[i].tolist(),target=y[i].tolist(),normalized_mse_old128_scale=float(common[i].mean()),physical_coordinate_squared_error=sq[i].tolist()) for i,row in enumerate(data[s]['rows'])]);start+=n
   configurations[name]=dict(alpha=alpha,evaluation=evaluation);models[variant+'_'+name]=dict(kernel=k,normalization=normalization,alpha=alpha)
  chosen=selection[variant]['selected']['config'];results[variant]=dict(selected_config=chosen,selected_alpha=selection[variant]['selected']['alpha'],selected_evaluation=configurations[chosen]['evaluation'],all_prespecified_configurations=configurations);print(variant,chosen,selection[variant]['selected']['alpha'],{s:v['normalized_mse_old128_scale'] for s,v in results[variant]['selected_evaluation'].items()},flush=True)
 for s in names:assert abs(results['mu32']['selected_evaluation'][s]['normalized_mse_old128_scale']-pr['results']['reference42']['selected_evaluation'][s]['normalized_mse_old128_scale'])<1e-12
 for path,digest in sources.items():assert sha256(Path(path))==digest
 torch.save(dict(fold_pcas=pcas,fulltrain_pca=fullpca,current_scalers=scalers,target_mean512=ym,target_scale512=ys,common_target_scale128=common_scale,models=models),out/'fit_artifacts.pt');report=dict(protocol_sha256=sha256(out/'protocol.json'),selection_sha256=sha256(out/'train_selection.json'),fit_artifacts_sha256=sha256(out/'fit_artifacts.pt'),mu32_previous_cv_and_eval_reproduced=True,pca_explained_variance=dict(folds=[v['explained_variance_fraction'] for v in pcas],full512=fullpca['explained_variance_fraction']),seconds=time.perf_counter()-started,results=results);(out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
