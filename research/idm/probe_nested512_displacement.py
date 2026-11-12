"""Nested512 fixed-code data-scale probe; train-only CV and common128 target metric."""
import json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from vendor.dataset.robomind import RoboMINDDataset
from research.idm.probe_spatial_current import geometry,combined_kernel,solve_many,CONFIGS
from research.idm.probe_displacement_ridge import scaler,FLOOR,sha256
ALPHAS=[1e-8,1e-7,1e-6,1e-5,1e-4,1e-3,.01,.1,1.,10.,100.,1000.,10000.]
VARIANTS=['zero','reference42'];SELECT=[0,1,2,9,10,11,12,19]
def gripper_metrics(y,p,tau,boundary):
 yy=y[:,7];pp=p[:,7];bins=dict(all=torch.ones(len(y),dtype=torch.bool),static=yy.abs()<=tau,small=(yy.abs()>tau)&(yy.abs()<=boundary),large=yy.abs()>boundary,large_positive=yy>boundary,large_negative=yy< -boundary)
 result={}
 for name,mask in bins.items():
  n=int(mask.sum())
  if not n:result[name]=dict(n=0);continue
  a,b=yy[mask],pp[mask];e=b-a;result[name]=dict(n=n,mse=float(e.square().mean()),mae=float(e.abs().mean()),bias_prediction_minus_target=float(e.mean()),target_mean=float(a.mean()),prediction_mean=float(b.mean()),target_abs_mean=float(a.abs().mean()),prediction_abs_mean=float(b.abs().mean()),predicted_movement=int((b.abs()>tau).sum()),wrong_sign_on_active=int(((a.abs()>tau)&(b.abs()>tau)&(a*b<0)).sum()),miss_on_active=int(((a.abs()>tau)&(b.abs()<=tau)).sum()),magnitude_underprediction_on_active=int(((a.abs()>tau)&(b.abs()<a.abs()-tau)).sum()))
 return result
@torch.inference_mode()
def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');out=ROOT/'research/idm/nested512-displacement-probe'
 if out.exists():raise FileExistsError(out)
 cache=ROOT/'runs/reference-idm42-nested512-prior10000';meta=json.loads((cache/'cache_manifest.json').read_text());selpath=Path(meta['selection_path']);selection=json.loads(selpath.read_text());assert sha256(selpath)==meta['selection_sha256'];sources={str(selpath):sha256(selpath),str(cache/'cache_manifest.json'):sha256(cache/'cache_manifest.json')};data={}
 for split in ('train','validation'):
  path=cache/f'{split}.pt';assert sha256(path)==meta['cache_sha256'][split];sources[str(path)]=sha256(path);data[split]=torch.load(path,weights_only=True,mmap=True)
 opath=ROOT/'runs/research-unused-96-prior6501/validation.pt';om=json.loads((opath.parent/'cache_manifest.json').read_text());assert sha256(opath)==om['cache_sha256']['validation'];sources[str(opath)]=sha256(opath);data['observed96']=torch.load(opath,weights_only=True,mmap=True)
 labeldir=ROOT/'research/idm/observed-displacement-labels';lm=json.loads((labeldir/'manifest.json').read_text());oldlabels={}
 for split in ('train','validation','observed96'):
  path=labeldir/f'{split}.pt';assert sha256(path)==lm['output_sha256'][path.name];sources[str(path)]=sha256(path);oldlabels[split]=torch.load(path,weights_only=True)
 dataset=RoboMINDDataset(selection['manifest'],**selection['options']);assert sha256(Path(selection['manifest']))==selection['manifest_sha256'];assert data['train']['rows']==selection['retained']+selection['added'];labels=[];states=[];array_hashes={}
 for j,row in enumerate(data['train']['rows']):
  mi,a,b,_=dataset.pairs[row['index']];entry=dataset.rows[mi];arr=dataset._arrays(mi);assert str(entry['episode_id'])==row['episode'] and abs(float(arr['timestamps'][b]-arr['timestamps'][a])-row['dt'])<1e-9
  if 'endpoint_indices' in row:assert row['endpoint_indices']==[a,b]
  assert arr['original_valid'][[a,b]].all() and not (arr['fill_age_seconds'][[a,b]]>0).any();st=arr['states'][[a,b]];states.append(torch.from_numpy(st[0].copy()));labels.append(torch.from_numpy((st[1,SELECT]-st[0,SELECT]).copy()));ap=dataset.path.parent/entry['array_file'];array_hashes[str(ap)]=sha256(ap)
 ytrain=torch.stack(labels).double();strain=torch.stack(states).float();assert torch.equal(ytrain[:128],oldlabels['train']['delta8'].double()) and torch.equal(strain[:128],oldlabels['train']['current_state20'])
 oldbase=ROOT/'research/idm/spatial-current-boundary-audit';oldr=json.loads((oldbase/'results.json').read_text())['results'];oldprotocol=json.loads((oldbase/'protocol.json').read_text());oldfit=torch.load(oldbase/'kernel_fit_artifacts.pt',weights_only=True);common_scale=oldfit['target_scale'];sources[str(oldbase/'results.json')]=sha256(oldbase/'results.json');sources[str(oldbase/'kernel_fit_artifacts.pt')]=sha256(oldbase/'kernel_fit_artifacts.pt')
 images={};currentstates={};codes={};ysplit={}
 for split,d in data.items():
  n=len(d['rows']);images[split]=d['current'].flatten(1).float();currentstates[split]=strain if split=='train' else oldlabels[split]['current_state20'].float();ysplit[split]=ytrain if split=='train' else oldlabels[split]['delta8'].double();codes[split]=dict(zero=torch.zeros(n,32,dtype=torch.float64),reference42=d['codes'].reshape(n,32).double())
  if split!='train':
   for row,lab in zip(d['rows'],oldlabels[split]['rows']):assert all(row[k]==lab[k] for k in ('episode','index','dt'))
  assert torch.isfinite(images[split]).all() and torch.isfinite(currentstates[split]).all() and torch.isfinite(ysplit[split]).all()
 train_episodes={r['episode'] for r in data['train']['rows']};evaluation_episodes={r['episode'] for s in ('validation','observed96') for r in data[s]['rows']};assert len(train_episodes)==512 and len(evaluation_episodes)==160 and not train_episodes&evaluation_episodes
 rng=np.random.default_rng(20261113);added=np.arange(128,512);rng.shuffle(added);folds=[torch.tensor(old+new.tolist(),dtype=torch.long) for old,new in zip(oldprotocol['folds'],np.array_split(added,5))];assert sorted(torch.cat(folds).tolist())==list(range(512));assert all([i for i in f.tolist() if i<128]==old for f,old in zip(folds,oldprotocol['folds']))
 binpath=ROOT/'research/idm/right-gripper-error-analysis/protocol.json';bins=json.loads(binpath.read_text());tau=bins['static_threshold'];boundary=bins['small_large_boundary']
 protocol=dict(train_size=512,retained128_exact_labels_states=True,source_sha256=sources,npz_sha256=array_hashes,head_training_pairs=128,head_calibration_refit=False,alphas=ALPHAS,current_configs=CONFIGS,variants=VARIANTS,folds=[f.tolist() for f in folds],fold_rule='Preserve128 old fold assignments; seed20261113 permutation of added384 indices split into5 groups, appended to corresponding folds',normalization=oldprotocol['normalization'],feature_scaling='Per-coordinate train-fold population std floor1e-6; current image/state groups equally weighted; linear code component; exact old kernel functions',target_training_scaling='Fit on each512-fold train subset, floors .001/.01; full512 refit after selection',evaluation_scaling='Report both full512 target scales and fixed old128 target scales; comparisons across dataset sizes use old128 scale',bin_thresholds=dict(static=tau,small_large=boundary),bin_protocol_sha256=sha256(binpath),selection='All5fold candidates/alphas selected on train labels only, all choices saved before evaluation64/96 prediction; no subsequent grid extension',threads=2,script_sha256=sha256(Path(__file__)),kernel_source_sha256=sha256(ROOT/'research/idm/probe_spatial_current.py'),scope='Already research-observed64/96, no independent finaltest; frozen future-conditioned code; only probe training, no world-model training or GPU')
 out.mkdir();(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');torch.save(dict(delta8=ytrain,current_state20=strain,delta8_mask=torch.ones(512,8,dtype=torch.bool),current_state_mask=torch.ones(512,20,dtype=torch.bool),rows=data['train']['rows']),out/'train_labels.pt');started=time.perf_counter();allidx=torch.arange(512);cv={v:{c:torch.empty(len(ALPHAS),512,dtype=torch.float64) for c in CONFIGS} for v in VARIANTS}
 for fi,held in enumerate(folds):
  keep=allidx[~torch.isin(allidx,held)];g,cross,norm,_=geometry(images['train'][keep],currentstates['train'][keep],images['train'][held],currentstates['train'][held]);ym,ys=scaler(ytrain[keep],FLOOR);target=(ytrain[keep]-ym)/ys
  for name,mult in CONFIGS.items():
   for variant in VARIANTS:
    k,q,_=combined_kernel(g,cross,norm,codes['train'][variant][keep],codes['train'][variant][held],mult);pred=solve_many(k,q,target,ALPHAS)
    for j,prediction in enumerate(pred):cv[variant][name][j,held]=(prediction-(ytrain[held]-ym)/ys).square().mean(-1)
  print('CV fold',fi+1,'complete',flush=True)
 selection_result={}
 for variant in VARIANTS:
  candidates=[dict(config=c,alpha=a,cv_normalized_mse=float(cv[variant][c][j].mean()),per_example=cv[variant][c][j].tolist()) for c in CONFIGS for j,a in enumerate(ALPHAS)];best=min(candidates,key=lambda c:c['cv_normalized_mse']);selection_result[variant]=dict(selected=best,candidates=candidates,alpha_at_boundary=best['alpha'] in (ALPHAS[0],ALPHAS[-1]),selected_by_config={c:min([v for v in candidates if v['config']==c],key=lambda v:v['cv_normalized_mse']) for c in CONFIGS})
 (out/'train_selection.json').write_text(json.dumps(selection_result,indent=2)+'\n')
 # All model choices fixed before evaluating previously observed sets.
 names=['train','validation','observed96'];qi=torch.cat([images[s] for s in names]);qs=torch.cat([currentstates[s] for s in names]);g,cross,norm,scalers=geometry(images['train'],currentstates['train'],qi,qs);ym,ys=scaler(ytrain,FLOOR);target=(ytrain-ym)/ys;results={};artifacts={}
 for variant in VARIANTS:
  zq=torch.cat([codes[s][variant] for s in names]);config_results={}
  for name,mult in CONFIGS.items():
   alpha=selection_result[variant]['selected_by_config'][name]['alpha'];k,q,kn=combined_kernel(g,cross,norm,codes['train'][variant],zq,mult);prediction=solve_many(k,q,target,[alpha])[0]*ys+ym;start=0;metrics={}
   for split in names:
    n=len(data[split]['rows']);pred=prediction[start:start+n];y=ysplit[split];sq=(pred-y).square();common=sq/common_scale.square();own=sq/ys.square();metrics[split]=dict(normalized_mse_old128_scale=float(common.mean()),normalized_mse_train512_scale=float(own.mean()),physical_coordinate_mse=sq.mean(0).tolist(),right_gripper=gripper_metrics(y,pred,tau,boundary),rows=[dict(row,prediction=pred[i].tolist(),target=y[i].tolist(),normalized_mse_old128_scale=float(common[i].mean()),normalized_mse_train512_scale=float(own[i].mean()),physical_coordinate_squared_error=sq[i].tolist()) for i,row in enumerate(data[split]['rows'])]);start+=n
   config_results[name]=dict(alpha=alpha,evaluation=metrics);artifacts[variant+'_'+name]=dict(kernel=k,alpha=alpha,normalization=kn)
  chosen=selection_result[variant]['selected']['config'];results[variant]=dict(selected_config=chosen,selected_alpha=selection_result[variant]['selected']['alpha'],selected_evaluation=config_results[chosen]['evaluation'],all_prespecified_configurations=config_results,old128_selected_evaluation=oldr[variant]['selected_evaluation']);print(variant,chosen,selection_result[variant]['selected']['alpha'],{s:v['normalized_mse_old128_scale'] for s,v in results[variant]['selected_evaluation'].items()},flush=True)
 for path,digest in sources.items():assert sha256(Path(path))==digest
 torch.save(dict(current_scalers=scalers,target_mean512=ym,target_scale512=ys,common_target_scale128=common_scale,models=artifacts),out/'fit_artifacts.pt');report=dict(protocol_sha256=sha256(out/'protocol.json'),selection_sha256=sha256(out/'train_selection.json'),labels_sha256=sha256(out/'train_labels.pt'),fit_artifacts_sha256=sha256(out/'fit_artifacts.pt'),seconds=time.perf_counter()-started,results=results)
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
