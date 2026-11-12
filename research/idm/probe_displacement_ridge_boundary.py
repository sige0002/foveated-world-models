"""One prespecified train-CV-boundary audit, keeping all original results immutable."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.probe_displacement_ridge import load,fit,predict,fold_indices,scaler,FLOOR,VARIANTS,sha256
ALPHAS=[.01,.1,1.,10.,100.,1000.,10000.,100000.,1000000.]
@torch.inference_mode()
def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
 old=ROOT/'research/idm/displacement-ridge-probe';out=ROOT/'research/idm/displacement-ridge-boundary-audit'
 if out.exists():raise FileExistsError(out)
 op=json.loads((old/'protocol.json').read_text());orr=json.loads((old/'results.json').read_text());data,sources=load();assert sources==op['source_sha256']
 episodes=[str(r['episode']) for r in data['train']['rows']];folds=fold_indices(episodes);assert [f.tolist() for f in folds]==op['folds']
 protocol=dict(op,alphas=ALPHAS,script_sha256=sha256(Path(__file__)),base_script_sha256=sha256(ROOT/'research/idm/probe_displacement_ridge.py'),original_protocol_sha256=sha256(old/'protocol.json'),original_results_sha256=sha256(old/'results.json'),reason='Original train-only CV selected the upper grid boundary1000 for zero and original6501. One shared-grid extension is prespecified; no validation-based selection and no further extension within this audit.',evaluation_order='Complete and save all three train-only selections before evaluating validation64/observed96')
 out.mkdir();(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
 train=data['train'];allidx=torch.arange(128);selection={};models={}
 for variant in VARIANTS:
  cv=[]
  for alpha in ALPHAS:
   losses=torch.empty(128,dtype=torch.float64)
   for held in folds:
    keep=allidx[~torch.isin(allidx,held)];m=fit(train['x'][variant][keep],train['y'][keep],alpha)
    losses[held]=((predict(m,train['x'][variant][held])-train['y'][held])/m['y_scale']).square().mean(-1)
   score=float(losses.mean());previous=next((v for v in orr['results'][variant]['cv'] if v['alpha']==alpha),None)
   if previous is not None:assert abs(score-previous['normalized_mse'])<1e-13
   cv.append(dict(alpha=alpha,normalized_mse=score,per_example=losses.tolist(),original_grid_score=None if previous is None else previous['normalized_mse']))
  chosen=min(cv,key=lambda v:v['normalized_mse'])['alpha'];selection[variant]=dict(selected_alpha=chosen,upper_boundary_selected=chosen==ALPHAS[-1],cv=cv);models[variant]=fit(train['x'][variant],train['y'],chosen)
 (out/'train_selection.json').write_text(json.dumps(selection,indent=2)+'\n')
 # No validation/observed96 prediction has occurred before all selections are saved.
 ym,ys=scaler(train['y'],FLOOR);results={}
 for variant in VARIANTS:
  evaluation={}
  for split,d in data.items():
   pred=predict(models[variant],d['x'][variant]);sq=(pred-d['y']).square();norm=sq/ys.square()
   evaluation[split]=dict(normalized_mse=float(norm.mean()),physical_coordinate_mse=sq.mean(0).tolist(),normalized_coordinate_mse=norm.mean(0).tolist(),rows=[dict(row,prediction=pred[i].tolist(),target=d['y'][i].tolist(),physical_coordinate_squared_error=sq[i].tolist(),normalized_mse=float(norm[i].mean())) for i,row in enumerate(d['rows'])])
  results[variant]=dict(**selection[variant],evaluation=evaluation,original_selected_alpha=orr['results'][variant]['selected_alpha'],original_evaluation_metrics={s:orr['results'][variant]['evaluation'][s]['normalized_mse'] for s in data})
  print(variant,selection[variant]['selected_alpha'],{s:v['normalized_mse'] for s,v in evaluation.items()},flush=True)
 for path,digest in sources.items():assert sha256(Path(path))==digest
 assert sha256(old/'results.json')==protocol['original_results_sha256'] and sha256(old/'protocol.json')==protocol['original_protocol_sha256']
 torch.save(dict(models=models,fulltrain_target_mean=ym,fulltrain_target_scale=ys),out/'fitted_models.pt')
 report=dict(protocol_sha256=sha256(out/'protocol.json'),train_selection_sha256=sha256(out/'train_selection.json'),models_sha256=sha256(out/'fitted_models.pt'),original_results_unchanged=True,original_grid_scores_reproduced=True,results=results,physical_mse_units=orr['physical_mse_units'])
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
