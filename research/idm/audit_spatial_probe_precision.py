"""Frozen selected hyperparameters, float64-geometry sensitivity; no reselection."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.probe_spatial_current import geometry,combined_kernel,solve_many,CONFIGS
from research.idm.probe_displacement_ridge import load,scaler,FLOOR,sha256
@torch.inference_mode()
def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');base=ROOT/'research/idm/spatial-current-boundary-audit';out=ROOT/'research/idm/spatial-current-numerical-audit'
 if out.exists():raise FileExistsError(out)
 prior=json.loads((base/'results.json').read_text());selection=json.loads((base/'train_selection.json').read_text());data,sources=load();labels=json.loads((ROOT/'research/idm/observed-displacement-labels/manifest.json').read_text());splitnames=['train','validation','observed96'];images={};states={};codes={}
 for s,d in data.items():
  src=labels['source_caches'][s];cached=torch.load(Path(src['cache'])/f"{src['split']}.pt",weights_only=True);images[s]=cached['current'].flatten(1).double();states[s]=d['x']['zero'][:,768:788];codes[s]={v:x[:,-32:] for v,x in d['x'].items()}
 protocol=dict(reason='Numerical sensitivity at boundary-selectedalpha1e-8; all selections fixed from prior train-only CV',selected={v:dict(config=r['selected_config'],alpha=r['selected_alpha']) for v,r in prior['results'].items()},prior_protocol_sha256=sha256(base/'protocol.json'),prior_selection_sha256=sha256(base/'train_selection.json'),geometry_change='All current-feature means/std/Gram computations float64 instead of float32; same normalization, kernels and selected hyperparameters',reselection=False,source_sha256=sources,threads=2,script_sha256=sha256(Path(__file__)))
 out.mkdir();(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');qi=torch.cat([images[s] for s in splitnames]);qs=torch.cat([states[s] for s in splitnames]);g,cross,norm,_=geometry(images['train'],states['train'],qi,qs);ym,ys=scaler(data['train']['y'],FLOOR);target=(data['train']['y']-ym)/ys;results={}
 for variant,chosen in protocol['selected'].items():
  zq=torch.cat([codes[s][variant] for s in splitnames]);k,q,_=combined_kernel(g,cross,norm,codes['train'][variant],zq,CONFIGS[chosen['config']]);eig=torch.linalg.eigvalsh((k+k.T)/2);alpha=chosen['alpha'];prediction=solve_many(k,q,target,[alpha])[0]*ys+ym;start=0;metrics={}
  for split in splitnames:
   n=len(data[split]['rows']);pred=prediction[start:start+n];old=prior['results'][variant]['selected_evaluation'][split];oldpred=torch.tensor([r['prediction'] for r in old['rows']],dtype=torch.float64);mse=float(((pred-data[split]['y'])/ys).square().mean());metrics[split]=dict(float64_geometry_normalized_mse=mse,float32_geometry_normalized_mse=old['normalized_mse'],normalized_mse_change=mse-old['normalized_mse'],max_abs_prediction_change=float((pred-oldpred).abs().max()),prediction_change_normalized_mse=float(((pred-oldpred)/ys).square().mean()));start+=n
  oldart=torch.load(base/'kernel_fit_artifacts.pt',weights_only=True)['models'][variant+'_'+chosen['config']]['kernel'];oldeig=torch.linalg.eigvalsh((oldart+oldart.T)/2)
  def summary(e):
   positive=e[e>1e-10]
   return dict(min_eigenvalue=float(e.min()),max_eigenvalue=float(e.max()),negative_eigenvalues_clipped=int((e<0).sum()),near_zero_abs_below1e10=int((e.abs()<1e-10).sum()),regularized_condition_after_clipping=float((e.max()+alpha)/(e.min().clamp_min(0)+alpha)),positive_subspace_condition=float(positive.max()/positive.min()))
  results[variant]=dict(chosen=chosen,float32_gram_spectrum=summary(oldeig),float64_gram_spectrum=summary(eig),evaluation=metrics)
 report=dict(protocol_sha256=sha256(out/'protocol.json'),results=results,interpretation='Centered kernels have a constant-vector nullspace, so regularized full condition number is large atalpha1e-8. Report positive-subspace conditioning and actual prediction sensitivity too. No hyperparameter or model selection changed using these evaluations.')
 (out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
