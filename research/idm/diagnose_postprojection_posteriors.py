"""Same-eight-current future swaps for four completed, matched Gaussian heads."""
import json,sys
from pathlib import Path
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.postprojection_heads import shared_initial_heads
from research.idm.postprojection_representations import representation
from research.idm.cache_reference_queries import sha256
@torch.inference_mode()
def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
 out=ROOT/'research/idm/postprojection-posterior-sensitivity400.json'
 if out.exists():raise FileExistsError(out)
 experiment=ROOT/'research/idm/postprojection-comparison400';er=json.loads((experiment/'results.json').read_text());query_path=ROOT/'research/idm/mature-transmission-step6501.pt';qr=json.loads((ROOT/'research/idm/mature-transmission-step6501.json').read_text());queries=torch.load(query_path,weights_only=True)['highest']
 assert sha256(query_path)==qr['query_tensor_sha256'];weights=ROOT/'runs/frozen-idm-step6501/reference_weights.pt';assert sha256(weights)==qr['weights_sha256']
 saved=torch.load(weights,weights_only=True,mmap=True)['state_dict'];w=saved['lam.encoder.out_proj.weight'].clone();b=saved['lam.encoder.out_proj.bias'].clone();del saved
 p={k:torch.cat([F.linear(row[None],w,b) for row in value]) for k,value in queries.items()}
 base=ROOT/'research/idm/mature-postprojection-cache';meta=json.loads((base/'manifest.json').read_text());calpath=base/'train_calibration.pt';assert sha256(calpath)==meta['output_sha256']['train_calibration.pt'];cal=torch.load(calpath,weights_only=True);val=torch.load(base/'validation.pt',weights_only=True)
 indices=qr['indices'];assert [val['rows'][i]['episode'] for i in indices]==[r['episode'] for r in qr['rows']]
 for i,row in zip(indices,qr['rows']):
  for key in ('episode','index','dt'):assert val['rows'][i][key]==row[key]
 assert torch.equal(queries['query'],val['query'][indices]);p_error=float((p['query']-val['projection'][indices]).abs().max());assert p_error<1e-5
 results={};head_hashes={};initial=None
 for result in er['results']:
  kind=result['variant'];path=ROOT/result['checkpoint'];assert sha256(path)==result['checkpoint_sha256'];head_hashes[kind]=sha256(path)
  ck=torch.load(path,weights_only=True);assert ck['steps']==400 and ck['seed']==42 and ck['calibration_sha256']==sha256(calpath)
  if initial is None:initial=result['fairness']
  assert result['fairness']==initial
  head=shared_initial_heads(42)['direction'].eval();head.load_state_dict({k[len('head.'):]:v for k,v in ck['state_dict'].items() if k.startswith('head.')},strict=True);del ck
  if kind=='constant':x={k:torch.zeros(len(v),1,34) for k,v in p.items()}
  else:x={k:representation(v,cal,kind) for k,v in p.items()}
  params={}
  for key,value in x.items():
   output=torch.cat([head(row[None]) for row in value]);mu,lv=output.chunk(2,-1);params[key]=(mu.double(),lv.clamp(-10,10).double())
  m,l=params['query'];mm,ll=params['swapped'];diff=(m-mm).square();sym=.25*((l-ll).expm1()+(ll-l).expm1()+diff*((-l).exp()+(-ll).exp())).sum(-1).flatten();mahal=(diff*(-l).exp()).sum(-1).flatten();bound=.5+.5*(sym.clamp_min(0)/2).sqrt().clamp_max(1)
  if kind=='constant':assert torch.equal(m,mm) and torch.equal(l,ll) and torch.equal(sym,torch.zeros_like(sym))
  rows=[dict(source_row=row,symmetric_kl=float(sym[j]),mu_squared_mahalanobis=float(mahal[j]),mu_change_energy=float(diff[j].sum()),logvar_change_energy=float((l[j]-ll[j]).square().sum()),single_sample_binary_accuracy_upper_bound=float(bound[j])) for j,row in enumerate(qr['rows'])]
  results[kind]=dict(rows=rows,symmetric_kl_mean=float(sym.mean()),mu_squared_mahalanobis_mean=float(mahal.mean()),mu_change_energy_mean=float(diff.sum(-1).mean()),logvar_change_energy_mean=float((l-ll).square().sum(-1).mean()),true_mu_between_pair_variance=float(m.var(0,unbiased=False).sum()),true_logvar_between_pair_variance=float(l.var(0,unbiased=False).sum()),true_posterior_variance_mean=float(l.exp().mean()),mean_single_sample_binary_accuracy_upper_bound=float(bound.mean()),constant_exact_equality_verified=kind=='constant')
 report=dict(results=results,indices=indices,p_reproduction_max_error=p_error,threads=2,precision='highest float32 projection/representation/head B1; float64 divergence arithmetic;logvar clamp[-10,10]',query_sha256=sha256(query_path),weights_sha256=sha256(weights),calibration_sha256=sha256(calpath),query_manifest_sha256=sha256(base/'manifest.json'),head_checkpoint_sha256=head_hashes,training_results_sha256=sha256(experiment/'results.json'),script_sha256=sha256(Path(__file__)),shared_fairness=initial,original6501_symmetric_kl_mean=.005624879497274804,pinsker_formula='.5+.5*min(1,sqrt(symmetric_KL/2)) per pair, then average',limitations='Balanced binary choice of actual versus cyclic-replaced future at each fixed current; single draw from diagonal Gaussian and no other future-dependent inputs. Bound not measured accuracy, not for deterministic mu/multiple draws. Future swap may be OOD, no action/camera semantics. Four adapted heads share training protocol; original6501 comparison differs in architecture/training history and is not causal same-condition comparison.')
 out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:{a:v for a,v in r.items() if a!='rows'} for k,r in results.items()},indent=2))
if __name__=='__main__':main()
