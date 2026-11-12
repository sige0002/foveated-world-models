"""Frozen6501 local decoder probes: no training and no prediction-quality claims."""
import json,sys,hashlib,time
from pathlib import Path
import torch,yaml
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from research.idm.cache_reference_queries import sha256
@torch.inference_mode()
def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
 cache=ROOT/'runs/frozen-idm-step6501';qpath=ROOT/'research/idm/mature-transmission-step6501.pt';out=ROOT/'research/idm/mature-decoder-gain-step6501.json'
 if out.exists():raise FileExistsError(out)
 previous=json.loads((ROOT/'research/idm/mature-transmission-step6501.json').read_text());hashes=dict(cache=sha256(cache/'validation.pt'),weights=sha256(cache/'reference_weights.pt'),query=sha256(qpath))
 assert hashes['cache']==previous['source_cache_sha256'] and hashes['weights']==previous['weights_sha256'] and hashes['query']==previous['query_tensor_sha256']
 data=torch.load(cache/'validation.pt',weights_only=True);q=torch.load(qpath,weights_only=True)['highest'];indices=previous['indices'];assert [data['rows'][i] for i in indices]==previous['rows']
 cfg=yaml.safe_load((cache/'resolved_config.yaml').read_text())['model'];saved=torch.load(cache/'reference_weights.pt',weights_only=True,mmap=True);assert saved['global_step']==6501;s=saved['state_dict']
 decoder=LAMDecoder_v2(context_dim=cfg['dim'],input_dim=768,code_dim=cfg['code_dim'],num_queries=cfg['num_queries'],num_layers=cfg['dec_layers'],num_heads=cfg['num_heads'],dropout=cfg['dropout'],grid_hw=tuple(v//cfg['patch_size'] for v in cfg['image_hw']),ffn_expansion_factor=cfg['ffn_expansion_factor'],last_ln=cfg.get('decoder_last_ln',True)).eval().requires_grad_(False)
 prefix='lam.decoder.';decoder.load_state_dict({k[len(prefix):]:v for k,v in s.items() if k.startswith(prefix)},strict=True)
 def encode(x):
  def lin(x,k):return F.linear(x,s['lam.'+k+'.weight'].double(),s['lam.'+k+'.bias'].double())
  p=lin(x.double(),'encoder.out_proj');n=F.layer_norm(p,(32,),s['lam.vq.pre_norm.weight'].double(),s['lam.vq.pre_norm.bias'].double(),1e-5)
  return lin(n,'vq.mu').float(),lin(n,'vq.logvar').clamp(-10,10).float()
 mu,lv=encode(q['query']);sm,sl=encode(q['swapped']);del saved,s
 gen=torch.Generator().manual_seed(1107);isotropic=torch.randn(mu.shape,generator=gen);isotropic=isotropic/isotropic.norm(dim=-1,keepdim=True)
 eps=torch.randn((2,)+tuple(mu.shape),generator=gen);rows=[];started=time.perf_counter()
 for j,i in enumerate(indices):
  h=data['current'][i:i+1];m,l,mm,ll=[x[j:j+1] for x in (mu,lv,sm,sl)];sd=(.5*l).exp();delta=mm-m;white=delta/sd;radius=white.norm(dim=-1,keepdim=True)
  if float(radius)==0:raise RuntimeError('Zero native direction; cannot normalize')
  iso=isotropic[j:j+1];directions=dict(native=delta,native_radius_isotropic=sd*iso*radius,radius01_native=delta*(.1/radius),radius01_isotropic=sd*iso*.1)
  pred=decoder(h,m);direction_results={}
  for name,d in directions.items():
   positive,negative=decoder(h,m+d),decoder(h,m-d);central=(positive-negative)/2
   zenergy=float(d.double().square().sum());output=float(central.double().square().mean())
   direction_results[name]=dict(latent_squared_norm=zenergy,posterior_mahalanobis_radius=float((d/sd).norm()),central_prediction_mean_square=output,gain_output_mse_per_latent_norm2=output/zenergy,positive_prediction_mean_square=float((positive-pred).double().square().mean()),negative_prediction_mean_square=float((negative-pred).double().square().mean()))
  # Fixed half radius in the same native direction: rounding/nonlinearity check only.
  d=directions['radius01_native']*.5;half=(decoder(h,m+d)-decoder(h,m-d))/2
  full=direction_results['radius01_native']['central_prediction_mean_square'];half_energy=float(half.double().square().mean())
  samples=[]
  for k in range(2):
   e=eps[k,j:j+1];base=decoder(h,m+sd*e)
   variants=dict(mu_only=mm+sd*e,logvar_only=m+(.5*ll).exp()*e,both=mm+(.5*ll).exp()*e)
   samples.append({name:dict(prediction_mean_square=float((decoder(h,z)-base).double().square().mean()),latent_squared_norm=float((z-(m+sd*e)).double().square().sum())) for name,z in variants.items()})
  rows.append(dict(source_row=data['rows'][i],index_in_cache=i,directions=direction_results,half_radius_native_central_mse=half_energy,half_over_quarter_full=half_energy/max(full*.25,1e-30),posterior_factor_swaps=samples))
  print('Completed',j+1,'/8',flush=True)
 summary={name:{key:sum(r['directions'][name][key] for r in rows)/len(rows) for key in rows[0]['directions'][name]} for name in directions}
 factor={name:sum(sample[name]['prediction_mean_square'] for r in rows for sample in r['posterior_factor_swaps'])/(len(rows)*2) for name in ('mu_only','logvar_only','both')}
 assert hashes==dict(cache=sha256(cache/'validation.pt'),weights=sha256(cache/'reference_weights.pt'),query=sha256(qpath))
 report=dict(checkpoint=6501,indices=indices,rows=rows,summary=summary,posterior_factor_swap_prediction_mse_mean=factor,source_hashes=hashes,script_sha256=sha256(Path(__file__)),decoder_source_sha256=sha256(ROOT/'vendor/lawam/latent_action_model/core/utils/lam_decoder.py'),eps_sha256=hashlib.sha256(eps.numpy().tobytes()).hexdigest(),isotropic_sha256=hashlib.sha256(isotropic.numpy().tobytes()).hexdigest(),precision='highest float32 decoder; float64 Gaussian head on saved highest float32 encoder Q, cast mu/logvar tofloat32',threads=2,batch_size=1,seconds=time.perf_counter()-started,torch_version=str(torch.__version__),limitations='Same-current future swaps may be OOD; no action/camera semantics. Isotropic in posterior-whitened coordinates, one seed1107 direction per example,2 common eps per example. Fixed radius0.1 can leave empirical mu cloud. Frozen local sensitivity only: no future-target errors, optimization or performance claim. Half-radius probe measures local scaling, not an independent decoder numerical precision bound.')
 out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(summary=summary,factors=factor),indent=2))
if __name__=='__main__':main()
