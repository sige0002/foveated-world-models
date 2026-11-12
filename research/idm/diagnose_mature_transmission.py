"""Fixed eight-row future sensitivity, precision and isotropic controls; no fitting."""
import json,sys
from pathlib import Path
import torch,yaml
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
from research.idm.cache_reference_queries import sha256
from latent_action_model.core.utils.lam_encoder import LAMEncoder

def energy(x):return float(x.double().square().sum(-1).mean())
@torch.inference_mode()
def main():
 torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
 cache=ROOT/'runs/frozen-idm-step6501';output=ROOT/'research/idm/mature-transmission-step6501.json'
 if output.exists():raise FileExistsError(output)
 data=torch.load(cache/'validation.pt',weights_only=True)
 cfg=yaml.safe_load((cache/'resolved_config.yaml').read_text())['model']
 enc=LAMEncoder(context_dim=cfg['dim'],input_dim=768,num_heads=cfg['num_heads'],num_queries=cfg['num_queries'],grid_hw=tuple(v//cfg['patch_size'] for v in cfg['image_hw']),ffn_expansion_factor=cfg['ffn_expansion_factor'],dropout=cfg['dropout'],code_dim=cfg['code_dim'],num_layers=cfg['enc_layers'],num_frames=cfg['num_frames'],modal_mask=cfg['enc_modal_mask'],add_state=False,max_state_dim=cfg['max_state_dim'],num_embodiments=cfg['num_embodiments']).eval().requires_grad_(False)
 saved=torch.load(cache/'reference_weights.pt',weights_only=True,mmap=True)['state_dict']
 prefix='lam.encoder.';enc.load_state_dict({k[len(prefix):]:v for k,v in saved.items() if k.startswith(prefix)},strict=True)
 names=['encoder.out_proj.weight','encoder.out_proj.bias','vq.pre_norm.weight','vq.pre_norm.bias','vq.mu.weight','vq.mu.bias','vq.logvar.weight','vq.logvar.bias']
 values=[saved['lam.'+k].clone() for k in names];del saved
 def head(q,double=False):
  W,b,g,beta,M,mb,V,vb=[v.double() for v in values] if double else values
  q=q.double() if double else q;p=F.linear(q,W,b);n=F.layer_norm(p,(32,),g,beta,1e-5)
  return dict(query=q,projection=p,layernorm=n,mu=F.linear(n,M,mb),logvar=F.linear(n,V,vb).clamp(-10,10))
 capture={};original=enc.out_proj.forward
 def forward(q):capture['q']=q.clone();return original(q)
 enc.out_proj.forward=forward
 indices=torch.linspace(0,len(data['current'])-1,8).long().tolist();queries={}
 for precision in ('highest','medium'):
  torch.set_float32_matmul_precision(precision);q=[];sw=[]
  for j,i in enumerate(indices):
   x=data['current'][i:i+1];enc(torch.cat((x,data['future'][i:i+1]),1));q.append(capture['q'])
   k=indices[(j+1)%len(indices)];enc(torch.cat((x,data['future'][k:k+1]),1));sw.append(capture['q'])
   print(precision,j+1,flush=True)
  queries[precision]=dict(query=torch.cat(q),swapped=torch.cat(sw))
 enc.out_proj.forward=original
 def batched_head(q,double=False):
  outputs=[head(row[None],double) for row in q]
  return {k:torch.cat([o[k] for o in outputs]) for k in outputs[0]}
 results={};states={}
 for name,qp,hp,double in [('highest_full','highest','highest',False),('medium_full','medium','medium',False),('medium_head_highest_query','highest','medium',False),('float64_head_highest_query','highest','highest',True)]:
  torch.set_float32_matmul_precision(hp);a=batched_head(queries[qp]['query'],double);b=batched_head(queries[qp]['swapped'],double);states[name]=a
  results[name]={k:dict(future_response_energy=energy(b[k]-a[k]),common_mean_energy=energy(a[k].mean(0,keepdim=True)),between_pair_variance=energy(a[k]-a[k].mean(0,keepdim=True))) for k in a}
 torch.set_float32_matmul_precision('highest');q=queries['highest']['query'];sw=queries['highest']['swapped'];a=batched_head(q,True);b=batched_head(sw,True)
 dq=(sw-q).double();gen=torch.Generator().manual_seed(713);random=torch.randn((64,)+tuple(q.shape),generator=gen,dtype=torch.float64);random=random/random.norm(dim=-1,keepdim=True)*dq.norm(dim=-1,keepdim=True)
 iso=head(q.double()[None]+random,True)
 controls={k:dict(actual_energy=energy(b[k]-a[k]),isotropic_mean_energy=energy(iso[k]-a[k][None])) for k in a}
 for v in controls.values():v['actual_over_isotropic']=v['actual_energy']/max(v['isotropic_mean_energy'],1e-30)
 p=a['projection'];dp=b['projection']-p;c=p-p.mean(-1,keepdim=True);dc=dp-dp.mean(-1,keepdim=True);scale=(c.square().mean(-1,keepdim=True)+1e-5).sqrt();radial=(dc*c).sum(-1,keepdim=True)/c.square().sum(-1,keepdim=True)*c
 linear=(dc/scale-c*(c*dc).mean(-1,keepdim=True)/scale.pow(3))*values[2].double()
 diagnostics=dict(projected_coordinate_std_mean=float(scale.mean()),delta_centered_energy=energy(dc),delta_radial_energy=energy(radial),delta_tangent_energy=energy(dc-radial),LN_jacobian_relative_error=energy(linear-(b['layernorm']-a['layernorm']))/max(energy(b['layernorm']-a['layernorm']),1e-30),medium_mu_cached_max_error=float((states['medium_full']['mu']-data['codes'][indices]).abs().max()))
 differences={name:{k:dict(max_abs=float((states[name][k].double()-states['float64_head_highest_query'][k]).abs().max()),difference_energy=energy(states[name][k].double()-states['float64_head_highest_query'][k])) for k in states[name]} for name in ('highest_full','medium_full','medium_head_highest_query')}
 tensor_path=output.with_suffix('.pt');torch.save(queries,tensor_path)
 report=dict(checkpoint=6501,indices=indices,rows=[data['rows'][i] for i in indices],threads=2,batch_size=1,torch_version=str(torch.__version__),source_cache_sha256=sha256(cache/'validation.pt'),weights_sha256=sha256(cache/'reference_weights.pt'),script_sha256=sha256(Path(__file__)),query_tensor_sha256=sha256(tensor_path),responses=results,isotropic_controls=controls,geometry=diagnostics,precision_differences=differences,limitations='Sensitivity only: cross-episode future swap may be OOD, not action counterfactual. Coordinate energies are not mutual information. Fixed eight uniformly spaced cached validation rows; no fitting, no vision extraction, no decoder evaluation. Full highest/medium differences include encoder arithmetic; fixed highest Q comparisons isolate head arithmetic. Isotropic control: 64 fixed seed713 directions per row at identical query perturbation norm; head evaluated float64 for geometric comparison.')
 output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['geometry'],indent=2))
if __name__=='__main__':main()
