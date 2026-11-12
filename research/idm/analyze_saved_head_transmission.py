"""Small saved-query analysis; no encoder run, gradient update, or decoder claim."""
from pathlib import Path
import json,sys
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2]
torch.set_num_threads(4);torch.set_float32_matmul_precision('medium')
cache=ROOT/'runs/frozen-idm-step2900'
qdata=torch.load(ROOT/'research/idm/reference-query-full/validation.pt',weights_only=True)
data=torch.load(cache/'validation.pt',weights_only=True)
saved=torch.load(cache/'reference_weights.pt',weights_only=True,mmap=True)['state_dict']
keys=['lam.encoder.out_proj.weight','lam.encoder.out_proj.bias',
      'lam.vq.pre_norm.weight','lam.vq.pre_norm.bias','lam.vq.mu.weight','lam.vq.mu.bias','lam.vq.logvar.weight','lam.vq.logvar.bias']
values=[saved[k].clone() for k in keys];del saved
W,b,gamma,beta,M,mb,V,vb=values
q=qdata['query'];other=qdata['future_swapped']
def forward(x,values):
 W,b,gamma,beta,M,mb,V,vb=values
 p=F.linear(x,W,b);normalized=F.layer_norm(p,(32,),gamma,beta,1e-5)
 return p,normalized,F.linear(normalized,M,mb),F.linear(normalized,V,vb).clamp(-10,10)
with torch.inference_mode():
 reproduction=torch.cat([forward(q[i:i+1],values)[2] for i in range(len(q))])
 maxerr=float((reproduction-data['codes']).abs().max())
 if maxerr>1e-5:raise ValueError(f'Mean reproduction mismatch: {maxerr}')
 # Double arithmetic isolates head geometry from the original float32 numerical path.
 p,n,mu,lv=forward(q.double(),[v.double() for v in values])
 pp,nn,mm,ll=forward(other.double(),[v.double() for v in values])
 dq=other.double()-q.double();dp=pp-p
 c=p-p.mean(-1,keepdim=True);dc=dp-dp.mean(-1,keepdim=True)
 radial=(dc*c).sum(-1,keepdim=True)/c.square().sum(-1,keepdim=True)*c
 tangent=dc-radial
 scale=(c.square().mean(-1,keepdim=True)+1e-5).sqrt()
 linear_ln=(dc/scale-c*(c*dc).mean(-1,keepdim=True)/scale.pow(3))*gamma.double()
 def energy(x):return float(x.square().sum(-1).mean())
 expected_random=energy(dq)*float(W.double().square().sum())/W.shape[1]
 out=dict(checkpoint=2900,pairs=len(q),mu_reproduction_max_error=maxerr,
     future_swap_query_energy=energy(dq),future_swap_projected_energy=energy(dp),
     isotropic_same_norm_projection_expected_energy=expected_random,
     actual_over_isotropic_projection_energy=energy(dp)/expected_random,
     projected_coordinate_std_mean=float(scale.mean()),
     projected_delta_coordinate_centered_energy=energy(dc),projected_delta_radial_energy=energy(radial),projected_delta_tangent_energy=energy(tangent),
     future_swap_layernorm_energy=energy(nn-n),future_swap_mu_energy=energy(mm-mu),future_swap_logvar_energy=energy(ll-lv),
     layernorm_first_order_relative_squared_error=energy(linear_ln-(nn-n))/max(energy(nn-n),1e-30),
     limitation='Saved step2900 queries only. Same-current cross-episode future swaps may be OOD. Energies depend on coordinate units, and projection/LN response does not identify action information. Step6501 encoder/head response was not measured.')
 path=ROOT/'research/idm/saved-head-transmission-step2900.json'
 if path.exists():raise FileExistsError(path)
 path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
