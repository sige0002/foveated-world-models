"""Float64 posterior separation on the saved highest-precision eight queries."""
from pathlib import Path
import sys,json,hashlib
import torch
from torch.nn import functional as F
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.cache_reference_queries import sha256
@torch.inference_mode()
def main():
 torch.set_num_threads(2)
 q=torch.load(ROOT/'research/idm/mature-transmission-step6501.pt',weights_only=True)['highest']
 weights=ROOT/'runs/frozen-idm-step6501/reference_weights.pt';s=torch.load(weights,weights_only=True,mmap=True)['state_dict']
 def encode(x):
  def lin(x,name):return F.linear(x,s['lam.'+name+'.weight'].double(),s['lam.'+name+'.bias'].double())
  p=lin(x.double(),'encoder.out_proj');n=F.layer_norm(p,(32,),s['lam.vq.pre_norm.weight'].double(),s['lam.vq.pre_norm.bias'].double(),1e-5)
  return lin(n,'vq.mu'),lin(n,'vq.logvar').clamp(-10,10)
 m,l=encode(q['query']);mm,ll=encode(q['swapped']);d=(m-mm).square()
 # Average of the two directed Gaussian KLs, summed over code coordinates.
 sym=.25*((l-ll).expm1()+(ll-l).expm1()+d*((-l).exp()+(-ll).exp())).sum(-1)
 out=dict(symmetric_KL_per_pair=sym.flatten().tolist(),symmetric_KL_mean=float(sym.mean()),posterior_variance_mean=float(l.exp().mean()),future_mu_difference_posterior_mahalanobis_mean=float((d*(-l).exp()).sum(-1).mean()),query_sha256=sha256(ROOT/'research/idm/mature-transmission-step6501.pt'),weights_sha256=sha256(weights),script_sha256=sha256(Path(__file__)),precision='float64 head on highest float32 encoder Q; CPU2',definition='0.5*(KL(q_true||q_swapped)+KL(q_swapped||q_true)); dependence diagnostic, no action meaning asserted')
 path=ROOT/'research/idm/mature-gaussian-step6501.json'
 if path.exists():raise FileExistsError(path)
 path.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
