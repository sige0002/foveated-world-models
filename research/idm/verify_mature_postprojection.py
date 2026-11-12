"""Read-only extraction/provenance/calibration checks after completion."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.postprojection_representations import fit_calibration,representation,VARIANTS,decompose
from research.idm.cache_reference_queries import sha256
@torch.inference_mode()
def main():
 torch.set_num_threads(2)
 base=ROOT/'research/idm/mature-postprojection-cache';manifest=json.loads((base/'manifest.json').read_text())
 for name,digest in manifest['output_sha256'].items():assert sha256(base/name)==digest
 train=torch.load(base/'train.pt',weights_only=True);val=torch.load(base/'validation.pt',weights_only=True);cal=torch.load(base/'train_calibration.pt',weights_only=True)
 fitted=fit_calibration(train['projection'])
 for k,v in fitted.items():
  if isinstance(v,torch.Tensor):assert torch.equal(v,cal[k])
  else:assert v==cal[k]
 errors={}
 for split,data in [('train',train),('validation',val)]:
  for name in VARIANTS:assert torch.equal(representation(data['projection'],cal,name),data['representations'][name])
  p=data['projection'];u,m,l=decompose(p);err=float((m+l.exp()*u-p).abs().max());errors[split]=err
  assert err<=32*torch.finfo(p.dtype).eps*float(p.abs().max())
  for key in ['query','projection']:assert torch.isfinite(data[key]).all()
 result=dict(all_checks_passed=True,counts={s:len(d['rows']) for s,d in [('train',train),('validation',val)]},reconstruction_max_abs_error=errors,calibration_exactly_train_refit=True,representations_exactly_reproduced=True,manifest_sha256=sha256(base/'manifest.json'),verification_script_sha256=sha256(Path(__file__)))
 print(json.dumps(result,indent=2))
if __name__=='__main__':main()
