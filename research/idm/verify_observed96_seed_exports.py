"""Read-only seed export audit against immutable observed96 source."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.cache_reference_queries import sha256
@torch.inference_mode()
def main():
 torch.set_num_threads(2);source=ROOT/'runs/research-unused-96-prior6501';m=json.loads((source/'cache_manifest.json').read_text());data=torch.load(source/'validation.pt',weights_only=True);assert sha256(source/'validation.pt')==m['cache_sha256']['validation']
 qdir=ROOT/'research/idm/observed96-reference-queries2900';qm=json.loads((qdir/'manifest.json').read_text());q=torch.load(qdir/'validation.pt',weights_only=True);assert q['rows']==data['rows'] and sha256(qdir/'validation.pt')==qm['query_sha256'];assert qm['head42_reproduction_max_error']==0
 checks={}
 for seed in (43,44):
  path=ROOT/f'runs/research-observed96-idm{seed}-prior6501';meta=json.loads((path/'cache_manifest.json').read_text());d=torch.load(path/'validation.pt',weights_only=True)
  assert sha256(path/'validation.pt')==meta['cache_sha256']['validation'];assert meta['export']['seed']==seed and meta['export']['steps']==2000
  assert sha256(Path(meta['head_checkpoint']))==meta['head_checkpoint_sha256'];assert meta['calibration_sha256']==qm['calibration_sha256'];assert meta['observed96_export']['source_cache_sha256']==m['cache_sha256']['validation']
  assert set(d)==set(data)
  for k in data:
   if k=='codes':continue
   if isinstance(data[k],torch.Tensor):assert torch.equal(data[k],d[k])
   else:assert data[k]==d[k]
  assert d['codes'].shape==data['codes'].shape and torch.isfinite(d['codes']).all()
  checks[seed]=dict(cache_sha256=meta['cache_sha256']['validation'],count=len(d['rows']),all_noncode_fields_preserved=True,finite_codes=True)
 print(json.dumps(dict(all_checks_passed=True,head42_reproduction_max_error=qm['head42_reproduction_max_error'],query_sha256=qm['query_sha256'],exports=checks,verification_script_sha256=sha256(Path(__file__))),indent=2))
if __name__=='__main__':main()
