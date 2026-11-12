"""Integrity and alignment checks; no fitting or label-distribution selection."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.extract_observed_displacements import SELECT
from research.idm.cache_reference_queries import sha256

def main():
 torch.set_num_threads(2);base=ROOT/'research/idm/observed-displacement-labels';m=json.loads((base/'manifest.json').read_text());checks={}
 for split,source in m['source_caches'].items():
  path=base/f'{split}.pt';assert sha256(path)==m['output_sha256'][path.name]
  d=torch.load(path,weights_only=True);original=torch.load(Path(source['cache'])/f"{source['split']}.pt",weights_only=True)
  assert len(d['rows'])==len(original['rows'])
  for r,s in zip(d['rows'],original['rows']):
   for key,value in s.items():assert r[key]==value
  torch.testing.assert_close(d['delta8'],d['endpoint_states20'][:,1,SELECT]-d['endpoint_states20'][:,0,SELECT],atol=0,rtol=0,equal_nan=True)
  assert torch.equal(d['current_state20'],d['endpoint_states20'][:,0])
  assert d['delta8_mask'].all() and d['original_valid'].all() and d['endpoint_state_observed_mask'].all()
  assert d['structural_state_mask32'][:,:,:20].all() and not d['structural_state_mask32'][:,:,20:].any()
  assert torch.isfinite(d['current_state20']).all() and torch.isfinite(d['delta8']).all()
  checks[split]=dict(count=len(d['rows']),all_original_endpoints_observed=True,valid_delta_coordinates=int(d['delta8_mask'].sum()),current_state_shape=list(d['current_state20'].shape),delta_shape=list(d['delta8'].shape))
 print(json.dumps(dict(all_checks_passed=True,splits=checks,manifest_sha256=sha256(base/'manifest.json'),script_sha256=sha256(Path(__file__)),fitting=False),indent=2))
if __name__=='__main__':main()
