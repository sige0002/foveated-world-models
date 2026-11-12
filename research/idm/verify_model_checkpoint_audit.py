"""Read-only snapshot/cached-feature/baseline integrity checks for a completed audit."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.cache_reference_queries import sha256
@torch.inference_mode()
def main():
 p=argparse.ArgumentParser();p.add_argument('--step',type=int,required=True);a=p.parse_args();torch.set_num_threads(2)
 base=ROOT/f'runs/frozen-model-step{a.step}';m=json.loads((base/'cache_manifest.json').read_text());report=json.loads((ROOT/f'research/idm/model-step{a.step}-baseline-audit.json').read_text());assert report['metadata']==m and m['checkpoint_step']==a.step
 assert sha256(base/'reference_weights.pt')==m['snapshot_sha256']
 observed=json.loads((ROOT/'runs/research-unused-96-prior6501/cache_manifest.json').read_text());assert m['vision_state_sha256']==observed['visual_state_sha256']
 checks={}
 for split,src in m['source_caches'].items():
  assert sha256(Path(src['path']))==src['sha256'];assert sha256(base/f'{split}.pt')==m['cache_sha256'][split]
  source=torch.load(src['path'],weights_only=True);d=torch.load(base/f'{split}.pt',weights_only=True)
  for key in ('current','future'):assert torch.equal(d[key],source[key])
  assert d['rows']==source['rows']
  if split!='observed96':
   old=torch.load(ROOT/f'runs/frozen-idm-step6501/{split}.pt',weights_only=True)
   for key in ('current','future'):assert torch.equal(d[key],old[key])
  for key in ('codes','baseline'):assert torch.isfinite(d[key]).all()
  errors=(d['baseline']-d['future']).square().flatten(1).mean(1)
  key='zero9802_mse' if a.step==9802 else 'zero_model_mse'
  rows=report['results'][split]['rows'];assert [v['source_row'] for v in rows]==d['rows']
  diff=max(abs(float(e)-r[key]) for e,r in zip(errors,rows));assert diff<1e-7
  checks[split]=dict(count=len(rows),h_y_rows_exact=True,zero_mse_reproduction_max_error=diff)
 print(json.dumps(dict(step=a.step,all_checks_passed=True,observed_visual_state_hash_matches=True,splits=checks,manifest_sha256=sha256(base/'cache_manifest.json'),script_sha256=sha256(Path(__file__))),indent=2))
if __name__=='__main__':main()
