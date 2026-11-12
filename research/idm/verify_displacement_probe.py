"""Reproduce frozen probe outputs; no refitting or selection changes."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.probe_displacement_ridge import load,predict,sha256
@torch.inference_mode()
def main():
 torch.set_num_threads(2);base=ROOT/'research/idm/displacement-ridge-probe';p=json.loads((base/'protocol.json').read_text());r=json.loads((base/'results.json').read_text())
 assert sha256(base/'protocol.json')==r['protocol_sha256'] and sha256(base/'fitted_models.pt')==r['models_sha256']
 assert sha256(ROOT/'research/idm/probe_displacement_ridge.py')==p['script_sha256']
 data,sources=load();assert sources==p['source_sha256'];saved=torch.load(base/'fitted_models.pt',weights_only=True)
 checks={}
 for name,result in r['results'].items():
  assert result['selected_alpha']==min(result['cv'],key=lambda v:v['normalized_mse'])['alpha']
  checks[name]={}
  for split,d in data.items():
   pred=predict(saved['models'][name],d['x'][name]);recorded=torch.tensor([row['prediction'] for row in result['evaluation'][split]['rows']],dtype=torch.float64)
   assert torch.equal(pred,recorded)
   norm=float(((pred-d['y'])/saved['fulltrain_target_scale']).square().mean())
   assert abs(norm-result['evaluation'][split]['normalized_mse'])<1e-14
   checks[name][split]=norm
 print(json.dumps(dict(all_checks_passed=True,frozen_predictions_exactly_reproduced=True,metrics=checks,protocol_sha256=r['protocol_sha256'],script_sha256=sha256(Path(__file__))),indent=2))
if __name__=='__main__':main()
