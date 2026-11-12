"""Read-only protocols, CV selections, row metrics and kernel-scale audit."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.probe_displacement_ridge import sha256,load
@torch.inference_mode()
def main():
 torch.set_num_threads(2);_,sources=load();checks={}
 for name in ('spatial-current-displacement-probe','spatial-current-boundary-audit'):
  base=ROOT/'research/idm'/name;p=json.loads((base/'protocol.json').read_text());r=json.loads((base/'results.json').read_text());s=json.loads((base/'train_selection.json').read_text())
  assert p['source_sha256']==sources and r['protocol_sha256']==sha256(base/'protocol.json') and r['selection_sha256']==sha256(base/'train_selection.json') and r['fit_artifacts_sha256']==sha256(base/'kernel_fit_artifacts.pt')
  assert all(abs(row['kernel_mean_diagonal']-1)<1e-12 for row in s['fold_normalization'])
  saved=torch.load(base/'kernel_fit_artifacts.pt',weights_only=True);ys=saved['target_scale']
  for v,result in r['results'].items():
   chosen=min(s['selection'][v]['candidates'],key=lambda x:x['cv_normalized_mse'])
   assert (chosen['config'],chosen['alpha'])==(result['selected_config'],result['selected_alpha'])
   for split,values in result['selected_evaluation'].items():
    pred=torch.tensor([t['prediction'] for t in values['rows']],dtype=torch.float64);target=torch.tensor([t['target'] for t in values['rows']],dtype=torch.float64)
    assert abs(float(((pred-target)/ys).square().mean())-values['normalized_mse'])<1e-13
  checks[name]=dict(train_cv_selection_verified=True,all_group_fold_kernel_scales_one=True,saved_predictions_reproduce_metrics=True)
 print(json.dumps(dict(all_checks_passed=True,checks=checks,script_sha256=sha256(Path(__file__))),indent=2))
if __name__=='__main__':main()
