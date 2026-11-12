"""Read-only nested512 probe provenance/selection/metric consistency audit."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.probe_displacement_ridge import sha256,scaler,FLOOR
from research.idm.probe_nested512_displacement import gripper_metrics
@torch.inference_mode()
def main():
 torch.set_num_threads(2);base=ROOT/'research/idm/nested512-displacement-probe';p=json.loads((base/'protocol.json').read_text());r=json.loads((base/'results.json').read_text());s=json.loads((base/'train_selection.json').read_text());fit=torch.load(base/'fit_artifacts.pt',weights_only=True);labels=torch.load(base/'train_labels.pt',weights_only=True)
 assert sha256(base/'protocol.json')==r['protocol_sha256'] and sha256(base/'train_selection.json')==r['selection_sha256'] and sha256(base/'fit_artifacts.pt')==r['fit_artifacts_sha256'] and sha256(base/'train_labels.pt')==r['labels_sha256']
 assert sha256(ROOT/'research/idm/probe_nested512_displacement.py')==p['script_sha256']
 for path,digest in p['source_sha256'].items():assert sha256(Path(path))==digest
 for path,digest in p['npz_sha256'].items():assert sha256(Path(path))==digest
 old=ROOT/'research/idm/spatial-current-boundary-audit';op=json.loads((old/'protocol.json').read_text());oldfit=torch.load(old/'kernel_fit_artifacts.pt',weights_only=True);oldlabels=torch.load(ROOT/'research/idm/observed-displacement-labels/train.pt',weights_only=True)
 assert torch.equal(labels['delta8'][:128],oldlabels['delta8'].double()) and torch.equal(labels['current_state20'][:128],oldlabels['current_state20']);assert labels['delta8_mask'].all() and labels['current_state_mask'].all();assert sorted(sum(p['folds'],[]))==list(range(512))
 assert all([i for i in new if i<128]==previous for new,previous in zip(p['folds'],op['folds']))
 assert torch.equal(fit['common_target_scale128'],oldfit['target_scale']);ym,ys=scaler(labels['delta8'],FLOOR);assert torch.equal(ym,fit['target_mean512']) and torch.equal(ys,fit['target_scale512'])
 comparisons={}
 for variant,result in r['results'].items():
  chosen=min(s[variant]['candidates'],key=lambda v:v['cv_normalized_mse']);assert chosen['config']==result['selected_config'] and chosen['alpha']==result['selected_alpha'];comparisons[variant]={}
  for split,m in result['selected_evaluation'].items():
   pred=torch.tensor([row['prediction'] for row in m['rows']],dtype=torch.float64);target=torch.tensor([row['target'] for row in m['rows']],dtype=torch.float64);sq=(pred-target).square()
   assert abs(float((sq/fit['common_target_scale128'].square()).mean())-m['normalized_mse_old128_scale'])<1e-13
   assert abs(float((sq/ys.square()).mean())-m['normalized_mse_train512_scale'])<1e-13
   assert gripper_metrics(target,pred,p['bin_thresholds']['static'],p['bin_thresholds']['small_large'])==m['right_gripper']
   if split!='train':
    prior=result['old128_selected_evaluation'][split];opred=torch.tensor([row['prediction'] for row in prior['rows']],dtype=torch.float64);otarget=torch.tensor([row['target'] for row in prior['rows']],dtype=torch.float64);assert torch.equal(target,otarget)
    assert [(x['episode'],x['index'],x['dt']) for x in m['rows']]==[(x['episode'],x['index'],x['dt']) for x in prior['rows']]
    comparisons[variant][split]=dict(old128_gripper=gripper_metrics(target,opred,p['bin_thresholds']['static'],p['bin_thresholds']['small_large']),new512_gripper=m['right_gripper'])
 print(json.dumps(dict(all_checks_passed=True,old128_labels_states_and_folds_preserved=True,common128_scale_exact=True,train_only_choices_verified=True,metrics_and_gripper_decomposition_reproduce=True,comparisons=comparisons,script_sha256=sha256(Path(__file__))),indent=2))
if __name__=='__main__':main()
