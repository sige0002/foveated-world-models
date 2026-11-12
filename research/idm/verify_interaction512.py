"""Read-only artifact and numerical consistency checks for interaction probe."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.probe_displacement_ridge import sha256
@torch.inference_mode()
def main():
 torch.set_num_threads(2);d=ROOT/'research/idm/interaction-probe512';p=json.loads((d/'protocol.json').read_text());r=json.loads((d/'results.json').read_text());s=json.loads((d/'train_selection.json').read_text());f=torch.load(d/'fit_artifacts.pt',weights_only=True)
 assert sha256(d/'protocol.json')==r['protocol_sha256'] and sha256(d/'train_selection.json')==r['selection_sha256'] and sha256(d/'fit_artifacts.pt')==r['fit_artifacts_sha256']
 for path,h in p['source_sha256'].items():assert sha256(Path(path))==h,path
 assert sorted(sum(p['folds'],[]))==list(range(512))
 for key,m in f['models'].items():assert abs(float(m['kernel'].diag().mean())-1)<1e-12
 old=json.loads((ROOT/'research/idm/nested512-displacement-probe/results.json').read_text());oldfit=ROOT/'research/idm/nested512-displacement-probe/fit_artifacts.pt';assert sha256(oldfit)==old['fit_artifacts_sha256'];assert torch.equal(f['common_target_scale128'],torch.load(oldfit,weights_only=True)['common_target_scale128'])
 for v,a in s.items():
  assert a['selected']==min(a['candidates'],key=lambda x:x['cv_normalized_mse'])
  for split,e in r['results'][v]['selected_evaluation'].items():
   rows=e['rows'];pred=torch.tensor([x['prediction'] for x in rows],dtype=torch.float64);target=torch.tensor([x['target'] for x in rows],dtype=torch.float64);mse=(pred-target).square();assert abs(float((mse/f['common_target_scale128'].square()).mean())-e['normalized_mse_old128_scale'])<1e-12
   assert torch.allclose(mse.mean(0),torch.tensor(e['physical_coordinate_mse'],dtype=torch.float64),atol=1e-12)
   if v in ('zero','additive'):assert abs(e['normalized_mse_old128_scale']-old['results']['zero' if v=='zero' else 'reference42']['selected_evaluation'][split]['normalized_mse_old128_scale'])<1e-12
 report=dict(pass_all=True,source_files_checked=len(p['source_sha256']),fold_partition_exact=True,prior_fit_hash_and_common_scale_verified=True,kernel_strength_one=True,selection_train_minimum=True,selected_boundary_status={v:a['alpha_at_boundary'] for v,a in s.items()},physical_and_normalized_metrics_recomputed=True,zero_and_additive_prior512_evaluation_reproduced=True,script_sha256=sha256(Path(__file__)),results_sha256=sha256(d/'results.json'))
 (ROOT/'research/idm/interaction-probe512-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
