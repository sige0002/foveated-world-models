"""Read-only artifact and numerical consistency checks for query probe."""
import json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.probe_displacement_ridge import sha256
@torch.inference_mode()
def main():
 torch.set_num_threads(2);d=ROOT/'research/idm/query-accessibility512';p=json.loads((d/'protocol.json').read_text());r=json.loads((d/'results.json').read_text());s=json.loads((d/'train_selection.json').read_text());f=torch.load(d/'fit_artifacts.pt',weights_only=True)
 assert sha256(d/'protocol.json')==r['protocol_sha256'] and sha256(d/'train_selection.json')==r['selection_sha256'] and sha256(d/'fit_artifacts.pt')==r['fit_artifacts_sha256']
 for path,h in p['source_sha256'].items():assert sha256(Path(path))==h,path
 assert sorted(sum(p['folds'],[]))==list(range(512))
 for pc in f['fold_pcas']+[f['fulltrain_pca']]:assert torch.allclose(pc['components'].T@pc['components'],torch.eye(32,dtype=torch.float64),atol=1e-12)
 for key,m in f['models'].items():assert abs(float(m['kernel'].diag().mean())-1)<1e-12
 old=json.loads((ROOT/'research/idm/nested512-displacement-probe/results.json').read_text())
 for v,a in s.items():
  assert a['selected']==min(a['candidates'],key=lambda x:x['cv_normalized_mse']);assert not a['alpha_at_boundary']
  for split,e in r['results'][v]['selected_evaluation'].items():
   rows=e['rows'];pred=torch.tensor([x['prediction'] for x in rows],dtype=torch.float64);target=torch.tensor([x['target'] for x in rows],dtype=torch.float64);mse=(pred-target).square();assert abs(float((mse/f['common_target_scale128'].square()).mean())-e['normalized_mse_old128_scale'])<1e-12
   assert torch.allclose(mse.mean(0),torch.tensor(e['physical_coordinate_mse'],dtype=torch.float64),atol=1e-12)
   if v=='mu32':assert abs(e['normalized_mse_old128_scale']-old['results']['reference42']['selected_evaluation'][split]['normalized_mse_old128_scale'])<1e-12
 report=dict(pass_all=True,source_files_checked=len(p['source_sha256']),fold_partition_exact=True,pca_orthonormal=True,kernel_strength_one=True,selection_train_minimum=True,all_selected_alphas_interior=True,physical_and_normalized_metrics_recomputed=True,mu_prior512_evaluation_reproduced=True,script_sha256=sha256(Path(__file__)),results_sha256=sha256(d/'results.json'))
 (ROOT/'research/idm/query-accessibility512-verification.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
