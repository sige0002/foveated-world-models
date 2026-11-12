"""Metadata/NPZ-only support count for a fixed selection and fixed128-derived bins."""
import json,hashlib,sys
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from vendor.dataset.robomind import RoboMINDDataset

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def main():
 torch.set_num_threads(2);selection_path=ROOT/'research/results/nested-train512-selection-20261113.json';out=ROOT/'research/idm/nested512-right-gripper-support.json'
 if out.exists():raise FileExistsError(out)
 selection=json.loads(selection_path.read_text());selection_hash=sha(selection_path);assert sha(selection['manifest'])==selection['manifest_sha256']
 bpath=ROOT/'research/idm/right-gripper-error-analysis/protocol.json';bins=json.loads(bpath.read_text());tau=bins['static_threshold'];boundary=bins['small_large_boundary'];dataset=RoboMINDDataset(selection['manifest'],**selection['options']);labels=torch.load(ROOT/'research/idm/observed-displacement-labels/train.pt',weights_only=True)
 rows={};array_hashes={}
 for group in ('retained','added'):
  rows[group]=[]
  for j,row in enumerate(selection[group]):
   mi,a,b,_=dataset.pairs[row['index']];entry=dataset.rows[mi];arrays=dataset._arrays(mi);dt=float(arrays['timestamps'][b]-arrays['timestamps'][a]);assert str(entry['episode_id'])==str(row['episode']) and abs(dt-row['dt'])<1e-9
   if 'endpoint_indices' in row:assert row['endpoint_indices']==[a,b]
   assert arrays['original_valid'][[a,b]].all() and not (arrays['fill_age_seconds'][[a,b]]>0).any()
   delta=float(np.float32(arrays['states'][b,19]-arrays['states'][a,19]))
   if group=='retained':
    assert delta==float(labels['delta8'][j,7]) and all(row[k]==labels['rows'][j][k] for k in ('episode','index','dt'))
   ap=dataset.path.parent/entry['array_file'];array_hashes[str(ap)]=sha(ap)
   rows[group].append(dict(row,right_gripper_delta=delta,endpoint_indices=[a,b]))
 def count(values):
  y=np.array([r['right_gripper_delta'] for r in values]);n=len(y)
  return dict(n=n,static=int((abs(y)<=tau).sum()),small_movement=int(((abs(y)>tau)&(abs(y)<=boundary)).sum()),large_movement=int((abs(y)>boundary).sum()),positive_active=int((y>tau).sum()),negative_active=int((y< -tau).sum()),large_positive=int((y>boundary).sum()),large_negative=int((y< -boundary).sum()),large_positive_fraction=float((y>boundary).mean()))
 allrows=rows['retained']+rows['added'];assert len(allrows)==512 and len({r['episode'] for r in allrows})==512;assert sha(selection_path)==selection_hash
 result=dict(selection_sha256=selection_hash,selection_path=str(selection_path),manifest_sha256=selection['manifest_sha256'],bin_protocol_sha256=sha(bpath),static_threshold=tau,small_large_boundary=boundary,counts={g:count(v) for g,v in [('retained128',rows['retained']),('added384',rows['added']),('full512',allrows)]},rows=rows,array_sha256=array_hashes,retained_targets_exactly_equal_previous128=True,all_endpoints_observed=True,no_selection_changes=True,no_threshold_refit=True,no_video_decoding=True,no_model_fitting=True,script_sha256=sha(Path(__file__)),scope='Training-support description only, not evaluation results. Right measured endpoint gripper value difference in sourceunits, not action command/aperture/velocity. Bins fixed from previoustrain128; no performance implication.')
 out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['counts'],indent=2))
if __name__=='__main__':main()
