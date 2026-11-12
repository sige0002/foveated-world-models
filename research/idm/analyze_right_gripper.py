"""Descriptive right-gripper decomposition of fixed spatial probe outputs; no fitting."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 source=ROOT/'research/idm/spatial-current-boundary-audit/results.json';out=ROOT/'research/idm/right-gripper-error-analysis'
 if out.exists():raise FileExistsError(out)
 r=json.loads(source.read_text())['results'];train=r['zero']['selected_evaluation']['train']['rows'];ty=np.array([x['target'][7] for x in train]);tau=.01;active=np.abs(ty)[np.abs(ty)>tau];boundary=float(np.median(active)) if len(active) else None
 protocol=dict(source=str(source),source_sha256=sha(source),target='right measured gripper endpoint delta, target index7/preparedstate index19; source units, not command',static_threshold=tau,static_rule='abs(delta)<=.01, inherited gripper target floor, fixed before evaluation analysis',small_large_boundary=boundary,boundary_rule='Median absolute nonstatic target among train128 only; small:(tau,boundary],large:>boundary; ties retained, empty bins allowed',prediction_detection='abs(prediction)>tau; otherwise missed movement when target active. Opposite signs count separately.',outlier_counts=[1,3,5],scope='Descriptive analysis of already observed evaluation predictions; no training, threshold selection or prediction correction from evaluation labels',script_sha256=sha(Path(__file__)))
 out.mkdir();(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');results={}
 for split in ('train','validation','observed96'):
  a=r['zero']['selected_evaluation'][split]['rows'];b=r['reference42']['selected_evaluation'][split]['rows'];assert [(x['episode'],x['index'],x['dt']) for x in a]==[(x['episode'],x['index'],x['dt']) for x in b]
  y=np.array([x['target'][7] for x in a]);yp=np.array([x['target'][7] for x in b]);assert np.array_equal(y,yp)
  p=np.array([x['prediction'][7] for x in a]);q=np.array([x['prediction'][7] for x in b]);ez=(p-y)**2;er=(q-y)**2;d=er-ez;static=np.abs(y)<=tau;moving=~static
  bins={'static':static,'small_movement':moving&(np.abs(y)<=boundary) if boundary is not None else np.zeros(len(y),bool),'large_movement':moving&(np.abs(y)>boundary) if boundary is not None else np.zeros(len(y),bool),'positive_target':y>tau,'negative_target':y< -tau}
  def stats(mask):
   n=int(mask.sum())
   if not n:return dict(n=0)
   return dict(n=n,zero_mse=float(ez[mask].mean()),reference_mse=float(er[mask].mean()),paired_mse_change=float(d[mask].mean()),contribution_to_overall_mse_change=float(d[mask].sum()/len(y)),zero_bias=float((p-y)[mask].mean()),reference_bias=float((q-y)[mask].mean()),target_abs_mean=float(np.abs(y[mask]).mean()),zero_prediction_abs_mean=float(np.abs(p[mask]).mean()),reference_prediction_abs_mean=float(np.abs(q[mask]).mean()))
  detection={}
  for name,pred in [('zero',p),('reference',q)]:
   predicted=np.abs(pred)>tau;correct_sign=pred*y>0;opposite=pred*y<0
   detection[name]=dict(static_false_movement_count=int((static&predicted).sum()),static_count=int(static.sum()),moving_miss_count=int((moving&~predicted).sum()),moving_count=int(moving.sum()),moving_detected_correct_sign=int((moving&predicted&correct_sign).sum()),moving_detected_wrong_sign=int((moving&predicted&opposite).sum()),predictions_outside_delta_range=int((np.abs(pred)>1).sum()),moving_correct_sign_under_magnitude_mse_contribution=float(np.where(moving&predicted&correct_sign&(np.abs(pred)<np.abs(y)),(pred-y)**2,0).sum()/len(y)))
  order=np.argsort(-d,kind='stable');concentration={}
  for k in (1,3,5):
   selected=order[:min(k,len(order))];remaining=np.ones(len(y),bool);remaining[selected]=False
   concentration[k]=dict(top_sum_error_increase=float(d[selected].sum()),share_of_net_error_increase=float(d[selected].sum()/d.sum()) if d.sum()!=0 else None,remainder_paired_mse_change=float(d[remaining].mean()),episodes=[str(a[i]['episode']) for i in selected],note='Error-ranked descriptive deletion, not a legitimate improved performance estimate')
  rows=[dict(episode=str(x['episode']),index=x['index'],dt=x['dt'],target=float(y[i]),zero_prediction=float(p[i]),reference_prediction=float(q[i]),zero_squared_error=float(ez[i]),reference_squared_error=float(er[i]),paired_squared_error_change=float(d[i]),bin='static' if static[i] else ('small_movement' if boundary is not None and abs(y[i])<=boundary else 'large_movement')) for i,x in enumerate(a)]
  results[split]=dict(overall=stats(np.ones(len(y),bool)),bins={k:stats(v) for k,v in bins.items()},detection=detection,episodes_worse=int((d>0).sum()),episodes_better=int((d<0).sum()),top_error_increases=[rows[i] for i in order[:5]],outlier_concentration=concentration,rows=rows)
 assert sha(source)==protocol['source_sha256']
 (out/'results.json').write_text(json.dumps(dict(protocol_sha256=sha(out/'protocol.json'),train_target_support_counts={str(v):int((ty==v).sum()) for v in np.unique(ty)},results=results),indent=2)+'\n')
 print(json.dumps(dict(boundary=boundary,summary={k:{a:v for a,v in x.items() if a not in ('rows','top_error_increases')} for k,x in results.items()}),indent=2))
if __name__=='__main__':main()
