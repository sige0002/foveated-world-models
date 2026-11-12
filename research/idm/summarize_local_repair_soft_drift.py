"""Fixed completed-run train drift summary; no fitting or checkpoint mutation."""
import json,hashlib
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[2]
RUNS=[('H',42),('H',43),('HD',42),('HD',43),('HD',44)]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 torch.set_num_threads(2);report=dict(scope='Descriptive train-only drift and recorded minibatch-gradient diagnostics of fixed completed H42/43 HD42/43/44. H44 excluded because unfinished at task start; no result-based inclusion. No fitting, hyperparameter selection, or future-swap/probe execution.',runs={})
 for arm,seed in RUNS:
  root=ROOT/f'runs/idm-local-repair-{arm}-seed{seed}-20261113';path=root/'results.json';r=json.loads(path.read_text());assert sha(root/'repair.pt')==r['checkpoint_sha256'];assert sha(root/'protocol.json')==r['protocol_sha256'];protocol=json.loads((root/'protocol.json').read_text());assert protocol['steps']==400 and protocol['execute_training']
  rows=r['evaluation']['train']['rows'];v=torch.tensor([a['drift_kl_to_reference'] for a in rows],dtype=torch.float64);order=torch.argsort(v,descending=True)
  grad=[]
  for h in r['history']:
   if 'gradients' not in h:continue
   g=h['gradients'];m=g['head_mse_gradient_norm'];d=g['head_weighted_drift_gradient_norm'];grad.append(dict(step=h['step'],head_mse_norm=m,head_weighted_drift_norm=d,weighted_drift_to_mse_ratio=d/m if m else None,minibatch_drift=h['drift'],global_active_gradient_norm=h['gradient_norm_before_clipping']))
  report['runs'][f'{arm}{seed}']=dict(source_sha256=sha(path),checkpoint_sha256=r['checkpoint_sha256'],count=len(v),mean=float(v.mean()),quantiles={str(q):float(torch.quantile(v,q)) for q in [.5,.9,.95,.99]},maximum=float(v.max()),above_one=int((v>1).sum()),above_ten=int((v>10).sum()),largest_one_share=float(v[order[:1]].sum()/v.sum()),largest_five_share=float(v[order[:5]].sum()/v.sum()),largest_26_share=float(v[order[:26]].sum()/v.sum()),mean_without_largest_one=float(v[order[1:]].mean()),mean_without_largest_five=float(v[order[5:]].mean()),top_five=[dict(episode=rows[int(i)]['episode'],index=rows[int(i)]['index'],drift=float(v[i])) for i in order[:5]],gradient_diagnostics=grad,drift_scope_failed=r['drift_scope_failed'])
 report['script_sha256']=sha(Path(__file__));out=ROOT/'research/idm/local-repair-soft-drift-summary.json';assert not out.exists();out.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
