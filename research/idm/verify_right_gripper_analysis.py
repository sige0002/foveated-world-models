"""Check descriptive decomposition against original measured labels, without fitting."""
import json,sys
from pathlib import Path
import torch,numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.cache_reference_queries import sha256

def main():
 torch.set_num_threads(2);base=ROOT/'research/idm/right-gripper-error-analysis';p=json.loads((base/'protocol.json').read_text());r=json.loads((base/'results.json').read_text());assert sha256(Path(p['source']))==p['source_sha256'] and sha256(base/'protocol.json')==r['protocol_sha256'];labeldir=ROOT/'research/idm/observed-displacement-labels';lm=json.loads((labeldir/'manifest.json').read_text());checks={}
 for split,d in r['results'].items():
  path=labeldir/f'{split}.pt';assert sha256(path)==lm['output_sha256'][path.name];labels=torch.load(path,weights_only=True)
  assert labels['delta8_mask'][:,7].all()
  for i,row in enumerate(d['rows']):
   assert float(labels['delta8'][i,7])==row['target']
   assert all(row[k]==labels['rows'][i][k] for k in ('episode','index','dt'))
  bins=d['bins'];assert sum(bins[b]['n'] for b in ('static','small_movement','large_movement'))==len(d['rows'])
  contribution=sum(bins[b].get('contribution_to_overall_mse_change',0) for b in ('static','small_movement','large_movement'));assert abs(contribution-d['overall']['paired_mse_change'])<1e-14
  checks[split]=dict(n=len(d['rows']),targets_exactly_match_measured_labels=True,bin_counts_and_mse_contributions_reconcile=True)
 train=np.array([x['target'] for x in r['results']['train']['rows']]);assert np.median(abs(train[abs(train)>.01]))==p['small_large_boundary']
 print(json.dumps(dict(all_checks_passed=True,checks=checks,protocol_sha256=sha256(base/'protocol.json'),script_sha256=sha256(Path(__file__))),indent=2))
if __name__=='__main__':main()
