"""Conditional paired episode intervals for already fixed spatial-probe models."""
import json,hashlib
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def main():
 source=ROOT/'research/idm/spatial-current-boundary-audit/results.json';out=ROOT/'research/idm/spatial-current-bootstrap'
 if out.exists():raise FileExistsError(out)
 protocol=dict(source=str(source),source_sha256=sha(source),seed=20261113,resamples=10000,interval='Paired episode bootstrap percentile95%; mean normalized8D MSE(reference minuszero)',models='Already fixed train-CV-selected models; no fitting/reselection',scope='Conditional on trained representations, scalers, CV choices and fitted models; does not cover training/model-selection uncertainty or adaptivity from previously observed research sets',script_sha256=sha(Path(__file__)))
 out.mkdir();(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n');r=json.loads(source.read_text())['results'];rng=np.random.default_rng(protocol['seed']);results={}
 for split in ('validation','observed96'):
  z=r['zero']['selected_evaluation'][split]['rows'];ref=r['reference42']['selected_evaluation'][split]['rows']
  assert [(v['episode'],v['index'],v['dt']) for v in z]==[(v['episode'],v['index'],v['dt']) for v in ref]
  assert len(set(v['episode'] for v in z))==len(z)
  a=np.array([v['normalized_mse'] for v in z]);b=np.array([v['normalized_mse'] for v in ref]);delta=b-a
  indices=rng.integers(0,len(a),(protocol['resamples'],len(a)));sample=delta[indices].mean(1)
  results[split]=dict(episodes=len(a),zero_mse=float(a.mean()),reference_mse=float(b.mean()),paired_mean_difference=float(delta.mean()),percentile95=np.quantile(sample,[.025,.975]).tolist(),relative_point_reduction=float(1-b.mean()/a.mean()),resample_indices_sha256=hashlib.sha256(indices.tobytes()).hexdigest())
 (out/'results.json').write_text(json.dumps(dict(protocol_sha256=sha(out/'protocol.json'),results=results),indent=2)+'\n');print(json.dumps(results,indent=2))
if __name__=='__main__':main()
