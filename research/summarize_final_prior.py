import json, shutil
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]; out=root/'research/results'
load=lambda p:json.loads((root/p).read_text())
report={'scope':'Fixed400, head/decoder seeds42/43/44, shared encoder2900 and final10000 prior. Previously observed evaluation sets. Mean model risk, not ensemble. True-future posterior reconstruction. Additional training cost.', 'splits':{}}
for split in ['validation','observed96']:
 rows=[]
 for s in [42,43,44]:
  p=f'runs/reference-idm{s}-final-prior-residual{s}-20261113/comparison.json'
  if split=='validation':
   d=load(p)['results'][0]['validation']; shutil.copyfile(root/p,out/f'reference-idm{s}-final-prior-dev64-20261113.json')
  else:d=load(f'research/results/reference-idm{s}-final-prior-observed96-20261113.json')['results'][0]
  rows.append(d['rows'])
 ids=[(r['episode'],r['index']) for r in rows[0]]
 assert all([(r['episode'],r['index']) for r in rr]==ids for rr in rows)
 vals=np.array([[r['mse'] for r in rr] for rr in rows]); mean=vals.mean(0)
 item={'per_seed_mse':vals.mean(1).tolist(),'mean_model_mse':float(mean.mean()),'seed_sd':float(vals.mean(1).std(ddof=1)),'comparisons':{}}
 for step in [9802,10000]:
  audit=load(f'research/idm/model-step{step}-baseline-audit.json')['results'][split]['rows']
  assert [(r['source_row']['episode'],r['source_row']['index']) for r in audit]==ids
  key='original9802_mse' if step==9802 else 'original_model_mse'
  base=np.array([r[key] for r in audit]); diff=mean-base
  rng=np.random.default_rng(20261113); ci=np.quantile(diff[rng.integers(0,len(diff),(20000,len(diff)))].mean(1),[.025,.975])
  item['comparisons'][str(step)]={'original_mse':float(base.mean()),'relative_reduction':float(1-mean.mean()/base.mean()),'difference':float(diff.mean()),'paired_episode_bootstrap_ci95':ci.tolist()}
 item['rows']=[{'episode':r['episode'],'index':r['index'],'mean_model_mse':float(v)} for r,v in zip(rows[0],mean)]
 report['splits'][split]=item
old=[]
for name in ['prior-correction-unused96-first-evaluation-20261113.json','reference-idm43-decoder43-observed96-20261113.json','reference-idm44-decoder44-observed96-20261113.json']:
 data=load('research/results/'+name)
 old.append(next(r for r in data['results'] if r['variant']=='prior_residual' and not r.get('constant_code_control',False))['rows'])
new=report['splits']['observed96']['rows']; ids=[(r['episode'],r['index']) for r in new]
assert all([(r['episode'],r['index']) for r in rr]==ids for rr in old)
a=np.array([[r['mse'] for r in rr] for rr in old]).mean(0); b=np.array([r['mean_model_mse'] for r in new]); diff=b-a
rng=np.random.default_rng(20261113)
report['splits']['observed96']['comparisons']['matched_prior6501_correction']={'original_mse':float(a.mean()),'relative_reduction':float(1-b.mean()/a.mean()),'difference':float(diff.mean()),'paired_episode_bootstrap_ci95':np.quantile(diff[rng.integers(0,len(diff),(20000,len(diff)))].mean(1),[.025,.975]).tolist(),'scope':'Same head seeds and correction seed/steps; prior checkpoint changed'}
p=out/'final-prior-correction-summary-20261113.json';p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({s:{k:v for k,v in d.items() if k!='rows'} for s,d in report['splits'].items()},indent=2))
