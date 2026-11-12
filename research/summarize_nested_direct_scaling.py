"""Summarize completed direct-prediction scaling, retaining fixed seed/step scope."""
import hashlib,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
def main():
 sources={}
 def load(p):
  f=ROOT/p;sources[p]=hashlib.sha256(f.read_bytes()).hexdigest();return json.loads(f.read_text())
 original=load('research/idm/model-step10000-baseline-audit.json')['results'];report={'splits':{},'scope':'Previously observed64/96, fixed endpoints.400 has3 paired head/decoder seeds;1600 onlyseed42.Encoder2900/prior10000/calibration128 shared; heads frozen from128. Mean risk is not an ensemble. Conditional bootstrap excludes training/selection uncertainty.','bootstrap_seed':20261113,'bootstrap_resamples':20000}
 for split in ('validation','observed96'):
  values={};ids=None
  def take(key,obj,dev=False):
   nonlocal ids
   if dev:obj=obj['validation']
   rows=obj['rows'];these=[(r['episode'],r['index'],r['dt']) for r in rows]
   if ids is None:ids=these
   assert ids==these
   values[key]=np.array([r['mse'] for r in rows])
  for seed in (42,43,44):
   if split=='validation':
    a=load(f'runs/reference-idm{seed}-final-prior-residual{seed}-20261113/comparison.json')['results'][0]
    name='nested512-direct400-dev64-20261113.json' if seed==42 else f'nested512-direct{seed}-400-dev64-20261113.json'
    b=load('research/results/'+name)['results'][0]
   else:
    a=load(f'research/results/reference-idm{seed}-final-prior-observed96-20261113.json')['results'][0]
    name='nested512-direct-constant400-observed96-20261113.json' if seed==42 else f'nested512-direct{seed}-400-observed96-20261113.json'
    b=load('research/results/'+name)['results'][0]
   take(f'128_400_{seed}',a,split=='validation');take(f'512_400_{seed}',b,split=='validation')
  for n in (128,512):
   if split=='validation':p=f'research/results/final-prior128-residual1600-dev64-20261113.json' if n==128 else 'research/results/nested512-direct1600-dev64-20261113.json'
   else:p='research/results/final-prior128-residual1600-observed96-20261113.json' if n==128 else 'research/results/nested512-direct1600-observed96-20261113.json'
   take(f'{n}_1600_42',load(p)['results'][0],split=='validation')
  rows=original[split]['rows'];assert [(r['source_row']['episode'],r['source_row']['index'],r['source_row']['dt']) for r in rows]==ids
  values['original10000']=np.array([r['original_model_mse'] for r in rows])
  for n in (128,512):values[f'{n}_400_mean3']=np.stack([values[f'{n}_400_{s}'] for s in (42,43,44)]).mean(0)
  def compare(a,b):
   x,y=values[a],values[b];delta=x-y;rng=np.random.default_rng(20261113)
   ci=np.quantile(delta[rng.integers(0,len(delta),(20000,len(delta)))].mean(1),[.025,.975])
   return dict(candidate_mse=float(x.mean()),baseline_mse=float(y.mean()),difference=float(delta.mean()),relative_reduction=float(1-x.mean()/y.mean()),ci95=ci.tolist())
  pairs=[('512_400_mean3','128_400_mean3'),('512_400_mean3','original10000'),('512_1600_42','512_400_42'),('512_1600_42','128_1600_42'),('512_1600_42','128_400_42'),('512_1600_42','original10000')]
  report['splits'][split]=dict(means={k:float(v.mean()) for k,v in values.items()},seed_sd_512_400=float(np.std([values[f'512_400_{s}'].mean() for s in (42,43,44)],ddof=1)),comparisons={a+'_vs_'+b:compare(a,b) for a,b in pairs})
 report['sources']=sources;report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();p=ROOT/'research/results/nested512-direct-scaling-summary-20261113.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report['splits'],indent=2))
if __name__=='__main__':main()
