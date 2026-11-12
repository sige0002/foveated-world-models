"""Summarize matched results; intervals resample held-out episodes, not training seeds."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('comparison',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
x=json.loads(a.comparison.read_text());groups=defaultdict(list)
for r in x['results']:groups[r['variant']].append(r)
summary={}
for variant,rs in groups.items():
    mean=np.array([r['validation']['metrics']['mean_mse'] for r in rs])
    summary[variant]=dict(seeds=[r['seed'] for r in rs],mean_mse_mean=float(mean.mean()),mean_mse_sd=float(mean.std(ddof=1)) if len(mean)>1 else None,
        train_mean_mse=float(np.mean([r['train_metrics']['mean_mse'] for r in rs])),
        sample_swap_gap=float(np.mean([r['validation']['metrics']['swapped_mse']-r['validation']['metrics']['sample_mse'] for r in rs])))
pairs={};rng=np.random.default_rng(2026)
reference={r['seed']:r for r in groups['reference']}
for variant in ('raw','centered','constant'):
    episodes=defaultdict(list)
    for r in groups[variant]:
        ref=reference[r['seed']]
        for left,right in zip(ref['validation']['rows'],r['validation']['rows']):
            assert left['index']==right['index'] and left['episode']==right['episode']
            episodes[left['episode']].append(left['mean_mse']-right['mean_mse'])
    # Equal episode weighting; aggregate seeds within each episode before resampling.
    deltas=np.array([np.mean(v) for v in episodes.values()]);boot=deltas[rng.integers(len(deltas),size=(5000,len(deltas)))].mean(1)
    pairs[variant]=dict(episode_count=len(deltas),reference_minus_comparator=float(deltas.mean()),episode_bootstrap95=np.quantile(boot,[.025,.975]).tolist())
report=dict(variants=summary,paired_episode_comparisons=pairs,
    limitation='Posterior sees real future; fixed encoder and tiny training cache. Episode intervals condition on the evaluated seeds; no action or camera semantics tested.')
a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
