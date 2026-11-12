"""Summarize fixed-IDM runs with paired episode bootstrap, conditional on trained seeds."""
import argparse
import json
from pathlib import Path

import numpy as np


def summarize(paths):
    records = [json.loads(Path(p).read_text()) for p in paths]
    hashes = records[0]['cache_manifest']['cache_sha256']
    if any(r['cache_manifest']['cache_sha256'] != hashes for r in records):
        raise ValueError('Reports must use the same immutable cache')
    results = [entry for record in records for entry in record['results']]
    index = {(r['variant'],r['seed']):r for r in results}
    if len(index) != len(results):
        raise ValueError('Duplicate variant/seed')
    reference = [(r['episode'],r['index'],r['dt']) for r in results[0]['validation']['rows']]
    if any([(r['episode'],r['index'],r['dt']) for r in item['validation']['rows']] != reference for item in results):
        raise ValueError('Validation pairs/order differ')
    summary = {}
    for variant in sorted({r['variant'] for r in results}):
        entries = [r for r in results if r['variant']==variant]
        summary[variant] = dict(seeds=sorted(r['seed'] for r in entries),
                               metrics={key:dict(mean=float(np.mean([r['validation']['metrics'][key] for r in entries])),
                                                seed_std=float(np.std([r['validation']['metrics'][key] for r in entries],ddof=1)) if len(entries)>1 else None)
                                        for key in ('mse','mean_mse','shuffled_mse','copy_mse')})
    comparisons = []
    rng = np.random.default_rng(2026)
    for baseline in ('residual','refinement'):
        seeds = sorted({s for kind,s in index if kind=='flow'} & {s for kind,s in index if kind==baseline})
        if not seeds:
            continue
        for metric in ('mse','mean_mse'):
            per_seed = np.array([[a[metric]-b['mse'] for a,b in zip(index['flow',s]['validation']['rows'],index[baseline,s]['validation']['rows'])] for s in seeds])
            episodes = sorted({r[0] for r in reference})
            # Pair comparisons and aggregate within an episode before resampling.
            per_episode = np.array([per_seed[:,[i for i,r in enumerate(reference) if r[0]==ep]].mean() for ep in episodes])
            sampled = per_episode[rng.integers(len(episodes),size=(10000,len(episodes)))].mean(1)
            comparisons.append(dict(flow_metric=metric, baseline=baseline, seeds=seeds,
                                    difference=float(per_episode.mean()), seed_differences=per_seed.mean(1).tolist(),
                                    episode_bootstrap_ci95=np.quantile(sampled,[.025,.975]).tolist()))
    return dict(summary=summary, comparisons=comparisons,
                interval_scope='Paired episode bootstrap conditional on the trained seeds; not uncertainty over all training seeds.',
                compute_note='flow mse uses individual 4-step samples; flow mean_mse averages 4 samples (16 network calls).',
                cache_sha256=hashes)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reports',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():
        parser.error('output already exists')
    result=summarize(args.reports)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
