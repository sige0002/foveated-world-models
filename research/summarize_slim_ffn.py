"""Summarize every fixed FFN endpoint, preserving seed pairing and missing runs."""
import json
from pathlib import Path
import numpy as np
from research.train_cached_decoders import file_hash

ROOT=Path(__file__).resolve().parents[1]


def identity(rows):
    return [(r['episode'],r['index'],r['dt'],r.get('endpoint_indices')) for r in rows]


def main():
    plan=json.loads((ROOT/'research/results/slim-ffn512-protocol-20261113.json').read_text())
    runs={};pending=[];sources={};common_source=None
    for arm in plan['arms']:
        name=f"ffn{arm['expansion']}-seed{arm['seed']}";run=ROOT/arm['output']
        if not (run/'results.json').exists():pending.append(name);continue
        r=json.loads((run/'results.json').read_text());p=json.loads((run/'protocol.json').read_text())
        assert file_hash(run/'decoder.pt')==r['checkpoint_sha256']
        assert file_hash(run/'protocol.json')==r['protocol_sha256']
        assert (p['expansion'],p['seed'],p['steps'])==(arm['expansion'],arm['seed'],400)
        for source in ['train_slim_ffn.py','slim_ffn_decoder.py']:
            path=str(Path(plan['worktree'])/'research'/source)
            assert p['implementation_sha256'][path]==plan['source_hashes'][path]
        if common_source is None:common_source=p['source_sha256']
        assert p['source_sha256']==common_source
        sources[str((run/'results.json').relative_to(ROOT))]=file_hash(run/'results.json')
        runs[name]=(r,p)
    comparisons={};errors={};paired=[]
    rng=np.random.default_rng(20261113);draws=rng.integers(0,96,(20000,96))
    for seed in [42,43,44]:
        names=[f'ffn{e}-seed{seed}' for e in [4,2]]
        if not all(n in runs for n in names):continue
        a,b=[runs[n][0] for n in names]
        assert a['batch_sha256']==b['batch_sha256']
        ar,br=[r['evaluation']['observed96']['rows'] for r in [a,b]]
        assert identity(ar)==identity(br)
        ea,eb=[np.array([r['mse'] for r in rows]) for rows in [ar,br]]
        delta=eb-ea;paired.append(delta)
        comparisons[str(seed)]={'difference':float(delta.mean()),'relative_percent':float(100*delta.mean()/ea.mean()),
                                'conditional_paired_ci95':np.quantile(delta[draws].mean(1),[.025,.975]).tolist()}
        errors[seed]=(ea,eb)
    aggregate=None
    if len(paired)==3:
        delta=np.stack(paired).mean(0)
        aggregate={'ffn4_mean_risk':float(np.mean([v[0].mean() for v in errors.values()])),
                   'ffn2_mean_risk':float(np.mean([v[1].mean() for v in errors.values()])),
                   'mean_difference':float(delta.mean()),
                   'conditional_paired_ci95':np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
                   'interpretation':'Mean of three risks, not prediction ensemble. Conditional episode bootstrap excludes seed/training/model-selection uncertainty.'}
    reproduction=None
    if 'ffn4-seed42' in runs:
        previous=json.loads((ROOT/'runs/reference-idm42-nested512-comparison400-20261113/comparison.json').read_text())
        old=next(r for r in previous['results'] if r['variant']=='prior_residual')
        new=runs['ffn4-seed42'][0]
        assert [(r['step'],r['mse']) for r in new['history']]==[(r['step'],r['train_objective']) for r in old['history']]
        reproduction='All four logged training objectives exactly reproduce original seed42 baseline'
    report={'complete':not pending,'pending':pending,'source_sha256':sources,
            'runs':{k:{'mse':{s:v['mse'] for s,v in r['evaluation'].items()},'parameters':p['parameters']} for k,(r,p) in runs.items()},
            'comparisons':comparisons,'aggregate':aggregate,'baseline_reproduction':reproduction,
            'scope':'Reference42 posterior mean codes, fresh corrections, same512 training pairs and reused64/96. Fixed400, three decoder seeds. No independent test or executable forecasting claim.'}
    path=ROOT/'research/results/slim-ffn512-summary-20261113.json';path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'pending':pending,'runs':report['runs'],'aggregate':aggregate},indent=2))


if __name__=='__main__':main()
