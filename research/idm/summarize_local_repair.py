"""Audit and summarize all available prespecified local-repair endpoints."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def identities(rows):
    return [(r['episode'], r['index'], r['dt'], r.get('endpoint_indices')) for r in rows]


def main():
    launch_path = ROOT/'research/idm/local-repair512-launch-20261113.json'
    launch = json.loads(launch_path.read_text())
    results = {}; pending = []; hashes = {}; baseline = None; common_initial = None
    batches = {}
    for item in launch['arms']:
        run = ROOT/item['output']; path = run/'results.json'
        name = f"{item['arm']}-{item['seed']}"
        if not path.exists():
            pending.append(name); continue
        report = json.loads(path.read_text())
        protocol = json.loads((run/'protocol.json').read_text())
        preflight = json.loads((run/'preflight.json').read_text())
        assert sha(run/'protocol.json') == report['protocol_sha256']
        assert sha(run/'preflight.json') == report['preflight_sha256']
        assert sha(run/'repair.pt') == report['checkpoint_sha256']
        assert protocol['arm'] == item['arm'] and protocol['minibatch_seed'] == item['seed']
        assert protocol['steps'] == 400 and report['history'][-1]['step'] == 400
        assert report['frozen_audit_passed']
        assert protocol['source_scripts'][str(ROOT/'research/idm/train_local_repair.py')] == launch['source_sha256']['research/idm/train_local_repair.py']
        if common_initial is None:
            common_initial = report['initial']; baseline = preflight['initial']['F']
        assert report['initial'] == common_initial
        assert preflight['initial']['F'] == baseline
        batches.setdefault(item['seed'], report['batch_sha256'])
        assert batches[item['seed']] == report['batch_sha256']
        for split in baseline:
            assert identities(report['evaluation'][split]['rows']) == identities(baseline[split]['rows'])
        hashes[str(path.relative_to(ROOT))] = sha(path)
        results[name] = report
    comparisons = {}
    def compare(a, b, split='observed96'):
        ar = results[a]['evaluation'][split]['rows']
        br = baseline[split]['rows'] if b == 'F' else results[b]['evaluation'][split]['rows']
        assert identities(ar) == identities(br)
        delta = np.array([r['mse']-s['mse'] for r,s in zip(ar,br)])
        draws = np.random.default_rng(20261113).integers(0,len(delta),(20000,len(delta)))
        return {'difference':float(delta.mean()),
                'relative_percent':float(100*delta.mean()/np.mean([r['mse'] for r in br])),
                'paired_ci95':np.quantile(delta[draws].mean(1),[.025,.975]).tolist()}
    for name in results:
        comparisons[name+'_minus_F'] = compare(name,'F')
    for seed in [42,43,44]:
        for a,b in [('D','K'),('HD','D'),('HD','H')]:
            an,bn = f'{a}-{seed}',f'{b}-{seed}'
            if an in results and bn in results:
                comparisons[an+'_minus_'+bn] = compare(an,bn)
    groups = {}
    for arm in ['H','D','HD','K']:
        names = [f'{arm}-{seed}' for seed in [42,43,44]]
        if all(name in results for name in names):
            values = np.array([results[name]['evaluation']['observed96']['mse'] for name in names])
            groups[arm] = {'individual_mse':values.tolist(), 'mean_mse':float(values.mean()),
                           'seed_sample_std':float(values.std(ddof=1)),
                           'scope_failures':sum(results[name]['drift_scope_failed'] for name in names),
                           'interpretation':'Mean of individual fitted-model risks, not a prediction ensemble; common pretrained weights.'}
    summary = {'launch_sha256':sha(launch_path),'complete':not pending,'pending':pending,
               'source_sha256':hashes,'baseline':None if baseline is None else {s:v['mse'] for s,v in baseline.items()},
               'runs':{name:{'metrics':{s:{k:v[k] for k in ('mse','mean_drift_kl','max_drift_kl')} for s,v in r['evaluation'].items()},
                             'drift_scope_failed':r['drift_scope_failed'],'trainable_parameters':r['trainable_parameters'],
                             'batch_sha256':r['batch_sha256']} for name,r in results.items()},
               'comparisons':comparisons, 'completed_three_seed_groups':groups,
               'scope':'Fixed400 additional updates, common pretrained weights, 3 batch-stream seeds planned. All evaluation episodes reused. Bootstrap conditional on fitted models; no selection/training uncertainty. Mean-code posterior reconstruction, not executable prediction. Scope-failing H/HD must not be adopted even if visual MSE improves.'}
    out = ROOT/'research/idm/local-repair512-summary-20261113.json'
    out.write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({'complete':summary['complete'],'pending':pending,'runs':summary['runs']},indent=2))


if __name__ == '__main__':
    main()
