"""Paired fixed400 data-scaling controls; observed episodes, seed42 only."""
import hashlib
import json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]


def main():
    sources={}
    def load(name):
        path=ROOT/name
        sources[name]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    new=load('research/results/nested512-direct-constant400-observed96-20261113.json')['results']
    old=load('research/results/final-prior-constant-controls-observed96-20261113.json')['results']
    originals=load('research/idm/model-step10000-baseline-audit.json')['results']
    report={'scope':'Prespecified fixed400, seed42/head42, nested128/512 with frozen128-trained head/calibration. Previously observed64/96; conditional episode bootstrap excludes training and adaptive-selection uncertainty.',
            'bootstrap_seed':20261113,'bootstrap_resamples':20000,'splits':{}}
    for split in ('validation','observed96'):
        if split=='validation':
            paths={'reference128':'runs/reference-idm42-final-prior-residual42-20261113/comparison.json',
                   'constant128':'research/results/constant-final-prior-residual42-dev64-20261113.json',
                   'reference512':'research/results/nested512-direct400-dev64-20261113.json',
                   'constant512':'research/results/nested512-constant400-dev64-20261113.json'}
            records={k:load(p)['results'][0]['validation']['rows'] for k,p in paths.items()}
        else:
            records={}
            for n,entries in ((128,old),(512,new)):
                for control in (False,True):
                    item,=[r for r in entries if r['seed']==42 and r['constant_code_control']==control]
                    records[('constant' if control else 'reference')+str(n)]=item['rows']
        ids=[(r['episode'],r['index'],r['dt']) for r in records['reference128']]
        assert len(set(i[0] for i in ids))==len(ids)
        for rows in records.values():
            assert [(r['episode'],r['index'],r['dt']) for r in rows]==ids
        orig=originals[split]['rows']
        assert [(r['source_row']['episode'],r['source_row']['index'],r['source_row']['dt']) for r in orig]==ids
        values={k:np.array([r['mse'] for r in rows]) for k,rows in records.items()}
        values['original10000']=np.array([r['original_model_mse'] for r in orig])
        def compare(a,b,mask=None):
            if mask is not None:a,b=a[mask],b[mask]
            diff=a-b;rng=np.random.default_rng(20261113)
            ci=np.quantile(diff[rng.integers(0,len(diff),(20000,len(diff)))].mean(1),[.025,.975])
            return dict(count=len(diff),candidate_mse=float(a.mean()),baseline_mse=float(b.mean()),
                        difference=float(diff.mean()),relative_reduction=float(1-a.mean()/b.mean()),ci95=ci.tolist())
        pairs=[('reference512','reference128'),('constant512','constant128'),
               ('reference512','constant512'),('reference512','original10000'),('constant512','original10000')]
        comparisons={a+'_vs_'+b:compare(values[a],values[b]) for a,b in pairs}
        mask=np.array([abs(i[2]-1.2)<1e-4 for i in ids])
        report['splits'][split]=dict(means={k:float(v.mean()) for k,v in values.items()},comparisons=comparisons,
          nominal1p2=compare(values['reference512'],values['original10000'],mask),
          rows=[dict(episode=i[0],index=i[1],dt=i[2],**{k:float(v[j]) for k,v in values.items()}) for j,i in enumerate(ids)])
    report['sources']=sources
    report['interpretation']='More data helps constant-code correction substantially. The extra-capacity explanation rejected at128 is not rejected at512: both current-only correction and reference-code conditioning contribute. This is not causal action identity or deployable-action evaluation.'
    report['script_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output=ROOT/'research/results/nested512-fixed400-summary-20261113.json'
    if output.exists():raise FileExistsError(output)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({s:{k:v for k,v in d.items() if k!='rows'} for s,d in report['splits'].items()},indent=2))


if __name__=='__main__':main()
