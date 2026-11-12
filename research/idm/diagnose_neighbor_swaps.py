"""Prespecified current-distance matched bijective latent swaps; exploratory only."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from research.idm.train_reference_heads import Model,representation
from research.train_cached_decoders import file_hash

def assignment(cost):
    """Square minimum-cost Hungarian assignment; deterministic tie order."""
    a=np.asarray(cost,dtype=float);n=len(a)
    if a.shape!=(n,n) or not np.isfinite(a).all():raise ValueError('Finite square cost required')
    u=np.zeros(n+1);v=np.zeros(n+1);p=np.zeros(n+1,dtype=int);way=np.zeros(n+1,dtype=int)
    for i in range(1,n+1):
        p[0]=i;j0=0;mins=np.full(n+1,np.inf);used=np.zeros(n+1,dtype=bool)
        while True:
            used[j0]=True;i0=p[j0];delta=np.inf;j1=0
            for j in range(1,n+1):
                if not used[j]:
                    cur=a[i0-1,j-1]-u[i0]-v[j]
                    if cur<mins[j]:mins[j]=cur;way[j]=j0
                    if mins[j]<delta:delta=mins[j];j1=j
            for j in range(n+1):
                if used[j]:u[p[j]]+=delta;v[j]-=delta
                else:mins[j]-=delta
            j0=j1
            if p[j0]==0:break
        while True:
            j1=way[j0];p[j0]=p[j1];j0=j1
            if j0==0:break
    result=np.empty(n,dtype=int)
    for j in range(1,n+1):result[p[j]-1]=j-1
    return result

def distances(x):
    x=x.flatten(1).double();x=x/x.norm(dim=1,keepdim=True).clamp_min(1e-12)
    return (2-2*x@x.T).clamp_min(0).sqrt().numpy()

def allowed_episodes(rows):
    e=np.array([r['episode'] for r in rows]);return e[:,None]!=e[None,:]

def summary(a):return dict(mean=float(np.mean(a)),quantiles={str(q):float(np.quantile(a,q)) for q in (0,.25,.5,.75,.95,1)})

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--queries',type=Path,required=True)
    p.add_argument('--comparison',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('New output required')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('medium')
    data={s:torch.load(a.cache/f'{s}.pt',weights_only=True) for s in ('train','validation')}
    manifest=json.loads((a.queries/'manifest.json').read_text())
    for s in data:assert file_hash(a.cache/f'{s}.pt')==manifest['parent_cache_sha256'][s]
    train_d=distances(data['train']['current']);val_d=distances(data['validation']['current'])
    train_allowed=allowed_episodes(data['train']['rows']);valid=allowed_episodes(data['validation']['rows'])
    np.fill_diagonal(train_allowed,False);np.fill_diagonal(valid,False)
    nearest=np.where(train_allowed,train_d,np.inf).min(1)
    if not np.isfinite(nearest).all():raise ValueError('No different-episode train neighbors')
    radius=float(np.quantile(nearest,.95));n=len(val_d);diagonal=np.arange(n)
    within=valid&(val_d<=radius)
    constrained=assignment(np.where(within,val_d,1e6))
    feasible=bool(within[diagonal,constrained].all())
    mappings={'global_cyclic':(diagonal+1)%n,
        'minimum_distance':assignment(np.where(valid,val_d,1e6)),
        'maximum_distance':assignment(np.where(valid,-val_d,1e6))}
    if feasible:mappings['radius_constrained']=constrained
    mapping_report={}
    for name,m in mappings.items():
        assert len(set(m.tolist()))==n and np.all(m!=diagonal)
        if not valid[diagonal,m].all():raise ValueError('Different-episode derangement unavailable')
        d=val_d[diagonal,m]
        mapping_report[name]=dict(indices=m.tolist(),distance=summary(d),within_train_radius_fraction=float((d<=radius).mean()))
    report=dict(distance='Full current patch tensor flattened, per-example L2 normalized, Euclidean distance; float64 arithmetic',
        train_radius95=radius,train_nearest_distances=summary(nearest),radius_constrained_feasible=feasible,
        validation_rows_without_allowed_neighbor=int((~within.any(1)).sum()),mappings=mapping_report,results=[],
        limitations='Exploratory dependence sensitivity, not a conditional independence test or action/camera semantic certificate. When infeasible, minimum-distance assignment is not a verified neighborhood assignment.')
    print('MATCHING',json.dumps({k:v for k,v in report.items() if k!='mappings'}),flush=True)
    q=torch.load(a.queries/'validation.pt',weights_only=True);cal=torch.load(a.queries/'train_calibration.pt',weights_only=True)
    assert q['rows']==data['validation']['rows']
    comparison=json.loads(a.comparison.read_text())
    for entry in comparison['results']:
        seed,kind=entry['seed'],entry['variant'];path=a.comparison.parent/f'{kind}-{seed}.pt'
        saved=torch.load(path,weights_only=True)
        assert saved['steps']==comparison['settings']['steps']
        model=Model().eval();model.load_state_dict(saved['state_dict'])
        mu,lv=model.posterior(representation(q,cal,kind))
        gen=torch.Generator().manual_seed(1407);eps=torch.randn(mu.shape,generator=gen)
        samples=mu+(.5*lv).exp()*eps
        values={name:[] for name in mappings};base_rows=[]
        for start in range(0,n,8):
            sl=slice(start,start+8);current=data['validation']['current'][sl];target=data['validation']['future'][sl]
            base_mean=model.decoder(current,mu[sl]);base_sample=model.decoder(current,samples[sl])
            bm=(base_mean-target).square().flatten(1).mean(1);bs=(base_sample-target).square().flatten(1).mean(1)
            for i in range(len(current)):base_rows.append(dict(index=start+i,mean_mse=float(bm[i]),sample_mse=float(bs[i])))
            for name,mapping in mappings.items():
                j=torch.tensor(mapping[start:start+8]);wrong=mu[j]+(.5*lv[j]).exp()*eps[sl]
                mean=model.decoder(current,mu[j]);sample=model.decoder(current,wrong)
                mm=(mean-target).square().flatten(1).mean(1);sm=(sample-target).square().flatten(1).mean(1)
                change=(sample-base_sample).square().flatten(1).mean(1)
                for i in range(len(current)):values[name].append(dict(index=start+i,mean_mse=float(mm[i]),sample_mse=float(sm[i]),
                    mean_gap=float(mm[i]-bm[i]),sample_gap=float(sm[i]-bs[i]),sample_prediction_change=float(change[i])))
        metrics={name:{k:float(np.mean([r[k] for r in rows])) for k in rows[0] if k!='index'} for name,rows in values.items()}
        if kind=='constant':assert all(v['sample_prediction_change']==0 for v in metrics.values())
        report['results'].append(dict(seed=seed,variant=kind,checkpoint_sha256=file_hash(path),base=base_rows,metrics=metrics,rows=values))
        a.output.write_text(json.dumps(report,indent=2)+'\n')
        print(seed,kind,metrics,flush=True)
if __name__=='__main__':main()
