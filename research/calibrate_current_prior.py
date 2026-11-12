"""Closed-form current-only calibration controls fitted on nested train512."""
import hashlib
import json
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(2**20),b''):h.update(block)
    return h.hexdigest()


def fit(b,h,y):
    b,h,y=[v.double() for v in (b,h,y)]
    assert b.ndim==3 and b.shape==h.shape==y.shape
    bm=b.mean((0,1));ym=y.mean((0,1));bc=b-bm;yc=y-ym
    variance=bc.square().sum((0,1));cov=(bc*yc).sum((0,1))
    slope=torch.where(variance>1e-12,cov/variance.clamp_min(1e-12),torch.zeros_like(cov))
    result={'scalar':{'slope':(b*y).sum()/b.square().sum().clamp_min(1e-12)},
            'channel_affine':{'slope':slope,'bias':ym-slope*bm},
            'patch_scalar':{'slope':((b*y).sum((0,2))/b.square().sum((0,2)).clamp_min(1e-12))[:,None]}}
    for name,axes in [('global_mix',(0,1,2)),('patch_mix',(0,2))]:
        bb=b.square().sum(axes);hh=h.square().sum(axes);bh=(b*h).sum(axes)
        gram=torch.stack((bb,bh,bh,hh),-1).reshape(*bb.shape,2,2)
        rhs=torch.stack(((b*y).sum(axes),(h*y).sum(axes)),-1)
        weight=(torch.linalg.pinv(gram,rtol=1e-12)@rhs.unsqueeze(-1)).squeeze(-1)
        result[name]={'weight':weight,'gram_condition':torch.linalg.cond(gram)}
    assert all(torch.isfinite(v).all() for m in result.values() for k,v in m.items() if k!='gram_condition')
    return result


def predict(name,parameters,b,h):
    # No future target or posterior code is an inference input.
    b,h=b.double(),h.double()
    if name=='global_mix':
        w=parameters['weight'];return w[0]*b+w[1]*h
    if name=='patch_mix':
        w=parameters['weight'];return w[:,0,None]*b+w[:,1,None]*h
    return parameters['slope']*b+parameters.get('bias',0.)


@torch.inference_mode()
def main():
    torch.set_num_threads(2)
    output=ROOT/'research/results/current-prior-calibration512-20261113'
    if output.exists():raise FileExistsError(output)
    base=ROOT/'runs/reference-idm42-nested512-prior10000'
    observed=ROOT/'runs/reference-idm42-observed96-prior10000'
    sources={};data={}
    for split,root,source_split in [('train',base,'train'),('validation',base,'validation'),('observed96',observed,'validation')]:
        mp=root/'cache_manifest.json';meta=json.loads(mp.read_text());path=root/f'{source_split}.pt'
        assert meta['prior_checkpoint_step']==10000 and sha(path)==meta['cache_sha256'][source_split]
        data[split]=torch.load(path,weights_only=True)
        sources[str(path)]=sha(path);sources[str(mp)]=sha(mp)
    assert [len(data[s]['rows']) for s in data]==[512,64,96]
    train_eps={r['episode'] for r in data['train']['rows']}
    assert all(not train_eps&{r['episode'] for r in data[s]['rows']} for s in ('validation','observed96'))
    protocol=dict(primary='channel_affine',diagnostics=['scalar','patch_scalar','global_mix','patch_mix'],
        fitting='Closed-form unregularized train512 least squares; report every prespecified transform, no evaluation-based transform selection.',
        inputs='Frozen final10000 baseline(current,zero code); global/patch mix additionally uses current feature. No sample-specific posterior code at inference.',
        precision='Float64 sufficient statistics/coefficients; output cast to cached float32 before identical MSE reduction; CPU2.',
        scope='Previously observed64/96; calibration controls to assess simple rescaling/correction explanations, not a new model novelty claim.',
        source_sha256=sources,script_sha256=sha(Path(__file__)))
    output.mkdir();(output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    train=data['train'];models=fit(*(train[k].squeeze(1) for k in ('baseline','current','future')))
    torch.save(models,output/'coefficients.pt')
    report={'protocol_sha256':sha(output/'protocol.json'),'coefficients_sha256':sha(output/'coefficients.pt'),'splits':{}}
    for split,d in data.items():
        rows=[]
        for start in range(0,len(d['rows']),8):
            b,h,y=[d[k][start:start+8].squeeze(1) for k in ('baseline','current','future')]
            errors={'prior':(b-y).square().flatten(1).mean(1)}
            for name,model in models.items():
                pred=predict(name,model,b,h).to(y.dtype)
                errors[name]=(pred-y).square().flatten(1).mean(1)
            for j in range(len(b)):
                row=d['rows'][start+j]
                rows.append(dict(episode=row['episode'],index=row['index'],dt=row['dt'],**{k:float(v[j]) for k,v in errors.items()}))
        metrics={k:sum(r[k] for r in rows)/len(rows) for k in errors}
        energy={k:float(d[k].double().square().mean()) for k in ('current','baseline','future')}
        report['splits'][split]=dict(metrics=metrics,mean_feature_energy=energy,rows=rows)
        print(split,metrics,flush=True)
    report['coefficient_summary']={name:{k:dict(min=float(v.min()),max=float(v.max()),mean=float(v.mean())) for k,v in m.items()} for name,m in models.items()}
    for path,digest in sources.items():assert sha(Path(path))==digest
    (output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
