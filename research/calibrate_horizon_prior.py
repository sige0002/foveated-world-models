"""One train-only scalar curvature control for physical-time prior interpolation."""
import argparse
import hashlib
import json
from pathlib import Path
import torch


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ratio(data):
    r=torch.tensor([v['dt']/1.2 for v in data['rows']],dtype=torch.float64)
    return torch.where((r-1).abs()<1e-4,torch.ones_like(r),r)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    root=Path('runs/reference-idm42-nested512-prior10000')
    paths={'train':root/'train.pt','dev64':root/'validation.pt',
           'observed96':Path('runs/reference-idm42-observed96-prior10000/validation.pt')}
    for path in paths.values():
        manifest=json.loads((path.parent/'cache_manifest.json').read_text())
        assert file_hash(path)==manifest['cache_sha256'][path.stem]
    protocol={'primary':'Fixed linear time interpolation b+(1-r)(current-b), r=dt/1.2.',
              'secondary':'Add c*r*(1-r)*(current-b), scalar c fitted by train-only least squares. Both preserve b at1.2s and current at0. No eval tuning.',
              'scope':'Current-only diagnostic; time query uses actual pair duration, not unknown future termination. Short intervals are tail-confounded.',
              'source_sha256':{k:file_hash(v) for k,v in paths.items()}}
    (args.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    train=torch.load(paths['train'],weights_only=True)
    r=ratio(train)
    numerator=denominator=0.
    for i in range(len(r)):
        delta=(train['current'][i]-train['baseline'][i]).double()
        error=(train['future'][i]-train['baseline'][i]).double()-(1-r[i])*delta
        design=r[i]*(1-r[i])*delta
        numerator+=float((design*error).sum())
        denominator+=float(design.square().sum())
    c=numerator/denominator if denominator>0 else 0.
    (args.output/'coefficient.json').write_text(json.dumps({'curvature':c,'normal_equation_residual':numerator-c*denominator},indent=2)+'\n')
    del train
    results={'curvature':c,'splits':{}}
    for split,path in paths.items():
        data=torch.load(path,weights_only=True);r=ratio(data);rows=[]
        for i in range(len(r)):
            h,b,y=[data[k][i] for k in ('current','baseline','future')]
            linear=b+float(1-r[i])*(h-b)
            curved=b+float(1-r[i]+c*r[i]*(1-r[i]))*(h-b)
            rows.append({'episode':data['rows'][i]['episode'],'dt':data['rows'][i]['dt'],
                         'prior':float((b-y).square().mean()),'linear':float((linear-y).square().mean()),
                         'curved':float((curved-y).square().mean())})
        strata={}
        for name,subset in [('all',rows),('nominal',[v for v in rows if abs(v['dt']-1.2)<1e-4]),('short',[v for v in rows if v['dt']<1.2-1e-4])]:
            strata[name]={'count':len(subset),**{k:sum(v[k] for v in subset)/len(subset) for k in ('prior','linear','curved')}}
        results['splits'][split]={'strata':strata,'rows':rows}
        print(split,strata,flush=True)
    (args.output/'results.json').write_text(json.dumps(results,indent=2)+'\n')


if __name__=='__main__':
    main()
