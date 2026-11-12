"""Oracle local feature-translation diagnostic, never a predictive accuracy claim."""
import argparse,hashlib,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for x in iter(lambda:f.read(2**20),b''):h.update(x)
    return h.hexdigest()


def solve_coefficients(basis,error,bound=.5):
    gram=torch.einsum('...di,...dj->...ij',basis,basis)
    rhs=torch.einsum('...di,...d->...i',basis,error)
    free=(torch.linalg.pinv(gram,rtol=1e-10)@rhs.unsqueeze(-1)).squeeze(-1)
    candidates=[torch.zeros_like(free),free.clamp(-bound,bound)]
    for fixed in (0,1):
        other=1-fixed
        for sign in (-1.,1.):
            c=torch.zeros_like(free);c[...,fixed]=sign*bound
            denom=gram[...,other,other]
            c[...,other]=torch.where(denom>1e-14,
                (rhs[...,other]-gram[...,other,fixed]*c[...,fixed])/denom.clamp_min(1e-14),
                torch.zeros_like(denom)).clamp(-bound,bound)
            candidates.append(c)
    c=torch.stack(candidates,-2)
    costs=torch.einsum('...ki,...ij,...kj->...k',c,gram,c)-2*(c*rhs.unsqueeze(-2)).sum(-1)
    index=costs.argmin(-1)
    best=c.gather(-2,index[...,None,None].expand(*index.shape,1,2)).squeeze(-2)
    return free,best


def gradients(features):
    gx=(features[:,1:-1,2:]-features[:,1:-1,:-2])/2
    gy=(features[:,2:,1:-1]-features[:,:-2,1:-1])/2
    return torch.stack((gx,gy),-1)


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path)
    parser.add_argument('--output',type=Path,default=ROOT/'research/results/spatial-tangent-oracle-20261113')
    args=parser.parse_args()
    torch.set_num_threads(2)
    out=args.output
    if out.exists():raise FileExistsError(out)
    base=ROOT/'runs/reference-idm42-nested512-prior10000'
    obs=ROOT/'runs/reference-idm42-observed96-prior10000'
    specs=[('train',base,'train'),('validation',base,'validation'),('observed96',obs,'validation')]
    sources={}
    for _,root,split in specs:
        m=json.loads((root/'cache_manifest.json').read_text());p=root/f'{split}.pt'
        assert sha(p)==m['cache_sha256'][split]
        sources[str(p)]=sha(p)
    model=None
    if args.checkpoint:
        sys.path.insert(0,str(ROOT))
        from research.train_cached_decoders import make_decoder
        saved=torch.load(args.checkpoint,weights_only=True)
        assert saved['variant']=='prior_residual' and saved['steps']==1600 and saved['seed']==42
        model=make_decoder(saved['variant'],saved['dim'],saved['layers'],768,32,16).eval()
        model.load_state_dict(saved['state_dict'],strict=True)
        sources[str(args.checkpoint)]=sha(args.checkpoint)
    protocol=dict(source_sha256=sources,script_sha256=sha(Path(__file__)),
        primary='Current-feature central-difference x/y basis; independent2D coefficient per interior patch, oracle optimized from true future residual to fixed prior. Box [-.5,.5] feature-grid units.',
        diagnostics='Unbounded local coefficients; shared global x/y coefficients; rank2 random directions with current-gradient column norms; prior-feature gradients. Outer border patches remain uncorrected.',
        random_seed=20261113,threads=2,
        error_anchor=str(args.checkpoint) if args.checkpoint else 'Frozen current-only prior',
        scope='Oracle representability of error, not learnability or inference performance. True future fits every coefficient, including validation. No coefficient is a deployable prediction. DINO feature warping need not equal image motion/re-encoding; central differences are only a tangent approximation. Half-patch bound is prespecified and not calibrated physical motion.',
        sources_context=[{'paper':'Unsupervised Learning for Physical Interaction through Video Prediction','url':'https://arxiv.org/abs/1605.07157','read':'NeurIPS PDF method overview of DNA/CDNA/STP; pixel advection, not evidence for DINO equivariance'},
                         {'paper':'Dynamic Filter Networks','url':'https://arxiv.org/abs/1605.09673','read':'NeurIPS PDF method overview of input-dependent local filtering; established design, not novelty claim'}])
    out.mkdir();(out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    report={'protocol_sha256':sha(out/'protocol.json'),'splits':{}}
    gen=torch.Generator().manual_seed(20261113)
    for split,root,source_split in specs:
        data=torch.load(root/f'{source_split}.pt',weights_only=True);rows=[]
        for start in range(0,len(data['rows']),8):
            x,b,y=[data[k][start:start+8].squeeze(1).reshape(-1,16,16,768).double()
                   for k in ('current','baseline','future')]
            anchor=b
            if model is not None:
                sl=slice(start,start+len(x))
                anchor=model(data['current'][sl],data['codes'][sl],baseline=data['baseline'][sl]).squeeze(1).reshape(-1,16,16,768).double()
            error=y-anchor;inner=error[:,1:-1,1:-1];total=error.square().sum((1,2,3))
            inner_sse=inner.square().sum((1,2,3));denom=16*16*768
            basis=gradients(x);prior_basis=gradients(b)
            random=torch.randn(basis.shape,generator=gen,dtype=basis.dtype)
            random=random/random.square().sum(-2,keepdim=True).sqrt().clamp_min(1e-12)
            random=random*basis.square().sum(-2,keepdim=True).sqrt()
            predictions={};coefficient_stats={}
            for name,bb in [('current',basis),('prior',prior_basis),('random',random)]:
                free,boxed=solve_coefficients(bb,inner)
                for suffix,c in [('unbounded',free),('box',boxed)]:
                    remaining=(inner-torch.einsum('...di,...i->...d',bb,c)).square().sum((1,2,3))
                    predictions[name+'_'+suffix]=(total-inner_sse+remaining)/denom
                coefficient_stats[name+'_unbounded_abs_gt_half']=free.abs().gt(.5).double().flatten(1).mean(1)
            gb=basis.reshape(len(x),-1,2);ge=inner.reshape(len(x),-1)
            gf,gc=solve_coefficients(gb,ge)
            for suffix,c in [('unbounded',gf),('box',gc)]:
                remaining=(ge-torch.einsum('bdi,bi->bd',gb,c)).square().sum(1)
                predictions['global_current_'+suffix]=(total-inner_sse+remaining)/denom
            baseline=total/denom
            assert all(torch.isfinite(v).all() and bool((v<=baseline+1e-10).all()) for v in predictions.values())
            for j in range(len(x)):
                row=data['rows'][start+j]
                rows.append(dict(episode=row['episode'],index=row['index'],dt=row['dt'],baseline_mse=float(baseline[j]),
                    oracle_mse={k:float(v[j]) for k,v in predictions.items()},coefficient_diagnostics={k:float(v[j]) for k,v in coefficient_stats.items()}))
        avg=sum(r['baseline_mse'] for r in rows)/len(rows)
        means={k:sum(r['oracle_mse'][k] for r in rows)/len(rows) for k in rows[0]['oracle_mse']}
        report['splits'][split]=dict(count=len(rows),baseline_mse=avg,oracle_mse=means,
            relative_oracle_reduction={k:1-v/avg for k,v in means.items()},rows=rows)
        print(split,report['splits'][split]['relative_oracle_reduction'],flush=True)
    for p,digest in sources.items():assert sha(Path(p))==digest
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
