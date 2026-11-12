"""Fixed-endpoint codec training; oracle reconstruction is NOT future prediction."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from research.conditional_residual_codec import ConditionalResidualCodec


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


@torch.inference_mode()
def evaluate(model, data):
    model.eval()
    rows,latents = [],[]
    for start in range(0,len(data['current']),8):
        h,b,y = [data[k][start:start+8] for k in ('current','baseline','future')]
        latent = model.encode(h,b,y)
        reconstructed = model.decode(h,b,latent)
        pca = b+((y-b)@model.basis)@model.basis.T
        zero = model.decode(h,b,torch.zeros_like(latent))
        latents.append(latent)
        errors = {'reconstruction_mse':(reconstructed-y).square().flatten(1).mean(1),
                  'pca_mse':(pca-y).square().flatten(1).mean(1),
                  'zero_code_mse':(zero-y).square().flatten(1).mean(1)}
        for j in range(len(h)):
            rows.append(dict(episode=data['rows'][start+j]['episode'],
                             **{k:float(v[j]) for k,v in errors.items()}))
    latents = torch.cat(latents)
    return {'metrics':{k:sum(r[k] for r in rows)/len(rows) for k in errors},'rows':rows},latents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--observed-cache',type=Path,required=True)
    parser.add_argument('--basis',type=Path,required=True)
    parser.add_argument('--rank',type=int,choices=[32,128],required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=42)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    paths={'train':args.cache/'train.pt','dev64':args.cache/'validation.pt',
           'observed96':args.observed_cache/'validation.pt'}
    hashes={k:digest(p) for k,p in paths.items()}
    for name,path in paths.items():
        manifest=json.loads((path.parent/'cache_manifest.json').read_text())
        assert hashes[name]==manifest['cache_sha256'][path.stem]
    stored_basis=torch.load(args.basis,weights_only=True)
    assert stored_basis['source_train_sha256']==hashes['train']
    basis=stored_basis['basis'][:,:args.rank].contiguous()
    train=torch.load(paths['train'],weights_only=True)
    dev=torch.load(paths['dev64'],weights_only=True)
    assert {r['episode'] for r in train['rows']}.isdisjoint({r['episode'] for r in dev['rows']})
    torch.manual_seed(args.seed)
    model=ConditionalResidualCodec(basis)
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=.01)
    generator=torch.Generator().manual_seed(args.seed+1000)
    history=[]
    started=time.perf_counter()
    for step in range(1,args.steps+1):
        model.train()
        idx=torch.randint(len(train['current']),(8,),generator=generator)
        h,b,y=[train[k][idx] for k in ('current','baseline','future')]
        optimizer.zero_grad()
        loss=(model(h,b,y)-y).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite codec loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        if step%200==0 or step==args.steps:
            result,_=evaluate(model,dev)
            history.append(dict(step=step,train_objective=float(loss.detach()),**result['metrics']))
            print(history[-1],flush=True)
    checkpoint=args.output/'codec.pt'
    torch.save({'state_dict':model.state_dict(),'rank':args.rank,'hidden':256,'context_dim':128,
                'seed':args.seed,'steps':args.steps},checkpoint)
    report={'settings':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
            'source_sha256':hashes,'basis_sha256':digest(args.basis),'checkpoint_sha256':digest(checkpoint),
            'script_sha256':digest(Path(__file__)),'parameters':sum(p.numel() for p in model.parameters()),
            'scope':'Encoder sees true future: reconstruction feasibility only, not prediction or an exact nonlinear decoder-manifold lower bound. Fixed1000 endpoint, reused eval splits. Zero-code is a diagnostic, not independently trained current-only control.',
            'history':history,'splits':{}}
    for split,path in paths.items():
        data=train if split=='train' else dev if split=='dev64' else torch.load(path,weights_only=True)
        result,latent=evaluate(model,data)
        cache=args.output/f'{split}-latents.pt'
        torch.save({'latents':latent,'rows':data['rows'],'source_sha256':hashes[split],
                    'codec_checkpoint_sha256':report['checkpoint_sha256']},cache)
        result['latent_cache_sha256']=digest(cache)
        report['splits'][split]=result
        if split=='train':
            flattened=latent.reshape(-1,args.rank)
            torch.save({'mean':flattened.mean(0),'std':flattened.std(0,unbiased=False),
                        'scope':'Train-only latent statistics; no whitening applied in codec.'},args.output/'train-latent-statistics.pt')
        print(split,result['metrics'],flush=True)
    report['elapsed_seconds']=time.perf_counter()-started
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    main()
