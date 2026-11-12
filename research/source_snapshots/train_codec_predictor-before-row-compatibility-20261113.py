"""Predict a frozen nonlinear codec latent from current features and fixed IDM z."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'vendor/lawam'))
import torch
from torch import nn
from research.conditional_residual_codec import ConditionalResidualCodec
from latent_action_model.core.utils.lam_decoder import AdaLNBlock
from latent_action_model.core.utils.pos_embs import Fixed2DPositionalEncoding


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CodecPredictor(nn.Module):
    def __init__(self, codec, mean, scale, code_dim=32, dim=128, layers=2, grid=16):
        super().__init__()
        self.codec=codec.requires_grad_(False).eval()
        channels,rank=codec.basis.shape
        self.register_buffer('latent_mean',mean.clone())
        self.register_buffer('latent_scale',scale.clamp_min(.05).clone())
        self.state_proj=nn.Linear(rank,dim)
        self.current_proj=nn.Linear(channels,dim)
        self.action_proj=nn.Linear(code_dim,dim)
        self.time_proj=nn.Sequential(nn.Linear(3,dim),nn.SiLU(),nn.Linear(dim,dim))
        self.position=Fixed2DPositionalEncoding(dim,grid,grid)
        self.blocks=nn.ModuleList([AdaLNBlock(dim,4,4,0.) for _ in range(layers)])
        self.output=nn.Linear(dim,rank)
        nn.init.zeros_(self.output.weight);nn.init.zeros_(self.output.bias)

    def velocity(self,state,time,current,idm):
        t=time.reshape(-1,1).to(current)
        condition=self.action_proj(idm.squeeze(1))+self.time_proj(torch.cat((t,torch.sin(torch.pi*t),torch.cos(torch.pi*t)),dim=-1))
        hidden=self.position(self.state_proj(state[:,0])+self.current_proj(current[:,0]))
        for block in self.blocks:
            hidden=block(hidden,condition)
        return self.output(hidden).unsqueeze(1)

    def generate(self,current,idm,*,noise=None):
        shape=(*current.shape[:-1],len(self.latent_mean))
        if noise is None:
            state=current.new_zeros(shape)
            return self.velocity(state,current.new_zeros(len(current)),current,idm)
        if noise.shape!=shape:
            raise ValueError('Noise shape differs from codec latent')
        state=noise
        for step in range(4):
            state=state+self.velocity(state,current.new_full((len(current),),step/4),current,idm)/4
        return state

    def decode(self,current,baseline,normalized_latent):
        return self.codec.decode(current,baseline,normalized_latent*self.latent_scale+self.latent_mean)


@torch.inference_mode()
def evaluate(model,data,variant):
    model.eval()
    generator=torch.Generator().manual_seed(2026)
    flow=variant.startswith('flow')
    sigma=.1 if variant=='flow_low_noise' else 1.
    rows=[]
    for start in range(0,len(data['current']),8):
        h,b,y,z=[data[k][start:start+8] for k in ('current','baseline','future','codes')]
        states=[]
        for _ in range(4 if flow else 1):
            noise=sigma*torch.randn((*h.shape[:-1],len(model.latent_mean)),generator=generator) if flow else None
            states.append(model.generate(h,z,noise=noise))
        predictions=torch.stack([model.decode(h,b,s) for s in states])
        mean_latent_decoded=model.decode(h,b,torch.stack(states).mean(0))
        errors={'sample_mse':(predictions-y).square().flatten(2).mean(2).mean(0),
                'mean_mse':(predictions.mean(0)-y).square().flatten(1).mean(1),
                'decode_mean_latent_mse':(mean_latent_decoded-y).square().flatten(1).mean(1)}
        for j in range(len(h)):
            row=data['rows'][start+j]
            rows.append({k:row[k] for k in ('episode','index','endpoint_indices','dt')}|{k:float(v[j]) for k,v in errors.items()})
    return {'metrics':{k:sum(r[k] for r in rows)/len(rows) for k in errors},'rows':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codec',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--variant',choices=['direct_feature','direct_latent','flow','flow_low_noise'],required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4)
    root=ROOT/'runs/reference-idm42-nested512-prior10000'
    observed=ROOT/'runs/reference-idm42-observed96-prior10000'
    paths={'train':root/'train.pt','dev64':root/'validation.pt','observed96':observed/'validation.pt'}
    data={k:torch.load(p,weights_only=True) for k,p in paths.items()}
    hashes={k:digest(p) for k,p in paths.items()}
    manifests={k:json.loads((p.parent/'cache_manifest.json').read_text()) for k,p in paths.items()}
    for k,p in paths.items():
        assert hashes[k]==manifests[k]['cache_sha256'][p.stem]
        assert manifests[k]['prior_checkpoint_sha256']==manifests['train']['prior_checkpoint_sha256']
        assert manifests[k]['prior_weights_sha256']==manifests['train']['prior_weights_sha256']
    episodes=[{r['episode'] for r in d['rows']} for d in data.values()]
    assert all(episodes[i].isdisjoint(episodes[j]) for i in range(3) for j in range(i+1,3))
    codec_path=args.codec/'codec.pt'
    stored=torch.load(codec_path,weights_only=True)
    codec=ConditionalResidualCodec(stored['state_dict']['basis'],stored['hidden'],stored['context_dim'])
    codec.load_state_dict(stored['state_dict'],strict=True)
    targets=torch.load(args.codec/'train-latents.pt',weights_only=True)
    assert targets['codec_checkpoint_sha256']==digest(codec_path)
    assert targets['source_sha256']==hashes['train']
    identity=lambda rows:[(r['episode'],r['index'],r['endpoint_indices']) for r in rows]
    assert identity(targets['rows'])==identity(data['train']['rows'])
    statistics=torch.load(args.codec/'train-latent-statistics.pt',weights_only=True)
    # Recompute train-only calibration rather than trusting an unbound sidecar.
    flat=targets['latents'].reshape(-1,stored['rank'])
    torch.testing.assert_close(flat.mean(0),statistics['mean'],rtol=0,atol=0)
    torch.testing.assert_close(flat.std(0,unbiased=False),statistics['std'],rtol=0,atol=0)
    torch.manual_seed(42)
    model=CodecPredictor(codec,statistics['mean'],statistics['std'])
    normalized=(targets['latents']-model.latent_mean)/model.latent_scale
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=.01)
    generator=torch.Generator().manual_seed(1042)
    history=[];started=time.perf_counter()
    for step in range(1,401):
        model.train()
        idx=torch.randint(len(normalized),(8,),generator=generator)
        h,b,y,z=[data['train'][k][idx] for k in ('current','baseline','future','codes')]
        optimizer.zero_grad()
        if args.variant.startswith('flow'):
            sigma=.1 if args.variant=='flow_low_noise' else 1.
            noise=sigma*torch.randn_like(normalized[idx])
            t=torch.rand(len(idx));weight=t[:,None,None,None]
            mixed=(1-weight)*noise+weight*normalized[idx]
            loss=(model.velocity(mixed,t,h,z)-(normalized[idx]-noise)).square().mean()
        else:
            prediction=model.generate(h,z)
            loss=((model.decode(h,b,prediction)-y).square().mean() if args.variant=='direct_feature'
                  else (prediction-normalized[idx]).square().mean())
        if not torch.isfinite(loss):
            raise RuntimeError('Nonfinite predictor loss')
        loss.backward();torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);optimizer.step()
        if step%200==0:
            result=evaluate(model,data['dev64'],args.variant)
            history.append(dict(step=step,objective=float(loss.detach()),**result['metrics']))
            print(history[-1],flush=True)
    checkpoint=args.output/'predictor.pt'
    torch.save({'state_dict':model.state_dict(),'variant':args.variant,'rank':stored['rank'],'steps':400,'seed':42},checkpoint)
    report={'variant':args.variant,'rank':stored['rank'],'steps':400,'seed':42,'history':history,
            'source_sha256':hashes,'codec_sha256':digest(codec_path),'checkpoint_sha256':digest(checkpoint),
            'script_sha256':digest(Path(__file__)),'parameters_trainable':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'scope':'Predicted codec latent uses current+fixed future-conditioned IDM z only; no encoded eval future w. Still posterior reconstruction, not executable action prediction. Reused dev64/observed96; one seed fixed400. Extra1000-step codec training cost, different capacity from original full-space models.',
            'results':{k:evaluate(model,data[k],args.variant) for k in ('dev64','observed96')}}
    report['elapsed_seconds']=time.perf_counter()-started
    (args.output/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v['metrics'] for k,v in report['results'].items()},flush=True)


if __name__=='__main__':
    main()
