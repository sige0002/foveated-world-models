"""Frozen corrected-model diagnostics: constant controls, layer variation, loss gradients."""
import argparse
import functools
import json
import os
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['LAM_ENABLE_MANUAL_WANDB']='0'
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset
from train_stage1 import load_config,check_resume_contract
from vendor.dataset.lerobot_robomind import LeRobotSource,LeRobotRoboMINDDataset,lerobot_collate,minmax
from latent_action_model.core.lam_lightinng import VJEPA_LAM
from latent_action_model.core.utils.utils import eef_reconstruction_loss
from research.train_early_condition import file_hash
from research.investigate_native_posterior import posterior_stats,geometry


def select(ds,n):
    groups={}
    for idx,(ep,_,_,tail) in enumerate(ds.pairs):
        if not tail:groups.setdefault(ep,[]).append(idx)
    episodes=sorted(groups)
    chosen=np.linspace(0,len(episodes)-1,min(n,len(episodes)),dtype=int)
    return [groups[episodes[i]][len(groups[episodes[i]])//2] for i in chosen]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():p.error('Output must be new')
    torch.set_num_threads(2);torch.manual_seed(42)
    cfg=load_config(a.run/'resolved_config.yaml');mc=cfg['model'];dc=cfg['data']
    source=LeRobotSource(dc['root'])
    options={k:mc[k] for k in ('image_hw','max_state_dim','num_embodiments')}
    options.update({k:dc[k] for k in ('horizon_seconds','tolerance_seconds')})
    datasets={s:LeRobotRoboMINDDataset(source,s,**options) for s in ('train','validation')}
    saved=torch.load(a.checkpoint,weights_only=True,map_location='cpu',mmap=True)
    check_resume_contract(saved,datasets['train'].contract)
    model=VJEPA_LAM(**mc).eval().requires_grad_(False)
    initial={n:v.detach().clone() for n,v in model.state_dict().items()
             if n in ('lam.encoder.QFormer.queries','lam.encoder.out_proj.weight','lam.vq.mu.weight','lam.vq.logvar.weight')}
    model.load_state_dict(saved['state_dict'],strict=True)
    changed={n:float((saved['state_dict'][n]-v).norm()) for n,v in initial.items()}
    opt=saved['optimizer_states'][0];steps=[float(s['step']) for s in opt['state'].values() if 'step' in s]
    report={'step':saved['global_step'],'checkpoint_sha256':file_hash(a.checkpoint),
            'source_sha256':file_hash(Path(__file__)),'selection':'one middle full-horizon pair per episode; 128 evenly spaced train episodes, all160 validation episodes',
            'optimizer_state':{'parameter_tensors_with_state':len(opt['state']),'min_step':min(steps),'max_step':max(steps),
                               'learning_rates':[g['lr'] for g in opt['param_groups']]},
            'selected_weight_change_from_seed42_initialization_norm':changed,
            'scope':'Frozen checkpoint, no updates, float32 highest, corrected images; gradient probes are local sensitivities, not causal training ablations.',
            'splits':{},'state_target_range':{}}
    del saved,opt,initial
    model=model.cuda();torch.set_float32_matmul_precision('highest');lam=model.lam
    cache={}
    for split,ds in datasets.items():
        # All pairs for exact state-range feasibility, without decoding video.
        deltas=[]
        for ep,first,last,_ in ds.pairs:
            st=minmax(ds._arrays(ep)['states'][[first,last]],source.minimum,source.maximum)
            deltas.append(st[1]-st[0])
        delta=np.stack(deltas);floor=np.maximum(np.abs(delta)-1,0)**2
        report['state_target_range'][split]={'pairs':len(delta),'coordinates_over_one':int((np.abs(delta)>1).sum()),
            'pairs_over_one':int((np.abs(delta)>1).any(1).sum()),'tanh_unavoidable_mse_floor':float(floor.mean()),
            'zero_prediction_mse':float((delta**2).mean()),'per_coordinate_floor':floor.mean(0).tolist()}
        ids=select(ds,128 if split=='train' else 160)
        c={k:[] for k in ('x','y','mu','lv','nodes','state','delta','mask','emb','rows')}
        layers={str(i):[] for i in range(len(lam.encoder.QFormer.layers))}
        hooks=[layer.register_forward_hook(lambda m,inp,out,i=i:layers[str(i)].append(out[:,-1:].detach().cpu()))
               for i,layer in enumerate(lam.encoder.QFormer.layers)]
        loader=DataLoader(Subset(ds,ids),batch_size=2,num_workers=2,collate_fn=functools.partial(lerobot_collate,max_state_dim=mc['max_state_dim']))
        with torch.no_grad():
            for batch in loader:
                batch=model.transfer_batch_to_device(batch,torch.device('cuda'),0)
                features=lam.extract_vision_features(batch['videos'])
                nodes=lam.encoder(features)
                mu,lv=lam.vq._encode(nodes)
                values={'x':features[:,:1],'y':features[:,1:],'mu':mu,'lv':lv,'nodes':nodes,
                        'state':batch['states'][:,:1],'delta':batch['delta_proprio'],
                        'mask':batch['state_mask'].any(1),'emb':batch['embodiment_ids']}
                for key,value in values.items():c[key].append(value.cpu())
            for hook in hooks:hook.remove()
        c={k:torch.cat(v) for k,v in c.items() if k!='rows'}
        c['rows']=[{'episode':int(ds.rows[ds.pairs[i][0]]['episode_index']),'pair_index':i} for i in ids]
        assert len({r['episode'] for r in c['rows']})==len(ids)
        cache[split]=c
        report['splits'][split]={'posterior':posterior_stats(c['mu'],c['lv'],lam.vq.beta),
            'layer_query_geometry':{k:geometry(torch.cat(v)) for k,v in layers.items()},'node_geometry':geometry(c['nodes'])}
        print('encoded',split,len(ids),flush=True)
    mean=cache['train']['mu'].mean(0,keepdim=True).cuda()
    with torch.no_grad():
        for split,c in cache.items():
            rows=[]
            for start in range(0,len(c['rows']),2):
                sl=slice(start,start+2);x,y,mu,lv=[c[k][sl].cuda() for k in ('x','y','mu','lv')]
                state,delta,mask,emb=[c[k][sl].cuda() for k in ('state','delta','mask','emb')]
                codes={'native':mu,'mean_constant':mean.expand_as(mu),'zero':torch.zeros_like(mu),
                       'other_episode':c['mu'].roll(1,0)[sl].cuda()}
                vals={}
                for name,z in codes.items():
                    pred=lam.decoder(x,z);sp=lam.state_decoder(z,state,emb)
                    assert sp.shape==delta.shape==mask.shape
                    vals[name]={'mse':(pred-y).square().flatten(1).mean(1),
                                'smooth_l1':F.smooth_l1_loss(pred,y,beta=.1,reduction='none').flatten(1).mean(1),
                                'state_mse':((sp-delta).square()*mask).sum(1)/mask.sum(1)}
                for j in range(len(x)):
                    rows.append({**c['rows'][start+j],'mu':c['mu'][start+j].flatten().tolist(),'state':c['state'][start+j].flatten().tolist(),'delta':c['delta'][start+j].flatten().tolist(),**{n:{k:float(v[j]) for k,v in d.items()} for n,d in vals.items()}})
            report['splits'][split]['rows']=rows
            report['splits'][split]['metrics']={n:{k:sum(r[n][k] for r in rows)/len(rows) for k in ('mse','smooth_l1','state_mse')} for n in codes}
            print('evaluated',split,report['splits'][split]['metrics'],flush=True)
    report['node_gradient_probes']=[]
    c=cache['train']
    for start in (0,32,64,96):
        sl=slice(start,start+2);x,y=[c[k][sl].cuda() for k in ('x','y')]
        nodes=c['nodes'][sl].cuda().detach().requires_grad_()
        mu,lv=lam.vq._encode(nodes)
        gen=torch.Generator(device='cuda').manual_seed(1942+start)
        z=mu+(.5*lv).exp()*torch.randn(mu.shape,device='cuda',generator=gen)
        pred=lam.decoder(x,z)
        state,delta,mask,emb=[c[k][sl].cuda() for k in ('state','delta','mask','emb')]
        sp=lam.state_decoder(z,state,emb)
        losses={'visual':F.smooth_l1_loss(pred,y,beta=.1),
                'state':mc['lambda_aux']*eef_reconstruction_loss(sp,delta,mask,state_loss_type=mc['state_loss_type']),
                'kl':lam.vq.beta*lam.vq._kl_divergence(mu,lv).mean()}
        grads={n:torch.autograd.grad(l,nodes,retain_graph=True)[0] for n,l in losses.items()}
        report['node_gradient_probes'].append({'train_indices':[start,start+1],
            'losses':{k:float(v.detach()) for k,v in losses.items()},
            'gradient_norms':{k:float(v.norm()) for k,v in grads.items()},
            'visual_state_cosine':float(F.cosine_similarity(grads['visual'].flatten(),grads['state'].flatten(),dim=0)),
            'visual_kl_cosine':float(F.cosine_similarity(grads['visual'].flatten(),grads['kl'].flatten(),dim=0))})
    report['complete']=True
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('COMPLETE',a.output,flush=True)


if __name__=='__main__':main()
