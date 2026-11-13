"""Matched raw/difference IDM fine-tuning with a fresh shared head and fixed milestones.

The trainer deliberately has no test-cache argument. Test scoring is a separate
terminal-only operation gated on completion of all six runs.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import numpy as np
import torch
from torch import nn
from research.train_difference_idm import build,norm,write,atomic_save
from research.train_early_condition import file_hash
from research.early_condition_adapter import state_hash
from research.evaluate_prediction_metrics import metrics


def initialize_ablation(model,train,representation,seed):
    """No trained reference-head weights survive; paired seeds share new weights."""
    if representation not in ('difference','raw'):raise ValueError(representation)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        head=nn.Sequential(nn.Linear(model.idm.encoder.context_dim,256),nn.GELU(),nn.Linear(256,32))
    model.idm.head=head
    model.idm.representation=representation
    features=train['query']-train['reference'] if representation=='difference' else train['query']
    scale=features.square().mean().sqrt().clamp_min(1e-3)
    model.idm.scale.copy_(scale)
    return float(scale)


def make_batches(seed,steps,size):
    g=torch.Generator().manual_seed(seed+1000)
    return torch.stack([torch.randint(size,(8,),generator=g) for _ in range(steps)])


def optimizer_for(model):
    groups=[]
    parts={'encoder':model.idm.encoder,'head':model.idm.head,'decoder':model.decoder}
    for name,part in parts.items():
        lr={'encoder':1e-5,'head':1e-4,'decoder':3e-5}[name]
        for decay in (True,False):
            ps=[v for v in part.parameters() if v.requires_grad and (v.ndim>=2)==decay]
            if ps:groups.append({'params':ps,'lr':lr,'weight_decay':.01 if decay else 0.,'name':name})
    return torch.optim.AdamW(groups)


def save_training(path,model,optimizer,step,history,protocol,training_seconds):
    atomic_save(path,{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
        'step':step,'history':history,'protocol':protocol,'training_seconds':training_seconds,
        'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state() if next(model.parameters()).is_cuda else None})


def restore_training(path,model,optimizer,protocol):
    saved=torch.load(path,weights_only=True,map_location='cpu')
    if saved['protocol']!=protocol:raise ValueError('Resume protocol differs')
    model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
    torch.set_rng_state(saved['cpu_rng'])
    if saved['cuda_rng'] is not None:torch.cuda.set_rng_state(saved['cuda_rng'])
    return saved['step'],saved['history'],saved['training_seconds']


@torch.no_grad()
def evaluate(model,data,device,batch=8):
    model.eval();out={}
    for split,d in data.items():
        codes=torch.cat([model.idm(d['current'][i:i+batch].to(device),d['future'][i:i+batch].to(device)).cpu()
                         for i in range(0,len(d['rows']),batch)])
        rows=[]
        for i in range(0,len(d['rows']),batch):
            x,y=[d[k][i:i+batch].to(device) for k in ('current','future')]
            pred=model.predict(x,codes[i:i+batch].to(device));vals=metrics(pred,y,x)
            wrong=model.predict(x,codes.roll(1,0)[i:i+batch].to(device))
            vals['swapped_mse']=(wrong-y).square().flatten(1).mean(1)
            same_z=model.idm(x,x)  # Raw control must compute Q(C,C), not head(0).
            vals['same_image_code_mse']=(model.predict(x,same_z)-y).square().flatten(1).mean(1)
            for j in range(len(x)):
                rows.append({'source_row':d['rows'][i+j],**{k:float(v[j]) for k,v in vals.items()}})
        out[split]={'metrics':{k:float(np.mean([r[k] for r in rows])) for k in vals},'rows':rows,
            'code_variance_sum':float(codes.var(0,unbiased=False).sum()),
            'code_mean_squared_norm':float(codes.square().sum(-1).mean())}
        print('EVAL',split,out[split]['metrics'],flush=True)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'runs/layerwise-gpu-inputs-20261113')
    p.add_argument('--checkpoint',type=Path,default=Path('/data2/yuki_sadasue/runs/foveated-world-models/stage1-robomind-main/checkpoints/final.ckpt'))
    p.add_argument('--selection',type=Path,default=ROOT/'research/results/difference-idm-ablation-20261113/test-selection.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--representation',choices=['difference','raw'],required=True)
    p.add_argument('--seed',type=int,default=42);p.add_argument('--steps',type=int,default=1600)
    p.add_argument('--eval-steps',type=int,nargs='+',default=[400,800,1600])
    p.add_argument('--device',default='cuda:0');p.add_argument('--microbatch',type=int,default=8)
    p.add_argument('--resume',action='store_true');p.add_argument('--smoke',action='store_true')
    p.add_argument('--stop-after',type=int,help='Controlled interruption after saving restart state; total budget unchanged')
    a=p.parse_args()
    if not torch.cuda.is_available() or not a.device.startswith('cuda'):p.error('CUDA required')
    if a.steps<1 or a.microbatch<1 or 8%a.microbatch:p.error('Positive steps; microbatch must divide8')
    if sorted(set(a.eval_steps))!=a.eval_steps or a.eval_steps[-1]!=a.steps or a.eval_steps[0]<1:p.error('Sorted unique milestones ending at steps required')
    if a.stop_after is not None and not 1<=a.stop_after<a.steps:p.error('stop-after must precede terminal step')
    if a.output.exists() and not a.resume:p.error('Use a new output directory')
    torch.cuda.set_device(a.device);torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    data,model,hashes,options=build(a.source.resolve(),a.checkpoint)
    scale=initialize_ablation(model,data['train'],a.representation,a.seed)
    assert all(m.p==0 for m in model.modules() if isinstance(m,nn.Dropout))
    model.to(a.device).train()
    parts={'encoder':model.idm.encoder,'head':model.idm.head,'decoder':model.decoder}
    initial_hashes={k:state_hash(v.state_dict()) for k,v in parts.items()}
    batches=make_batches(a.seed,a.steps,len(data['train']['rows']))
    for path in (Path(__file__),ROOT/'research/train_difference_idm.py',ROOT/'research/train_early_condition.py',
                 ROOT/'research/evaluate_prediction_metrics.py',ROOT/'research/idm/difference_idm.py',
                 ROOT/'research/idm/load_difference_idm.py',ROOT/'vendor/lawam/latent_action_model/core/utils/lam_encoder.py',
                 ROOT/'vendor/lawam/latent_action_model/core/utils/lam_decoder.py',ROOT/'vendor/lawam/latent_action_model/core/utils/modules.py'):
        hashes[str(path)]=file_hash(path)
    protocol={'representation':a.representation,'seed':a.seed,'steps':a.steps,'eval_steps':a.eval_steps,
        'batch_size':8,'microbatch':a.microbatch,'lr':{'encoder':1e-5,'head':1e-4,'decoder':3e-5},
        'weight_decay_matrix':.01,'weight_decay_bias_norm':0.,'gradient_clip':1.,'smoke_only':a.smoke,
        'objective':'Deterministic MSE; no VAE, KL or state auxiliary','precision':'float32 highest; no autocast',
        'device':torch.cuda.get_device_name(),'torch':str(torch.__version__),'options':options,'scale':scale,
        'initial_hashes':initial_hashes,'source_sha256':hashes,'batch_hash':state_hash({'batches':batches}),
        'test_selection_sha256':file_hash(a.selection),'calibration':'Fixed scalar RMS of each representation on train512 only',
        'trainable_parameters':{k:sum(p.numel() for p in v.parameters()) for k,v in parts.items()},
        'scope':'Paired raw/difference, new random mean-head per seed. Pretrained Q and decoder10000 shared. Frozen DINO. True-future reconstruction. Same steps, not same FLOPs. No test scoring in training.'}
    optimizer=optimizer_for(model);start_step=0;history=[];training_seconds=0.
    if a.resume:
        start_step,history,training_seconds=restore_training(a.output/'latest.pt',model,optimizer,protocol)
    else:
        a.output.mkdir(parents=True);write(a.output/'protocol.json',protocol)
        (a.output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
        with torch.no_grad():
            x,y=[data['train'][k][:2].to(a.device) for k in ('current','future')]
            z,q,r=model.idm(x,y,return_queries=True)
            cached=data['train']['query'][:2]
            if r is not None:cached=cached-data['train']['reference'][:2]
            expected=model.idm.head(cached.to(a.device)/scale)
            delta=float((z-expected).abs().max());assert delta<1e-3,delta
            same=model.idm(x,x)
            write(a.output/'preflight.json',{'cached_query_code_max_abs':delta,
                'initial_two_train_mse':float((model.predict(x,z)-y).square().mean()),
                'initial_code_norm':float(z.norm()),'same_image_code_norm':float(same.norm())})
    evaldata=data
    if a.smoke:evaldata={s:{k:(v[:8] if isinstance(v,(torch.Tensor,list)) else v) for k,v in d.items()} for s,d in data.items()}
    def milestone(step):
        checkpoint=a.output/f'model-step{step}.pt';resultpath=a.output/f'evaluation-step{step}.json'
        if resultpath.exists():return
        atomic_save(checkpoint,{'model':model.state_dict(),'protocol':protocol,'step':step})
        ev=evaluate(model,evaldata,a.device,a.microbatch)
        write(resultpath,{'step':step,'evaluation':ev,'checkpoint_sha256':file_hash(checkpoint),
            'training_seconds':training_seconds,'smoke_only':a.smoke})
    # A crash after saving latest but before finishing milestone scoring is recoverable.
    if start_step in a.eval_steps:milestone(start_step)
    train={k:data['train'][k].to(a.device) for k in ('current','future')}
    torch.cuda.reset_peak_memory_stats()
    for step in range(start_step+1,a.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize();tick=time.perf_counter();loss_value=0.
        for micro,idx in enumerate(batches[step-1].split(a.microbatch)):
            x,y=train['current'][idx],train['future'][idx]
            z,q,r=model.idm(x,y,return_queries=True)
            if step==1 and micro==0:
                q.retain_grad()
                if r is not None:r.retain_grad()
            loss=(model.predict(x,z)-y).square().mean()*(len(idx)/8)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();loss_value+=float(loss.detach())
            if step==1 and micro==0:
                assert q.grad is not None and q.grad.norm()>0
                if r is not None:
                    assert r.grad is not None and r.grad.norm()>0
                    torch.testing.assert_close(q.grad,-r.grad,rtol=0,atol=0)
                layers=[norm(layer.parameters()) for layer in model.idm.encoder.QFormer.layers]
                assert all(v>0 for v in layers)
                write(a.output/'gradient_check.json',{'transition_grad_norm':float(q.grad.norm()),
                    'reference_grad_norm':float(r.grad.norm()) if r is not None else None,
                    'encoder_layer_grad_norms':layers,'component_grad_norms':{k:norm(v.parameters()) for k,v in parts.items()}})
        gradient_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step();torch.cuda.synchronize();training_seconds+=time.perf_counter()-tick
        if step<=5 or step%20==0 or step==a.steps or step==a.stop_after:
            history.append({'step':step,'mse':loss_value,'gradient_norm':float(gradient_norm),
                'training_seconds':training_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
            write(a.output/'progress.json',history);print(a.representation,a.seed,history[-1],flush=True)
        if step%100==0 or step in a.eval_steps or step==a.stop_after:
            save_training(a.output/'latest.pt',model,optimizer,step,history,protocol,training_seconds)
        if step in a.eval_steps:milestone(step)
        if step==a.stop_after:
            print('INTERRUPTED AT REQUESTED STEP',step,flush=True);return
    final_hashes={k:state_hash(v.state_dict()) for k,v in parts.items()}
    changed={k:initial_hashes[k]!=final_hashes[k] for k in parts};assert all(changed.values())
    write(a.output/'results.json',{'complete':True,'smoke_only':a.smoke,'changed':changed,
        'final_hashes':final_hashes,'training_seconds':training_seconds,'history':history,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'evaluations':{str(s):file_hash(a.output/f'evaluation-step{s}.json') for s in a.eval_steps}})
    print('DONE',a.representation,a.seed,flush=True)


if __name__=='__main__':main()
