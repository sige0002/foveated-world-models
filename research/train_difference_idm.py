"""CUDA joint training of reference-difference IDM and original direct decoder.

Images use the immutable DINO cache. Both Q branches remain differentiable.
The encoder/head/decoder are initialized from available trained checkpoints;
this is fine-tuning a new architecture, not training random weights from scratch.
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
import yaml
from latent_action_model.core.utils.lam_encoder import LAMEncoder
from research.idm.difference_idm import DifferenceIDM,DifferenceWorldModel
from research.train_early_condition import load_inputs,file_hash
from research.early_condition_adapter import state_hash
from research.evaluate_prediction_metrics import metrics


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def atomic_save(path,obj):
    temporary=path.with_suffix('.tmp')
    torch.save(obj,temporary)
    temporary.replace(path)


def build(source,checkpoint):
    data,decoder,hashes,options=load_inputs(source)
    cfg=yaml.safe_load((source/'runs/frozen-model-step10000/resolved_config.yaml').read_text())['model']
    enc=LAMEncoder(context_dim=cfg['dim'],input_dim=768,num_heads=cfg['num_heads'],
        num_queries=cfg['num_queries'],grid_hw=tuple(v//cfg['patch_size'] for v in cfg['image_hw']),
        ffn_expansion_factor=cfg['ffn_expansion_factor'],dropout=cfg['dropout'],code_dim=cfg['code_dim'],
        num_layers=cfg['enc_layers'],num_frames=cfg['num_frames'],modal_mask=cfg['enc_modal_mask'],
        add_state=False,max_state_dim=cfg['max_state_dim'],num_embodiments=cfg['num_embodiments'])
    meta=json.loads((source/'prepare_protocol.json').read_text())
    hashes[str(checkpoint)]=file_hash(checkpoint)
    assert hashes[str(checkpoint)]==meta['checkpoint_sha256']
    weights=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    assert weights['global_step']==10000
    enc.load_state_dict({k[len('lam.encoder.'):]:v for k,v in weights['state_dict'].items() if k.startswith('lam.encoder.')},strict=True)
    # Match the exact pre-out_proj query used to train the reference head.
    enc.out_proj=nn.Identity()
    del weights
    hp=source/'reference-head.pt'
    saved=torch.load(hp,weights_only=True,map_location='cpu')
    hashes[str(hp)]=file_hash(hp)
    head=nn.Sequential(nn.Linear(cfg['dim'],256),nn.GELU(),nn.Linear(256,32))
    head.load_state_dict({'0.weight':saved['state_dict']['head.0.weight'],
        '0.bias':saved['state_dict']['head.0.bias'],
        '2.weight':saved['state_dict']['head.2.weight'][:32],
        '2.bias':saved['state_dict']['head.2.bias'][:32]},strict=True)
    model=DifferenceWorldModel(DifferenceIDM(enc,head,saved['scale']),decoder)
    return data,model,hashes,{'encoder':cfg,'decoder':options}


def norm(parameters):
    grads=[p.grad.detach().float().square().sum() for p in parameters if p.grad is not None]
    return float(torch.stack(grads).sum().sqrt()) if grads else 0.


@torch.no_grad()
def evaluate(model,data,device,batch=2):
    model.eval();out={}
    for split,d in data.items():
        codes=[]
        for start in range(0,len(d['rows']),batch):
            sl=slice(start,start+batch)
            z=model.idm(d['current'][sl].to(device),d['future'][sl].to(device))
            codes.append(z.cpu())
        codes=torch.cat(codes);rows=[]
        for start in range(0,len(d['rows']),batch):
            sl=slice(start,start+batch)
            x,y=[d[k][sl].to(device) for k in ('current','future')]
            z=codes[sl].to(device)
            pred=model.predict(x,z)
            vals=metrics(pred,y,x)
            swapped=model.predict(x,codes.roll(1,0)[sl].to(device))
            vals['swapped_mse']=(swapped-y).square().flatten(1).mean(1)
            # Identical images give zero QUERY DIFFERENCE, not necessarily z=0.
            same_code=model.idm.head(x.new_zeros((len(x),1,model.idm.encoder.context_dim)))
            same_pred=model.predict(x,same_code)
            vals['same_image_code_mse']=(same_pred-y).square().flatten(1).mean(1)
            for j in range(len(x)):
                rows.append({'source_row':d['rows'][start+j],**{k:float(v[j]) for k,v in vals.items()}})
        out[split]={'metrics':{k:float(np.mean([r[k] for r in rows])) for k in vals},
            'code_variance_sum':float(codes.var(0,unbiased=False).sum()),
            'code_mean_squared_norm':float(codes.square().sum(-1).mean()),'rows':rows}
        print('EVAL',split,out[split]['metrics'],flush=True)
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'runs/layerwise-gpu-inputs-20261113')
    p.add_argument('--checkpoint',type=Path,default=Path('/data2/yuki_sadasue/runs/foveated-world-models/stage1-robomind-main/checkpoints/final.ckpt'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['joint','frozen_encoder'],default='joint')
    p.add_argument('--device',default='cuda:0');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=400);p.add_argument('--microbatch',type=int,default=2)
    p.add_argument('--encoder-lr',type=float,default=1e-5)
    p.add_argument('--head-lr',type=float,default=1e-4)
    p.add_argument('--decoder-lr',type=float,default=3e-5)
    p.add_argument('--smoke',action='store_true');p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not torch.cuda.is_available() or not a.device.startswith('cuda'):p.error('CUDA required')
    if a.steps<1 or a.microbatch<1 or 8%a.microbatch:p.error('Positive steps; microbatch must divide 8')
    if not a.resume and a.output.exists():p.error('Output must be new unless --resume')
    torch.cuda.set_device(a.device);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    torch.manual_seed(a.seed)
    data,model,hashes,options=build(a.source.resolve(),a.checkpoint)
    model.idm.encoder.requires_grad_(a.arm=='joint')
    # Zero dropout in this checkpoint; train mode keeps both arms on the same
    # attention path even when the control encoder has no trainable parameters.
    assert all(m.p==0 for m in model.modules() if isinstance(m,nn.Dropout))
    model.to(a.device).train()
    parts={'encoder':model.idm.encoder,'head':model.idm.head,'decoder':model.decoder}
    initial_hashes={k:state_hash(v.state_dict()) for k,v in parts.items()}
    groups=[]
    for name,part in parts.items():
        lr=getattr(a,name+'_lr')
        for decay in (True,False):
            ps=[v for v in part.parameters() if v.requires_grad and (v.ndim>=2)==decay]
            if ps:groups.append({'params':ps,'lr':lr,'weight_decay':.01 if decay else 0.,'name':name})
    optimizer=torch.optim.AdamW(groups)
    generator=torch.Generator().manual_seed(a.seed+1000)
    batches=torch.stack([torch.randint(512,(8,),generator=generator) for _ in range(a.steps)])
    for path in (Path(__file__),ROOT/'research/idm/difference_idm.py',
                 ROOT/'vendor/lawam/latent_action_model/core/utils/lam_encoder.py',
                 ROOT/'vendor/lawam/latent_action_model/core/utils/lam_decoder.py',
                 ROOT/'vendor/lawam/latent_action_model/core/utils/modules.py'):
        hashes[str(path)]=file_hash(path)
    protocol={'arm':a.arm,'seed':a.seed,'steps':a.steps,'batch_size':8,'microbatch':a.microbatch,
        'lr':{n:getattr(a,n+'_lr') for n in parts},'weight_decay_matrix':.01,'weight_decay_bias_norm':0.,
        'gradient_clip':1.,'objective':'MSE of deterministic mean prediction; no posterior sampling, KL or state loss',
        'precision':'float32 highest; no autocast','device':torch.cuda.get_device_name(),'torch':str(torch.__version__),
        'initial_hashes':initial_hashes,'batch_hash':state_hash({'batches':batches}),
        'trainable_parameters':{k:sum(p.numel() for p in v.parameters() if p.requires_grad) for k,v in parts.items()},
        'source_sha256':hashes,'options':options,'scale':float(model.idm.scale),
        'scope':'New difference IDM, no external correction or zero-code prior. Shared Q encoder10000, pretrained mean-head42, original decoder10000. DINO fixed. True-future posterior reconstruction on reused512/64/96. Seeds change minibatches, not pretrained weights. Frozen control also updates head and full decoder.'}
    start_step=0;history=[]
    if a.resume:
        old=json.loads((a.output/'protocol.json').read_text())
        if old!=protocol:raise ValueError('Resume protocol differs')
        saved=torch.load(a.output/'latest.pt',weights_only=True,map_location='cpu')
        model.load_state_dict(saved['model'],strict=True);optimizer.load_state_dict(saved['optimizer'])
        start_step=saved['step'];history=saved['history']
    else:
        a.output.mkdir(parents=True)
        write(a.output/'protocol.json',protocol)
        (a.output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
        with torch.no_grad():
            x=data['train']['current'][:2].to(a.device);y=data['train']['future'][:2].to(a.device)
            z=model.idm(x,y)
            delta=float((z-data['train']['codes'][:2].to(a.device)).abs().max())
            assert delta<1e-3,delta
            same,q,r=model.idm(x,x,return_queries=True)
            assert torch.equal(q,r)
            initial={'initial_code_vs_cached_max_abs':delta,'equal_pair_difference_exact_zero':True,
                'equal_pair_code_norm':float(same.norm()),'initial_two_train_mse':float((model.predict(x,z)-y).square().mean())}
            write(a.output/'preflight.json',initial)
    train={k:data['train'][k].to(a.device) for k in ('current','future')}
    torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();started=time.perf_counter()
    for step in range(start_step+1,a.steps+1):
        model.train();optimizer.zero_grad(set_to_none=True);loss_value=0.
        for micro,idx in enumerate(batches[step-1].split(a.microbatch)):
            x,y=train['current'][idx],train['future'][idx]
            z,q,r=model.idm(x,y,return_queries=True)
            if step==1 and micro==0 and a.arm=='joint':q.retain_grad();r.retain_grad()
            prediction=model.predict(x,z)
            loss=(prediction-y).square().mean()*(len(idx)/8)
            if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
            loss.backward();loss_value+=float(loss.detach())
            if step==1 and micro==0 and a.arm=='joint':
                assert q.grad is not None and r.grad is not None
                assert q.grad.norm()>0 and r.grad.norm()>0
                torch.testing.assert_close(q.grad,-r.grad,rtol=0,atol=0)
                layers=[norm(layer.parameters()) for layer in model.idm.encoder.QFormer.layers]
                assert all(v>0 for v in layers)
                write(a.output/'gradient_check.json',{'transition_query_norm':float(q.grad.norm()),
                    'reference_query_norm':float(r.grad.norm()),'opposite_branch_gradients_exact':True,
                    'encoder_layer_norms':layers,'component_norms':{k:norm(v.parameters()) for k,v in parts.items()}})
        total_norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        optimizer.step()
        if step<=5 or step%20==0 or step==a.steps:
            torch.cuda.synchronize()
            history.append({'step':step,'mse':loss_value,'gradient_norm':float(total_norm),
                'elapsed_seconds':time.perf_counter()-started,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
            write(a.output/'progress.json',history)
            print(a.arm,a.seed,history[-1],flush=True)
        if not a.smoke and (step%100==0 or step==a.steps):
            atomic_save(a.output/'latest.pt',{'model':model.state_dict(),'optimizer':optimizer.state_dict(),
                'step':step,'history':history})
    torch.cuda.synchronize();training_seconds=time.perf_counter()-started
    final_hashes={k:state_hash(v.state_dict()) for k,v in parts.items()}
    changed={k:final_hashes[k]!=initial_hashes[k] for k in parts}
    assert changed['head'] and changed['decoder']
    assert changed['encoder']==(a.arm=='joint')
    if a.smoke:
        write(a.output/'results.json',{'complete':True,'smoke_only':True,'changed':changed,'history':history,
            'training_seconds':training_seconds,'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
        return
    # Evaluation starts only after the fixed update budget, with no best-step selection.
    evaluation=evaluate(model,data,a.device,a.microbatch)
    atomic_save(a.output/'model.pt',{'model':model.state_dict(),'protocol':protocol,'step':a.steps})
    write(a.output/'results.json',{'complete':True,'smoke_only':False,'evaluation':evaluation,
        'changed':changed,'initial_hashes':initial_hashes,'final_hashes':final_hashes,'history':history,
        'training_seconds':training_seconds,'checkpoint_sha256':file_hash(a.output/'model.pt'),
        'peak_allocated_bytes':torch.cuda.max_memory_allocated()})
    print('DONE',a.arm,a.seed,flush=True)


if __name__=='__main__':main()
