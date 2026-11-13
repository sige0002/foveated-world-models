"""Matched direct/external Stage1 predictors with learned constant controls."""
import argparse
import copy
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import torch
from torch import nn
from research.train_difference_idm import build,write,atomic_save,norm
from research.train_idm_ablation import initialize_ablation,make_batches,save_training,restore_training
from research.train_cached_decoders import make_decoder
from research.train_early_condition import file_hash
from research.early_condition_adapter import state_hash

SOURCE=ROOT/'runs/layerwise-gpu-inputs-20261113'
CHECKPOINT=Path('/data2/yuki_sadasue/runs/foveated-world-models/stage1-robomind-main/checkpoints/final.ckpt')
OLD=ROOT/'runs/difference-idm-ablation-20261113'
ARMS=('direct-real','direct-constant','external-real','external-constant')


class MatchedModel(nn.Module):
    def __init__(self,idm,decoder,constant=False,prior=None):
        super().__init__();self.idm=idm;self.decoder=decoder;self.constant=constant;self.prior=prior
        if constant:self.idm.encoder.requires_grad_(False)
        if prior is not None:prior.requires_grad_(False).eval()

    def code(self,x,y):
        if self.constant:return self.null_code(x)
        return self.idm(x,y)

    def null_code(self,x):
        return self.idm.head(x.new_zeros((len(x),1,self.idm.encoder.context_dim)))

    def predict(self,x,z,baseline=None):
        if self.prior is None:return self.decoder(x,z)
        if baseline is None:
            with torch.no_grad():baseline=self.prior(x,torch.zeros_like(z))
        return self.decoder(x,z,baseline=baseline)

    def forward(self,x,y,baseline=None):
        z=self.code(x,y);return self.predict(x,z,baseline),z


def make_model(arm,seed):
    data,base,hashes,options=build(SOURCE.resolve(),CHECKPOINT)
    scale=initialize_ablation(base,data['train'],'difference',seed)
    initial={k:state_hash(v.state_dict()) for k,v in {'encoder':base.idm.encoder,'head':base.idm.head,'decoder':base.decoder}.items()}
    previous=json.loads((OLD/f'difference-{seed}'/'protocol.json').read_text())
    assert initial==previous['initial_hashes']
    assert scale==previous['scale'] and previous['steps']==1600
    assert json.loads(json.dumps(options))==previous['options']
    for path,digest in hashes.items():assert previous['source_sha256'][path]==digest,path
    assert previous['batch_hash']==state_hash({'batches':make_batches(seed,1600,512)})
    assert previous['lr']=={'encoder':1e-5,'head':1e-4,'decoder':3e-5}
    assert previous['microbatch']==previous['batch_size']==8
    prior=None;decoder=base.decoder
    if arm.startswith('external'):
        prior=decoder
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed+2000)
            decoder=make_decoder('prior_residual',128,2,768,32,16)
    model=MatchedModel(base.idm,decoder,arm.endswith('constant'),prior)
    meta={'arm':arm,'seed':seed,'steps':400,'batch_size':8,'scale':scale,'options':options,
          'initial_backbone_hashes':initial,'initial_predictor_hash':state_hash(decoder.state_dict()),
          'source_sha256':hashes,'prior_sha256':state_hash(prior.state_dict()) if prior is not None else None,
          'batch_hash':state_hash({'batches':make_batches(seed,400,512)}),
          'lr':{'encoder':1e-5,'head':1e-4,'decoder':3e-5},'precision':'float32 highest; no autocast; cudnn TF32 disabled',
          'trainable_parameters':{k:sum(p.numel() for p in v.parameters() if p.requires_grad) for k,v in
                                  {'encoder':model.idm.encoder,'head':model.idm.head,'decoder':model.decoder}.items()},
          'plan_sha256':file_hash(ROOT/'docs/z-utility-matched-plan.md'),
          'script_sha256':file_hash(Path(__file__)),
          'scope':'Fixed400; same steps/data/initial head but not same FLOPs/capacity. Reused evaluation sets.'}
    for f in ['research/prior_correction.py','research/wide_head_flow.py','research/train_cached_decoders.py','research/idm/difference_idm.py']:
        meta['source_sha256'][str(ROOT/f)]=file_hash(ROOT/f)
    return data,model,meta


def optimizer(model):
    groups=[]
    for name,module in [('encoder',model.idm.encoder),('head',model.idm.head),('decoder',model.decoder)]:
        for decay in [True,False]:
            ps=[p for p in module.parameters() if p.requires_grad and (p.ndim>=2)==decay]
            if ps:groups.append({'params':ps,'lr':{'encoder':1e-5,'head':1e-4,'decoder':3e-5}[name],
                                 'weight_decay':.01 if decay else 0.})
    return torch.optim.AdamW(groups)


def compact_state(model):
    # The large frozen encoder/prior stay in the immutable shared source.
    return {k:v for k,v in model.state_dict().items() if not k.startswith('prior.') and
            not (model.constant and k.startswith('idm.encoder.'))}


def load_state(model,state):
    missing,unexpected=model.load_state_dict(state,strict=False)
    assert not unexpected
    assert all(k.startswith('prior.') or (model.constant and k.startswith('idm.encoder.')) for k in missing)


def train(a):
    torch.cuda.set_device(a.device);torch.set_num_threads(4);torch.manual_seed(a.seed)
    torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    data,model,meta=make_model(a.arm,a.seed)
    meta['steps']=a.steps;meta['smoke_only']=a.smoke
    meta['batch_hash']=state_hash({'batches':make_batches(a.seed,a.steps,512)})
    meta['device']=torch.cuda.get_device_name();meta['torch']=str(torch.__version__)
    meta['stop_after']=None # Runtime interruption does not change the fixed protocol.
    model.to(a.device).train();opt=optimizer(model)
    parts={'encoder':model.idm.encoder,'head':model.idm.head,'decoder':model.decoder}
    before={k:state_hash(v.state_dict()) for k,v in parts.items()}
    if a.output.exists() and not a.resume:raise FileExistsError(a.output)
    a.output.mkdir(parents=True,exist_ok=True)
    start=0;history=[];seconds=0.
    if a.resume:
        saved=torch.load(a.output/'latest.pt',weights_only=True,map_location='cpu')
        assert saved['protocol']==meta
        load_state(model,saved['model']);opt.load_state_dict(saved['optimizer'])
        start=saved['step'];history=saved['history'];seconds=saved['training_seconds']
        torch.set_rng_state(saved['cpu_rng']);torch.cuda.set_rng_state(saved['cuda_rng'])
        del saved
    else:
        write(a.output/'protocol.json',meta)
        (a.output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    train=data['train'];x,y=[train[k].to(a.device) for k in ['current','future']]
    b=train['baseline'].to(a.device) if model.prior is not None else None
    if model.prior is not None:
        with torch.no_grad():
            actual=model.prior(x[:8],x.new_zeros((8,1,32)))
            err=float((actual-b[:8]).abs().max());assert err<1e-4,err
            if not start:
                pred,_=model(x[:8],y[:8],b[:8]);torch.testing.assert_close(pred,b[:8],rtol=0,atol=0)
            write(a.output/'prior-check.json',{'cached_baseline_max_abs':err})
    if model.constant:
        with torch.no_grad():torch.testing.assert_close(model.code(x[:8],y[:8]),model.code(x[:8],y[:8].roll(1,0)),rtol=0,atol=0)
    batches=make_batches(a.seed,a.steps,512);torch.cuda.reset_peak_memory_stats()
    gradients=[]
    for step in range(start+1,a.steps+1):
        idx=batches[step-1];opt.zero_grad(set_to_none=True);torch.cuda.synchronize();tick=time.perf_counter()
        pred,z=model(x[idx],y[idx],b[idx] if b is not None else None)
        loss=(pred-y[idx]).square().mean();assert torch.isfinite(loss)
        loss.backward()
        if step<=4:
            g={k:norm(v.parameters()) for k,v in parts.items()};g['step']=step;gradients.append(g)
            if model.constant:assert g['encoder']==0
            # Zero output first learns a readout; zero AdaLN gates then open.
            elif step>=3:assert g['encoder']>0 and g['head']>0,g
            assert g['decoder']>0
        gn=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step();torch.cuda.synchronize();seconds+=time.perf_counter()-tick
        if step<=5 or step%20==0 or step==a.steps or step==a.stop_after:
            row={'step':step,'mse':float(loss.detach()),'gradient_norm':float(gn),'training_seconds':seconds,
                 'peak_allocated_bytes':torch.cuda.max_memory_allocated()}
            history.append(row);write(a.output/'progress.json',history);print(a.arm,a.seed,row,flush=True)
        if step%100==0 or step==a.steps or step==a.stop_after:
            atomic_save(a.output/'latest.pt',{'model':compact_state(model),'optimizer':opt.state_dict(),'protocol':meta,
                'step':step,'history':history,'training_seconds':seconds,'cpu_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state()})
        if step==a.stop_after:return
    after={k:state_hash(v.state_dict()) for k,v in parts.items()}
    changed={k:after[k]!=before[k] for k in before}
    assert changed['decoder'] and changed['head']
    assert changed['encoder']==(not model.constant)
    if model.prior is not None:assert state_hash(model.prior.state_dict())==meta['prior_sha256']
    atomic_save(a.output/'model.pt',{'model':compact_state(model),'protocol':meta,'step':a.steps})
    model.eval()
    with torch.no_grad():expected=model(x[:8],y[:8],b[:8] if b is not None else None)[0]
    saved=torch.load(a.output/'model.pt',weights_only=True,map_location='cpu',mmap=True)
    load_state(model,saved['model'])
    with torch.no_grad():replayed=model(x[:8],y[:8],b[:8] if b is not None else None)[0]
    error=float((expected-replayed).abs().max());assert error==0
    write(a.output/'results.json',{'complete':True,'smoke_only':a.smoke,'changed':changed,'final_hashes':after,
        'gradient_checks':gradients,'replay_max_abs':error,'training_seconds':seconds,'history':history,
        'peak_allocated_bytes':torch.cuda.max_memory_allocated(),'checkpoint_sha256':file_hash(a.output/'model.pt')})
    print('DONE',a.arm,a.seed,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=ARMS,required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--steps',type=int,default=400);p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args()
    if a.steps!=400 and not a.smoke:p.error('Fixed400 outside smoke')
    train(a)


if __name__=='__main__':main()
