"""Fixed protocol early-conditioning pilot. No optimization unless --train is supplied."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'vendor/lawam')]
import torch
import yaml
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from research.early_condition_adapter import install_adapter, frozen_hash, state_hash
from research.layerwise_condition import install_layerwise_adapter


def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()


def save_json(path, data):
    path.write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def load_inputs(source):
    roots={'train': source/'runs/reference-idm42-nested512-prior10000',
           'validation': source/'runs/reference-idm42-nested512-prior10000',
           'observed96': source/'runs/reference-idm42-observed96-prior10000'}
    data={}; hashes={}; manifests={}
    for split,root in roots.items():
        mp=root/'cache_manifest.json'; manifest=json.loads(mp.read_text())
        disk='validation' if split=='observed96' else split
        path=root/(disk+'.pt'); digest=file_hash(path)
        assert digest==manifest['cache_sha256'][disk],path
        d=torch.load(path,weights_only=True,map_location='cpu')
        assert len(d['rows'])=={'train':512,'validation':64,'observed96':96}[split]
        for key in ['current','future','codes','baseline']:
            assert len(d[key])==len(d['rows']) and torch.isfinite(d[key]).all(),(split,key)
        assert d['current'].shape==d['future'].shape==d['baseline'].shape
        assert d['codes'].shape[-1]==32
        assert manifest['prior_checkpoint_step']==10000
        data[split]=d; manifests[split]=manifest
        hashes[str(path)]=digest;hashes[str(mp)]=file_hash(mp)
    episodes={s:{str(r['episode']) for r in d['rows']} for s,d in data.items()}
    for a,b in [('train','validation'),('train','observed96'),('validation','observed96')]:
        assert not episodes[a]&episodes[b],(a,b)
    weights=source/'runs/frozen-model-step10000/reference_weights.pt'
    wh=file_hash(weights)
    for m in manifests.values(): assert m['prior_weights_sha256']==wh
    hashes[str(weights)]=wh
    configpath=weights.parent/'resolved_config.yaml';hashes[str(configpath)]=file_hash(configpath)
    config=yaml.safe_load(configpath.read_text())['model']
    options=dict(context_dim=config['dim'], input_dim=data['train']['current'].shape[-1],
                 code_dim=config['code_dim'],num_queries=config['num_queries'],
                 num_layers=config['dec_layers'],num_heads=config['num_heads'],dropout=config['dropout'],
                 grid_hw=tuple(v//config['patch_size'] for v in config['image_hw']),
                 ffn_expansion_factor=config['ffn_expansion_factor'],last_ln=config.get('decoder_last_ln',True))
    model=LAMDecoder_v2(**options).eval()
    saved=torch.load(weights,weights_only=True,mmap=True,map_location='cpu')
    assert saved['global_step']==10000
    prefix='lam.decoder.'
    model.load_state_dict({k[len(prefix):]:v for k,v in saved['state_dict'].items() if k.startswith(prefix)},strict=True)
    return data,model,hashes,options


@torch.no_grad()
def evaluate(model,data,constant):
    out={}
    for split,d in data.items():
        rows=[]; baseline_difference=0.
        for start in range(0,len(d['rows']),8):
            device=next(model.parameters()).device
            x,y,z=[d[k][start:start+8].to(device) for k in ['current','future','codes']]
            if constant is not None: z=constant.expand_as(z)
            pred=model(x,z); zero=model(x,torch.zeros_like(z))
            mse=(pred-y).square().flatten(1).mean(1)
            prior=(zero-y).square().flatten(1).mean(1)
            baseline_difference=max(baseline_difference,float((zero-d['baseline'][start:start+len(x)].to(device)).abs().max()))
            for i in range(len(x)):
                rows.append({'source_row':d['rows'][start+i],'mse':float(mse[i]),'zero_mse':float(prior[i])})
        out[split]={'mse':sum(r['mse'] for r in rows)/len(rows),
                    'zero_mse':sum(r['zero_mse'] for r in rows)/len(rows),
                    'zero_vs_cached_baseline_max_abs':baseline_difference,'rows':rows}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['real','constant','rms_constant'],required=True)
    p.add_argument('--architecture', choices=['shared','layerwise'], default='shared')
    p.add_argument('--device', choices=['cpu','cuda'], default='cpu')
    p.add_argument('--threads',type=int,default=4);p.add_argument('--train',action='store_true')
    a=p.parse_args()
    if a.output.exists(): p.error('Output must be new')
    if a.threads<1: p.error('threads must be positive')
    torch.set_num_threads(a.threads);torch.set_float32_matmul_precision('highest');torch.manual_seed(42)
    data,model,hashes,options=load_inputs(a.source.resolve())
    device=torch.device(a.device)
    model=model.to(device)
    for path in [Path(__file__),ROOT/'research/early_condition_adapter.py',ROOT/'research/layerwise_condition.py',ROOT/'vendor/lawam/latent_action_model/core/utils/lam_decoder.py']:
        hashes[str(path)]=file_hash(path)
    d=data['train']; x=d['current'][:8].to(device); z=d['codes'][:8].to(device)
    with torch.no_grad(): original=model(x,torch.zeros_like(z)).clone()
    model=(install_layerwise_adapter(model) if a.architecture=='layerwise' else install_adapter(model)); frozen=frozen_hash(model)
    with torch.no_grad():
        initial=model(x,z);initialzero=model(x,torch.zeros_like(z))
    assert torch.equal(initial,original) and torch.equal(initialzero,original),'Initial prior mismatch'
    constant=d['codes'].mean(0,keepdim=True).to(device)
    assert constant.square().sum()>0
    original_mean_norm=float(constant.norm())
    train_code_rms=float(d['codes'].square().sum(-1).mean().sqrt())
    if a.arm=='rms_constant': constant=constant*(train_code_rms/original_mean_norm)
    g=torch.Generator().manual_seed(1042)
    batches=torch.stack([torch.randint(len(d['rows']),(8,),generator=g) for _ in range(400)])
    a.output.mkdir(parents=True)
    protocol={'device':str(device),'architecture':a.architecture,'arm':a.arm,'seed':42,'minibatch_seed':1042,'execute_training':a.train,'steps':400,'batch_size':8,'lr':3e-4,'weight_decay':.01,
              'gradient_clip':1.,'initial_backend_tolerance':1e-4,'precision':'highest','threads':a.threads,'source_hashes':hashes,
              'initial_state_hash':state_hash(model.state_dict()),'frozen_hash':frozen,
              'train_mean_norm':original_mean_norm,'train_code_rms_norm':train_code_rms,'used_constant_norm':float(constant.norm()),
              'batch_hash':state_hash({'batches':batches}),'constant_hash':state_hash({'constant':constant}),
              'options':options,'scope':'Posterior reconstruction on already observed evaluation sets; frozen encoder and all original decoder weights.'}
    save_json(a.output/'protocol.json',protocol)
    save_json(a.output/'preflight.json',{'initial_exact':True,'initial_zero_exact':True,
        'trainable_parameters':sum(v.numel() for v in model.parameters() if v.requires_grad),
        'source_split_counts':{s:len(v['rows']) for s,v in data.items()},
        'initial_diagnostics':model.action_in_proj.diagnostics(d['codes'].to(device))})
    if not a.train: return
    opt=torch.optim.AdamW([model.action_in_proj.U.weight],lr=3e-4,weight_decay=.01)
    logs=[];t=time.monotonic()
    for step,idx in enumerate(batches,1):
        opt.zero_grad(set_to_none=True)
        code=d['codes'][idx].to(device) if a.arm=='real' else constant.expand_as(d['codes'][idx])
        prediction=model(d['current'][idx].to(device),code)
        if step==1:
            with torch.no_grad():
                inference_prior=model(d['current'][idx].to(device),torch.zeros_like(code))
            backend_delta=float((prediction.detach()-inference_prior).abs().max())
            cache_delta=float((prediction.detach()-d['baseline'][idx].to(device)).abs().max())
            save_json(a.output/'initial_backend.json',{'train_vs_inference_max_abs':backend_delta,
                'train_vs_cached_baseline_max_abs':cache_delta,'train_vs_inference_tolerance':1e-4,
                'cache_note':'Cached baseline used B2; diagnostic only, not bitwise equivalence.'})
            if backend_delta>1e-4: raise RuntimeError('Initial autograd/backend discrepancy exceeds declared tolerance')
        loss=(prediction-d['future'][idx].to(device)).square().mean()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        loss.backward()
        norm=torch.nn.utils.clip_grad_norm_([model.action_in_proj.U.weight],1.,error_if_nonfinite=True)
        if step==1 and norm==0: raise RuntimeError('Initial condition gradient is zero')
        opt.step()
        if step in [1,10] or step%20==0:
            logs.append({'step':step,'mse':float(loss.detach()),'preclip_gradient_norm':float(norm),'elapsed_seconds':time.monotonic()-t})
            print(json.dumps(logs[-1]),flush=True)
        if step%100==0:
            temporary=a.output/'latest.tmp'
            torch.save({'U':model.action_in_proj.U.state_dict(),'optimizer':opt.state_dict(),
                        'step':step,'protocol':protocol},temporary)
            temporary.replace(a.output/'latest.pt')
    assert frozen_hash(model)==frozen,'Frozen state changed'
    with torch.no_grad(): assert torch.equal(model(x,torch.zeros_like(z)),original),'Zero-code invariant failed'
    metrics=evaluate(model,data,constant if a.arm!='real' else None)
    assert all(v['zero_vs_cached_baseline_max_abs']<=1e-4 for v in metrics.values()),'Zero-code/cache discrepancy exceeds declared tolerance'
    diagnostics={s:model.action_in_proj.diagnostics(v['codes'].to(device) if a.arm=='real' else constant.expand_as(v['codes'])) for s,v in data.items()}
    ck=a.output/'adapter.pt'
    torch.save({'U':model.action_in_proj.U.state_dict(),'constant':constant,'protocol':protocol,
                'step':400,'frozen_hash':frozen},ck)
    save_json(a.output/'results.json',{'evaluation':metrics,'condition_diagnostics':diagnostics,
        'logs':logs,'zero_exact':True,'frozen_hash_unchanged':True,'seconds':time.monotonic()-t,
        'checkpoint_sha256':file_hash(ck),'protocol_sha256':file_hash(a.output/'protocol.json')})


if __name__=='__main__': main()
