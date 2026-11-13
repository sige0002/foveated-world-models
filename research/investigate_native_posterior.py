"""Frozen local/released LaWAM posterior, conditioning and noise diagnostics.

Same-current future replacement is a synthetic dependence probe, not a valid
alternative-action ground truth. No training or checkpoint selection occurs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'vendor/lawam')]
import torch
import yaml
from latent_action_model.core.utils.lam_encoder import LAMEncoder
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from latent_action_model.core.vq import VAEQuantizer
from research.train_early_condition import file_hash
from research.z_utility_matched import SOURCE, CHECKPOINT

RELEASE = Path('/data2/yuki_sadasue/models/lawam-lam-bd993da/checkpoints/pytorch_model.pt')
RELEASE_SHA = '57c31dcba13c2998c4e866cdcfc427f14ae3c1b762a586778276e2100b48af88'


def geometry(t):
    x=t.double().flatten(1); c=x-x.mean(0)
    variation=float(c.square().sum()/len(x)); common=float(x.mean(0).square().sum())
    gram=c@c.T/len(x); squares=float(gram.square().sum())
    return dict(variance_sum=variation, common_mean_energy=common,
        variance_fraction=variation/(common+variation) if common+variation else 0.,
        effective_rank=variation**2/squares if squares else 0.)


def gaussian_kl(m,l,om,ol):
    m,l,om,ol=[t.double() for t in (m,l,om,ol)]
    d=l-ol
    return .5*(d.expm1()-d+(m-om).square()*(-ol).exp()).sum(-1)


def posterior_stats(mu,lv,beta):
    m,l=mu.double().flatten(1),lv.double().flatten(1)
    var=l.exp(); c=m-m.mean(0)
    cov=c.T@c/len(m)+torch.diag(var.mean(0))
    sign,logdet=torch.linalg.slogdet(cov); assert sign>0
    # Gaussian maximizes entropy at fixed covariance; empirical mixture over rows.
    upper=.5*(logdet-l.sum(1).mean())
    kl=.5*(m.square()+var-1-l).sum(1)
    signal=float(c.square().sum()/len(m));noise=float(var.sum(1).mean())
    return dict(mu=geometry(mu),logvar=geometry(lv),
        mean_sigma_squared=float(var.mean()), noise_variance_sum=noise,
        signal_to_noise_trace_ratio=signal/noise,
        noise_fraction_of_sampled_z_variance=noise/(noise+signal),
        kl_to_unit_gaussian_mean=float(kl.mean()),beta_weighted_kl=float(kl.mean()*beta),
        empirical_sample_identity_information_upper_bound_nats=float(upper),
        information_bound_scope='One draw Z from empirical diagonal-Gaussian posterior mixture over these rows; does not bound deterministic mu or unseen population.')


def load_data():
    data={};hashes={}
    folder=SOURCE/'runs/reference-idm42-nested512-prior10000'
    manifest=json.loads((folder/'cache_manifest.json').read_text())
    for split in ('train','validation'):
        p=folder/f'{split}.pt';h=file_hash(p);assert h==manifest['cache_sha256'][split]
        d=torch.load(p,weights_only=True,map_location='cpu',mmap=True)
        data[split]={k:d[k] for k in ('current','future','rows','native_codes')}
        hashes[str(p)]=h
    p=ROOT/'runs/difference-idm-ablation-inputs-20261113/test.pt'
    manifest=json.loads(p.with_suffix('.json').read_text());h=file_hash(p)
    assert h==manifest['cache_sha256'];assert file_hash(CHECKPOINT)==manifest['checkpoint_sha256']
    data['reused_test170']=torch.load(p,weights_only=True,map_location='cpu',mmap=True)
    hashes[str(p)]=h
    return data,hashes


def make_model(label,device):
    cfgpath=(ROOT/'research/results/official-implementation-comparison-20261113/official-release-config.yaml'
             if label=='official' else CHECKPOINT.parents[1]/'resolved_config.yaml')
    cfg=yaml.safe_load(cfgpath.read_text())['model']
    path=RELEASE if label=='official' else CHECKPOINT
    digest=file_hash(path)
    if label=='official':assert digest==RELEASE_SHA
    saved=torch.load(path,weights_only=True,mmap=True,map_location='cpu')['state_dict']
    assert cfg['enc_add_state'] is False and cfg['latent_layer_to_use']==-2
    assert cfg['norm_latents'] and cfg['norm_latents_type']=='ln'
    backbone_check=None
    if label=='official':
        local=torch.load(CHECKPOINT,weights_only=True,mmap=True,map_location='cpu')['state_dict']
        keys=[k for k in local if k.startswith('lam.vision_encoder.')]
        official_keys=[k for k in saved if k.startswith('lam.vision_encoder.')]
        assert len(keys)==len(official_keys)==211
        for k in keys:
            # Transformers version renamed the block container; tensors must be exact.
            other=k.replace('lam.vision_encoder.model.model.layer.','lam.vision_encoder.model.layer.')
            assert torch.equal(local[k],saved[other]), k
        backbone_check=dict(tensors_checked=len(keys),all_exact=True,
            rename='lam.vision_encoder.model.model.layer.* -> lam.vision_encoder.model.layer.*')
        del local
    options=dict(context_dim=cfg['dim'],input_dim=768,code_dim=cfg['code_dim'],
        num_queries=cfg['num_queries'],num_heads=cfg['num_heads'],dropout=cfg['dropout'],
        ffn_expansion_factor=cfg['ffn_expansion_factor'],
        grid_hw=tuple(v//cfg['patch_size'] for v in cfg['image_hw']))
    modules=torch.nn.ModuleDict(dict(
        encoder=LAMEncoder(**options,num_layers=cfg['enc_layers'],num_frames=cfg['num_frames'],
            modal_mask=cfg['enc_modal_mask'],add_state=False,max_state_dim=cfg['max_state_dim'],
            num_embodiments=cfg['num_embodiments']),
        vq=VAEQuantizer(code_dim=cfg['code_dim'],**cfg['vq_kwargs']),
        decoder=LAMDecoder_v2(**options,num_layers=cfg['dec_layers'],last_ln=cfg.get('decoder_last_ln',True))))
    loaded={}
    for name,module in modules.items():
        prefix='lam.'+name+'.';state={k[len(prefix):]:v for k,v in saved.items() if k.startswith(prefix)}
        module.load_state_dict(state,strict=True);loaded[name]=len(state)
    return modules.eval().requires_grad_(False).to(device),dict(checkpoint=str(path),sha256=digest,
        config_sha256=file_hash(cfgpath),strict_loaded_tensor_counts=loaded,
        backbone_vs_local=backbone_check)


@torch.no_grad()
def encode(modules,x,y,device):
    stages={k:[] for k in ('query','nodes','pre_norm','mu','logvar','decoder_projection')}
    query=[]
    hook=modules['encoder'].out_proj.register_forward_pre_hook(lambda m,args:query.append(args[0].detach().cpu()))
    for start in range(0,len(x),8):
        nodes=modules['encoder'](torch.cat((x[start:start+8],y[start:start+8]),1).to(device))
        mu,lv=modules['vq']._encode(nodes)
        vals=dict(nodes=nodes,pre_norm=modules['vq'].pre_norm(nodes.mean(1,keepdim=True)),
            mu=mu,logvar=lv,decoder_projection=modules['decoder'].action_in_proj(mu))
        for key,value in vals.items():stages[key].append(value.cpu())
    hook.remove();stages['query']=query
    return {k:torch.cat(v) for k,v in stages.items()}


@torch.no_grad()
def evaluate(modules,d,s,train,device,samples):
    x,y=d['current'],d['future'];mu,lv=s['mu'],s['logvar']
    mean=train['mu'].mean(0,keepdim=True)
    # Match average conditional noise power when fixing sigma across examples.
    std_const=train['logvar'].exp().mean(0,keepdim=True).sqrt()
    alt=encode(modules,x,y.roll(1,0),device)
    null=encode(modules,x,x,device)
    sym=.5*(gaussian_kl(mu,lv,alt['mu'],alt['logvar'])+gaussian_kl(alt['mu'],alt['logvar'],mu,lv))
    dec=modules['decoder'];rows=[];gen=torch.Generator(device=device).manual_seed(20261113)
    for start in range(0,len(x),8):
        sl=slice(start,start+8);current,target=x[sl].to(device),y[sl].to(device)
        m,l=mu[sl].to(device),lv[sl].to(device);sd=(l*.5).exp()
        pred=dec(current,m)
        vals=dict(mse=(pred-target).square().flatten(1).mean(1),copy_mse=(current-target).square().flatten(1).mean(1))
        codes=dict(train_mean=mean.expand_as(mu[sl]).to(device),zero=torch.zeros_like(m),
            swapped=mu.roll(1,0)[sl].to(device),same_current_other_future=alt['mu'][sl].to(device),
            same_current_no_change=null['mu'][sl].to(device))
        for name,z in codes.items():
            p=dec(current,z);vals[name+'_mse']=(p-target).square().flatten(1).mean(1)
            vals[name+'_output_delta_mse']=(p-pred).square().flatten(1).mean(1)
        for draw in range(samples):
            eps=torch.randn(m.shape,device=device,generator=gen)
            sampled=dec(current,m+sd*eps)
            variants=dict(sample_true=sampled,
                sample_fixed_mu=dec(current,mean.to(device)+sd*eps),
                sample_fixed_sigma=dec(current,m+std_const.to(device)*eps),
                sample_fixed_posterior=dec(current,mean.to(device)+std_const.to(device)*eps))
            for name,p in variants.items():
                for suffix,value in [('mse',(p-target).square().flatten(1).mean(1)),
                    ('output_delta_vs_sample_true_mse',(p-sampled).square().flatten(1).mean(1))]:
                    key=name+'_'+suffix;vals[key]=vals.get(key,0)+value/samples
            key='sample_noise_output_delta_vs_mu_mse'
            vals[key]=vals.get(key,0)+(sampled-pred).square().flatten(1).mean(1)/samples
        for j in range(len(current)):
            rows.append(dict(source_row=d['rows'][start+j],
                same_current_other_future_symmetric_kl=float(sym[start+j]),
                **{key:float(value[j]) for key,value in vals.items()}))
        if start%64==0:print('decoded',start+len(current),'/',len(x),flush=True)
    keys=[k for k in rows[0] if k!='source_row']
    return dict(metrics={k:sum(r[k] for r in rows)/len(rows) for k in keys},rows=rows,
        same_current_replaced_future_stages={k:geometry(v) for k,v in alt.items()})


def gradient_probe(modules,d,device,samples=4):
    # Frozen decoder, no updates: compare reconstruction vs KL gradients at nodes.
    encoder,vq,decoder=[modules[k] for k in ('encoder','vq','decoder')]
    x,y=[d[k][:8].to(device) for k in ('current','future')]
    with torch.no_grad():base=encoder(torch.cat((x,y),1))
    gen=torch.Generator(device=device).manual_seed(1907);rows=[]
    for _ in range(samples):
        nodes=base.detach().requires_grad_(True);mu,lv=vq._encode(nodes)
        eps=torch.randn(mu.shape,device=device,generator=gen)
        pred=decoder(x,mu+(lv*.5).exp()*eps)
        recon=torch.nn.functional.smooth_l1_loss(pred,y,beta=.1)
        kl=vq._kl_divergence(mu,lv).mean()*vq.beta
        gr=torch.autograd.grad(recon,nodes,retain_graph=True)[0]
        gk=torch.autograd.grad(kl,nodes)[0]
        nr,nk=float(gr.norm()),float(gk.norm())
        rows.append(dict(reconstruction_loss=float(recon.detach()),weighted_kl=float(kl.detach()),
            reconstruction_gradient_norm=nr,kl_gradient_norm=nk,kl_over_reconstruction_gradient_norm=nk/nr,
            gradient_cosine=float(torch.nn.functional.cosine_similarity(gr.flatten(),gk.flatten(),dim=0))))
    return dict(rows=rows,scope='First 8 validation pairs, frozen checkpoint, no augmentation, sampled posterior, gradients wrt 32D nodes. Excludes physical auxiliary loss; not a causal training ablation.')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--label',choices=['local','official'],required=True)
    p.add_argument('--device',default='cuda:1');p.add_argument('--samples',type=int,default=4)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    assert a.samples>0
    torch.set_num_threads(4);torch.cuda.set_device(a.device);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32=False;torch.manual_seed(42)
    data,hashes=load_data();modules,meta=make_model(a.label,a.device)
    report=dict(label=a.label,model=meta,cache_sha256=hashes,script_sha256=file_hash(Path(__file__)),
        samples=a.samples,splits={},precision='FP32 highest; frozen eval mode; no augmentation',
        scope='Reused RoboMIND evaluation pairs, local padded-image feature preprocessing; official DINO weights must equal local before sharing caches. No claim about official training population or valid alternate-action futures.')
    states={}
    for split,d in data.items():
        states[split]=s=encode(modules,d['current'],d['future'],a.device)
        if a.label=='local' and 'native_codes' in d:
            error=float((s['mu']-d['native_codes']).abs().max());assert error<1e-3,error
        else:error=None
        stages={k:geometry(v) for k,v in s.items()}
        with torch.no_grad():
            for index in (0,len(modules['decoder'].dec_layers)-1):
                v=modules['decoder'].dec_layers[index].adaLN_modulation(s['decoder_projection'].to(a.device))
                stages[f'adaLN_modulation_layer_{index}']=geometry(v.cpu())
        report['splits'][split]=dict(count=len(d['rows']),stages=stages,
            posterior=posterior_stats(s['mu'],s['logvar'],modules['vq'].beta),native_cache_max_abs=error)
        print(a.label,'encoded',split,report['splits'][split]['posterior'],flush=True)
    for split in ('validation','reused_test170'):
        report['splits'][split]['evaluation']=evaluate(modules,data[split],states[split],states['train'],a.device,a.samples)
        print(a.label,split,report['splits'][split]['evaluation']['metrics'],flush=True)
    report['gradient_probe']=gradient_probe(modules,data['validation'],a.device)
    report['peak_cuda_bytes']=torch.cuda.max_memory_allocated(a.device)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('COMPLETE',a.label,flush=True)


if __name__=='__main__':main()
