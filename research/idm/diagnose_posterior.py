"""Factor the future -> posterior -> decoder path without retraining or DINO reload.

Future/current swaps form synthetic, often OOD pairs. They measure dependence,
not valid counterfactual actions. Common random numbers isolate posterior factors.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vendor/lawam'))
import numpy as np
import torch
import yaml
from latent_action_model.core.utils.lam_encoder import LAMEncoder
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from latent_action_model.core.vq import VAEQuantizer


def gaussian_kl(mu,lv,other_mu,other_lv):
    mu,lv,other_mu,other_lv = [x.double() for x in (mu,lv,other_mu,other_lv)]
    difference = lv-other_lv
    return .5*(difference.expm1()-difference+(mu-other_mu).square()*(-other_lv).exp()).sum(-1)


def summarize_tensor(tensor):
    x=tensor.flatten(1)
    return dict(mean_norm_squared=float(x.mean(0).square().sum()),
                between_pair_total_variance=float(x.var(0,unbiased=False).sum()),
                minimum=float(x.min()),maximum=float(x.max()))


@torch.inference_mode()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--pairs',type=int,default=16)
    parser.add_argument('--samples',type=int,default=2)
    args=parser.parse_args()
    if args.output.exists() or args.pairs<2 or args.samples<1:
        parser.error('New output, at least two pairs, and positive samples required')
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('medium')
    config=yaml.safe_load((args.cache/'resolved_config.yaml').read_text())['model']
    if config['vq_type'] not in ('vae','beta_vae') or config['enc_add_state']:
        parser.error('This diagnostic requires visual-only VAE IDM')
    saved=torch.load(args.cache/'reference_weights.pt',weights_only=True,mmap=True)
    data=torch.load(args.cache/'validation.pt',weights_only=True)
    indices=np.linspace(0,len(data['current'])-1,min(args.pairs,len(data['current'])),dtype=int)
    h,y,z_ref=[data[key][indices] for key in ('current','future','codes')]
    grid=tuple(v//config['patch_size'] for v in config['image_hw'])
    options=dict(context_dim=config['dim'],input_dim=h.shape[-1],num_heads=config['num_heads'],
                 num_queries=config['num_queries'],grid_hw=grid,
                 ffn_expansion_factor=config['ffn_expansion_factor'],dropout=config['dropout'],
                 code_dim=config['code_dim'])
    encoder=LAMEncoder(**options,num_layers=config['enc_layers'],num_frames=config['num_frames'],
                       modal_mask=config['enc_modal_mask'],add_state=False,
                       max_state_dim=config['max_state_dim'],num_embodiments=config['num_embodiments']).eval()
    vq=VAEQuantizer(code_dim=config['code_dim'],**config['vq_kwargs']).eval()
    decoder=LAMDecoder_v2(**options,num_layers=config['dec_layers'],last_ln=config.get('decoder_last_ln',True)).eval()
    for name,module in [('encoder',encoder),('vq',vq),('decoder',decoder)]:
        prefix='lam.'+name+'.'
        module.load_state_dict({k[len(prefix):]:v for k,v in saved['state_dict'].items() if k.startswith(prefix)},strict=True)
    step=saved['global_step'];del saved
    modes={'true':(h,y),'future_swapped':(h,y.roll(1,0)),
           'current_swapped':(h.roll(1,0),y),'no_change':(h,h),'reversed':(y,h)}
    stats={}
    for name,(first,second) in modes.items():
        mus,lvs,nodes=[],[],[]
        for start in range(0,len(h),1):
            n=encoder(torch.cat((first[start:start+1],second[start:start+1]),dim=1))
            mu,lv=vq._encode(n)
            mus.append(mu);lvs.append(lv);nodes.append(n)
        stats[name]=(torch.cat(mus),torch.cat(lvs),torch.cat(nodes))
        print('encoded',name,flush=True)
    mu,lv,nodes=stats['true']
    reproduction=float((mu-z_ref).abs().max())
    if reproduction>1e-5:
        raise RuntimeError(f'Cached mu mismatch: {reproduction}')
    print('verified mu max error',reproduction,flush=True)
    posterior={}
    for name,(m,l,n) in stats.items():
        posterior[name]=dict(mu=summarize_tensor(m),logvar=summarize_tensor(l),nodes=summarize_tensor(n),
                            squared_mu_change=float((m-mu).square().sum(-1).mean()),
                            squared_logvar_change=float((l-lv).square().sum(-1).mean()),
                            symmetric_kl_to_true=float((gaussian_kl(mu,lv,m,l)+gaussian_kl(m,l,mu,lv)).mean()/2))
    gen=torch.Generator().manual_seed(1907)
    rows=[]
    for start in range(0,len(h),1):
        current,target,m,l=h[start:start+1],y[start:start+1],mu[start:start+1],lv[start:start+1]
        other_m=mu.roll(1,0)[start:start+1];other_l=lv.roll(1,0)[start:start+1]
        mean_true=decoder(current,m)
        per=dict(mean_mse=(mean_true-target).square().flatten(1).mean(1),
                 mean_swap_mse=(decoder(current,other_m)-target).square().flatten(1).mean(1))
        for sample in range(args.samples):
            eps=torch.randn(m.shape,generator=gen)
            codes={'sample_true':m+(l*.5).exp()*eps,
                   'sample_mu_swapped':other_m+(l*.5).exp()*eps,
                   'sample_logvar_swapped':m+(other_l*.5).exp()*eps,
                   'sample_both_swapped':other_m+(other_l*.5).exp()*eps}
            prediction={name:decoder(current,z) for name,z in codes.items()}
            for name,pred in prediction.items():
                key=name+'_mse';per[key]=per.get(key,0)+(pred-target).square().flatten(1).mean(1)/args.samples
                key=name+'_prediction_change';per[key]=per.get(key,0)+(pred-prediction['sample_true']).square().flatten(1).mean(1)/args.samples
        for i in range(len(current)):
            row=dict(data['rows'][int(indices[start+i])],**{key:float(value[i]) for key,value in per.items()})
            rows.append(row)
        print('decoded',start+len(current),'/',len(h),flush=True)
    metrics={key:sum(row[key] for row in rows)/len(rows) for key in per}
    report=dict(checkpoint_step=step,pairs=len(h),samples=args.samples,mu_max_reproduction_error=reproduction,
                posterior=posterior,decoder_metrics=metrics,rows=rows,
                limitations='Swapped/no-change/reversed pairs are dependency probes, not demonstrated valid counterfactual trajectories. Mean diagnostics do not identify action semantics or global collapse.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(posterior=posterior,decoder_metrics=metrics),indent=2),flush=True)


if __name__=='__main__':main()
