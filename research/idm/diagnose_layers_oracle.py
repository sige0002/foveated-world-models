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
    encoder.requires_grad_(False);vq.requires_grad_(False);decoder.requires_grad_(False)
    captured={}
    def hook(name,query=False):
        def capture(module,inputs,output):
            value=output[:,-1:] if query else output
            captured[name]=value.detach().clone()
        return capture
    originals=[]
    def wrap(module,name,query=False):
        original=module.forward
        def forward(*inputs,**kwargs):
            output=original(*inputs,**kwargs)
            hook(name,query)(module,inputs,output)
            return output
        originals.append((module,original));module.forward=forward
    wrap(encoder.project_in,'project_input')
    wrap(encoder.QFormer.q_cross_attn,'initial_cross_attention')
    wrap(encoder.out_proj,'output_projection')
    for i,layer in enumerate(encoder.QFormer.layers):wrap(layer,f'layer_{i+1:02}',True)
    records={};mus=[];lvs=[]
    for i in range(len(h)):
        with torch.inference_mode():
            nodes=encoder(torch.cat((h[i:i+1],y[i:i+1]),1))
            mu,lv=vq._encode(nodes);mus.append(mu);lvs.append(lv)
            true={k:v.clone() for k,v in captured.items()}
            encoder(torch.cat((h[i:i+1],y[(i+1)%len(h):(i+1)%len(h)+1]),1))
            for name,base in true.items():
                alt=captured[name]
                record=records.setdefault(name,dict(true=[],swapped=[]))
                record['true'].append(base.flatten());record['swapped'].append(alt.flatten())
        print('layer pair',i+1,flush=True)
    for module,original in originals:module.forward=original
    layer_stats={}
    for name,record in records.items():
        base=torch.stack(record['true']).double();alt=torch.stack(record['swapped']).double()
        common=base.mean(0).square().sum();change=(alt-base).square().sum(1).mean()
        layer_stats[name]=dict(common_norm_squared=float(common),
            between_pair_variance=float(base.var(0,unbiased=False).sum()),
            future_swap_squared_difference=float(change),difference_to_common=float(change/common))
    mu=torch.cat(mus).detach();lv=torch.cat(lvs).detach()
    error=float((mu-z_ref).abs().max())
    if error>1e-5:raise RuntimeError(f'mu reproduction {error}')
    print('layers',json.dumps(layer_stats),flush=True)
    oracle=[]
    # Radius is in posterior-standard-deviation coordinates, total L2, not per dimension.
    for i in range(len(h)):
        current,target=h[i:i+1],y[i:i+1];base=mu[i:i+1];std=(lv[i:i+1]*.5).exp()
        offset=torch.zeros_like(base,requires_grad=True)
        opt=torch.optim.Adam([offset],lr=.1)
        row={'pair':int(indices[i]),'radius':1.,'steps':12,'penalty_weight':1e-4}
        for t in range(13):
            pred=decoder(current,base+std*offset)
            mse=(pred-target).square().mean()
            if t==0:row['initial_mse']=float(mse.detach())
            if t==12:
                row['final_mse']=float(mse.detach());break
            opt.zero_grad(set_to_none=True)
            (mse+1e-4*offset.square().sum()).backward();opt.step()
            with torch.no_grad():offset.mul_(min(1.,1./float(offset.norm().clamp_min(1e-12))))
        row['mahalanobis_radius']=float(offset.detach().norm())
        row['latent_displacement_l2']=float((std*offset.detach()).norm())
        oracle.append(row);print('oracle',row,flush=True)
    report=dict(checkpoint_step=step,pairs=len(h),mu_max_reproduction_error=error,
        layer_stats=layer_stats,oracle=oracle,
        warning='Future-target oracle with frozen decoder; not a deployable improvement. Forward wrappers preserve transformer fast paths; exact mu reproduction is checked. Tiny changes below numerical precision are not interpreted.')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')

if __name__=='__main__':main()
