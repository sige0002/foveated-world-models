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
    queries=[]
    original=encoder.out_proj.forward
    def forward(q):
        queries.append(q.detach().clone())
        return original(q)
    encoder.out_proj.forward=forward
    with torch.inference_mode():
        for i in range(len(h)):encoder(torch.cat((h[i:i+1],y[i:i+1]),1))
    q=torch.cat(queries).double();W=encoder.out_proj.weight.double();b=encoder.out_proj.bias.double()
    eps=vq.pre_norm.eps
    def ln(v,epsilon=eps):return (v-v.mean(-1,keepdim=True))/torch.sqrt(v.var(-1,keepdim=True,unbiased=False)+epsilon)
    m=q.mean(-1,keepdim=True);scale=torch.sqrt(q.var(-1,keepdim=True,unbiased=False)+1e-5)
    raw=q@W.T+b
    proposed=ln(q)@W.T+b
    algebra=raw-m*(W@torch.ones(q.shape[-1],dtype=q.dtype))+(scale-1)*b
    # Epsilon scales under multiplication: LN(algebra/scale, eps)=LN(algebra, eps*scale²).
    expected=(algebra-algebra.mean(-1,keepdim=True))/torch.sqrt(algebra.var(-1,keepdim=True,unbiased=False)+eps*scale.square())
    out=dict(query_coordinate_mean=float(m.mean()),query_coordinate_std=float(scale.mean()),
        projection_bias_norm=float(b.norm()),projection_ones_norm=float((W@torch.ones(q.shape[-1],dtype=q.dtype)).norm()),
        normalization_algebra_max_error=float((ln(proposed)-expected).abs().max()),
        original_vs_preLN_normalized_rms=float((ln(raw)-ln(proposed)).square().mean().sqrt()),
        original_between_pair_variance=float(ln(raw).var(0,unbiased=False).sum()),
        preLN_between_pair_variance=float(ln(proposed).var(0,unbiased=False).sum()),
        scale_only_without_bias_max_difference=float((ln(q@W.T,0)-ln((q/scale)@W.T,0)).abs().max()),
        limitation='Eight held-out inputs. Algebra and representation diagnostic only; no decoder evaluation, no training improvement claim.')
    args.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2),flush=True)

if __name__=='__main__':main()
