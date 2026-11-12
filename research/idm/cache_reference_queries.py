"""Cache frozen IDM queries from the exact pre-existing Stage1 feature pairs.

No image decoding, augmentation, new pair selection, GPU access, or training.
Full-cache default preserves all 128/64 parent rows; --pairs limits each split.
"""
import argparse, hashlib, json, sys
from pathlib import Path
import numpy as np
import torch
import yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'vendor/lawam'))
from latent_action_model.core.utils.lam_encoder import LAMEncoder
from latent_action_model.core.vq import VAEQuantizer

def sha256(path):
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--pairs',type=int,default=0,help='0: all rows; otherwise uniform existing cache indices')
    a=p.parse_args()
    if a.output.exists() or a.pairs<0 or a.pairs==1:p.error('Use new output and zero or >=2 pairs')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('medium')
    manifest=json.loads((a.cache/'cache_manifest.json').read_text())
    hashes={split:sha256(a.cache/f'{split}.pt') for split in ('train','validation')}
    if hashes!=manifest['cache_sha256']:raise ValueError('Parent cache hashes do not match extraction manifest')
    cfg=yaml.safe_load((a.cache/'resolved_config.yaml').read_text())['model']
    if cfg['enc_add_state'] or cfg['vq_type'] not in ('vae','beta_vae'):raise ValueError('Unsupported encoder')
    enc=LAMEncoder(context_dim=cfg['dim'],input_dim=768,num_heads=cfg['num_heads'],
        num_queries=cfg['num_queries'],grid_hw=tuple(v//cfg['patch_size'] for v in cfg['image_hw']),
        ffn_expansion_factor=cfg['ffn_expansion_factor'],dropout=cfg['dropout'],code_dim=cfg['code_dim'],
        num_layers=cfg['enc_layers'],num_frames=cfg['num_frames'],modal_mask=cfg['enc_modal_mask'],
        add_state=False,max_state_dim=cfg['max_state_dim'],num_embodiments=cfg['num_embodiments']).eval()
    vq=VAEQuantizer(code_dim=cfg['code_dim'],**cfg['vq_kwargs']).eval()
    saved=torch.load(a.cache/'reference_weights.pt',weights_only=True,mmap=True)
    for name,mod in [('encoder',enc),('vq',vq)]:
        prefix='lam.'+name+'.';mod.load_state_dict({k[len(prefix):]:v for k,v in saved['state_dict'].items() if k.startswith(prefix)},strict=True)
    step=saved['global_step'];del saved
    capture={};original=enc.out_proj.forward
    def forward(query):
        capture['query']=query.detach().clone()
        return original(query)
    enc.out_proj.forward=forward
    extracted={};errors={}
    for split in ('train','validation'):
        data=torch.load(a.cache/f'{split}.pt',weights_only=True)
        n=len(data['current']);indices=np.arange(n) if not a.pairs else np.linspace(0,n-1,min(n,a.pairs),dtype=int)
        modes={key:[] for key in ('query','reference','future_swapped')};maxerr=0.
        for j,i in enumerate(indices):
            x,y=data['current'][i:i+1],data['future'][i:i+1]
            nodes=enc(torch.cat((x,y),1));mu,_=vq._encode(nodes)
            maxerr=max(maxerr,float((mu-data['codes'][i:i+1]).abs().max()))
            modes['query'].append(capture['query'])
            enc(torch.cat((x,x),1));modes['reference'].append(capture['query'])
            k=int(indices[(j+1)%len(indices)])
            enc(torch.cat((x,data['future'][k:k+1]),1));modes['future_swapped'].append(capture['query'])
            print(split,j+1,'/',len(indices),'mu_error',maxerr,flush=True)
        if maxerr>1e-5:raise ValueError(f'{split} mu reproduction failure {maxerr}')
        extracted[split]={key:torch.cat(value) for key,value in modes.items()}
        extracted[split].update(indices=torch.tensor(indices),rows=[data['rows'][int(i)] for i in indices])
        errors[split]=maxerr
    if {r['episode'] for r in extracted['train']['rows']} & {r['episode'] for r in extracted['validation']['rows']}:
        raise ValueError('Episode overlap between extracted splits')
    train=extracted['train'];q=train['query'];d=q-train['reference']
    center=q.mean(0,keepdim=True);floor=1e-3
    # Scalars keep feature anisotropy and within-split amplitude; validation never calibrates.
    scales={'raw':q.square().mean().sqrt().clamp_min(floor),
        'centered':(q-center).square().mean().sqrt().clamp_min(floor),
        'reference':d.square().mean().sqrt().clamp_min(floor)}
    diagnostics={}
    for split,values in extracted.items():
        q,r,sw=[values[k] for k in ('query','reference','future_swapped')]
        variants={'raw':q/scales['raw'],'centered':(q-center)/scales['centered'],'reference':(q-r)/scales['reference']}
        diagnostics[split]={}
        for name,value in variants.items():
            perturb=(sw-q)/scales[name]
            diagnostics[split][name]={'mean_norm_squared':float(value.mean(0).square().sum()),
                'between_pair_variance':float(value.var(0,unbiased=False).sum()),
                'future_swap_squared_difference':float(perturb.square().sum(-1).mean())}
    a.output.mkdir(parents=True)
    for split,value in extracted.items():torch.save(value,a.output/f'{split}.pt')
    torch.save(dict(center=center,scales=scales,floor=floor),a.output/'train_calibration.pt')
    report=dict(step=step,parent_cache=str(a.cache.resolve()),parent_cache_sha256=hashes,
        parent_manifest_sha256=sha256(a.cache/'cache_manifest.json'),mu_max_errors=errors,
        pairs={k:len(v['query']) for k,v in extracted.items()},train_scales={k:float(v) for k,v in scales.items()},
        diagnostics=diagnostics,
        limitations='Frozen-query dependence only. Future swaps may be OOD. No camera labels: camera invariance cannot be evaluated. Different train-calibrated scales change numerical units, not information. No decoder utilization or learned improvement measured.')
    (a.output/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
