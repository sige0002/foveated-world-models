"""Post-hoc fixed-weight trace of cached native queries through the VAE head.

Small matrix operations on CPU; no training, no test selection. The C,C
reference is a synthetic no-change diagnostic, not an action intervention.
"""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import torch
import yaml
from latent_action_model.core.vq import VAEQuantizer
from research.z_utility_matched import SOURCE,CHECKPOINT
from research.train_early_condition import file_hash


def geometry(x,ref):
    x=x.double().flatten(1);ref=ref.double().flatten(1)
    common=float(x.mean(0).square().sum());variation=float(x.var(0,unbiased=False).sum())
    delta=float((x-ref).square().sum(1).mean())
    return {'common_mean_energy':common,'between_example_variance':variation,
            'variance_fraction':variation/(common+variation),
            'same_current_future_vs_reference_energy':delta,'reference_difference_over_common':delta/common}


@torch.no_grad()
def main():
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    out=ROOT/'research/results/z-utility-matched-20261113'
    report=json.loads((out/'results.json').read_text());assert report['complete']
    cfg=yaml.safe_load((SOURCE/'runs/frozen-model-step10000/resolved_config.yaml').read_text())['model']
    digest=file_hash(CHECKPOINT);assert digest==report['original_native']['checkpoint_sha256']
    w=torch.load(CHECKPOINT,weights_only=True,map_location='cpu',mmap=True)['state_dict']
    proj=torch.nn.Linear(cfg['dim'],cfg['code_dim'])
    proj.load_state_dict({k.removeprefix('lam.encoder.out_proj.'):v for k,v in w.items() if k.startswith('lam.encoder.out_proj.')})
    vq=VAEQuantizer(code_dim=cfg['code_dim'],**cfg['vq_kwargs'])
    vq.load_state_dict({k.removeprefix('lam.vq.'):v for k,v in w.items() if k.startswith('lam.vq.')})
    result={'checkpoint_sha256':digest,'script_sha256':file_hash(Path(__file__)),'vq_kwargs':cfg['vq_kwargs'],'splits':{},
            'scope':'Post-hoc CPU trace of cached Q(C,F) and Q(C,C) through unchanged original projection/norm/VAE. Ratios are diagnostics, not mutual information or causal attribution.'}
    for split,folder,disk in [('train','reference-idm42-nested512-prior10000','train'),
                             ('validation','reference-idm42-nested512-prior10000','validation'),
                             ('observed96','reference-idm42-observed96-prior10000','validation')]:
        folder=SOURCE/'runs'/folder;path=folder/(disk+'.pt')
        manifest=json.loads((folder/'cache_manifest.json').read_text());dh=file_hash(path);assert dh==manifest['cache_sha256'][disk]
        d=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
        q,ref=d['query'],d['reference'];pq,pr=proj(q).mean(1,keepdim=True),proj(ref).mean(1,keepdim=True)
        nq,nr=vq.pre_norm(pq),vq.pre_norm(pr);mu,lv=vq._encode(proj(q));mr,lr=vq._encode(proj(ref))
        error=float((mu-d['native_codes']).abs().max());assert error<1e-4,error
        result['splits'][split]={'source_sha256':dh,'cached_native_code_max_abs':error,
            'stages':{'query':geometry(q,ref),'projection':geometry(pq,pr),'pre_norm':geometry(nq,nr),
                      'mu':geometry(mu,mr),'logvar':geometry(lv,lr)}}
    (out/'native-head-trace.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for s,v in result['splits'].items():print(s,v['cached_native_code_max_abs'],v['stages'],flush=True)


if __name__=='__main__':main()
