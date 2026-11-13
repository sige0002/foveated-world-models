"""Evaluate fixed models with the original visual loss and complementary metrics."""
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.train_early_condition import load_inputs
from research.train_cached_decoders import make_decoder, file_hash


def metrics(pred, target, current):
    err=(pred-target).square()
    patch_error=err.mean(-1).squeeze(1)
    change=(target-current).square().mean(-1).squeeze(1)
    order=change.argsort(dim=-1,stable=True)
    n=order.shape[-1]
    return {'mse':err.flatten(1).mean(1),
            'smooth_l1_beta_0_1':F.smooth_l1_loss(pred,target,beta=.1,reduction='none').flatten(1).mean(1),
            'mae':(pred-target).abs().flatten(1).mean(1),
            'cosine_similarity':F.cosine_similarity(pred,target,dim=-1).flatten(1).mean(1),
            'high_feature_change_decile_mse':patch_error.gather(1,order[:,-math.ceil(n*.1):]).mean(1),
            'low_feature_change_half_mse':patch_error.gather(1,order[:,:n//2]).mean(1)}


@torch.inference_mode()
def main():
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    out=ROOT/'research/results/prediction-multimetric-20261113'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    data,original,hashes,_=load_inputs(ROOT)
    models={}
    checkpoints={
        'correction400':ROOT/'runs/reference-idm42-nested512-comparison400-20261113/prior_residual-seed42.pt',
        'correction1600':ROOT/'runs/reference-idm42-nested512-residual1600-20261113/prior_residual-seed42.pt',
        'constant400':ROOT/'runs/constant-nested512-residual400-20261113/prior_residual-seed42.pt'}
    for name,path in checkpoints.items():
        saved=torch.load(path,weights_only=True,map_location='cpu')
        assert saved['seed']==42 and saved['variant']=='prior_residual'
        assert saved['steps']==(1600 if name=='correction1600' else 400)
        model=make_decoder('prior_residual',saved['dim'],saved['layers'],768,32,16)
        model.load_state_dict(saved['state_dict'],strict=True)
        models[name]=model.eval().requires_grad_(False);hashes[str(path)]=file_hash(path)
    constant_root=ROOT/'runs/constant-nested512-prior10000'
    constant_manifest=json.loads((constant_root/'cache_manifest.json').read_text())
    constant_path=constant_root/'train.pt'
    hashes[str(constant_path)]=file_hash(constant_path)
    assert hashes[str(constant_path)]==constant_manifest['cache_sha256']['train']
    constant=torch.load(constant_path,weights_only=True,mmap=True)['codes']
    assert torch.equal(constant,constant[:1].expand_as(constant))
    constant=constant[:1].clone()
    native={};native_root=ROOT/'runs/frozen-model-step10000'
    nm=json.loads((native_root/'cache_manifest.json').read_text())
    for split in ('validation','observed96'):
        p=native_root/(split+'.pt');hashes[str(p)]=file_hash(p)
        assert hashes[str(p)]==nm['cache_sha256'][split]
        native[split]=torch.load(p,weights_only=True,mmap=True)
        for key in ('current','future'):
            assert torch.equal(data[split][key],native[split][key])
        assert [(r['episode'],r['index']) for r in data[split]['rows']]==[(r['episode'],r['index']) for r in native[split]['rows']]
    names=['original_native','zero_prior',*models]
    protocol={'models':names,'source_sha256':hashes,'script_sha256':file_hash(Path(__file__)),
        'threads':2,'precision':'float32 highest','fixed_weights':True,'no_training':True,
        'metrics':['mse','smooth_l1_beta_0_1','mae','cosine_similarity','high_feature_change_decile_mse','low_feature_change_half_mse'],
        'primary_added_metric':'smooth_l1_beta_0_1 matches the original visual reconstruction loss, not its auxiliary state+KL total',
        'region_definition':'Per-example ranking of true current-future feature change; top ceil10% and bottom50%. Not a robot segmentation mask.',
        'scope':'Fixed seed42 checkpoints, reused64/96, posterior true-future codes. Original model has original code; correction has reference code. Extra training and parameter costs differ. No official released baseline, physical behavior, or deployment claim.'}
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    report={'protocol_sha256':file_hash(out/'protocol.json'),'splits':{}}
    motion=json.loads((ROOT/'research/results/pair-motion-audit-20261113/results.json').read_text())
    native_audit=json.loads((ROOT/'research/idm/model-step10000-baseline-audit.json').read_text())
    for split in ('validation','observed96'):
        d=data[split];rows={name:[] for name in names}
        for start in range(0,len(d['rows']),8):
            sl=slice(start,start+8);x,y,z,b=[d[k][sl] for k in ('current','future','codes','baseline')]
            preds={'original_native':original(x,native[split]['codes'][sl]),'zero_prior':b}
            preds.update({name:m(x,constant.expand_as(z) if name=='constant400' else z,baseline=b) for name,m in models.items()})
            for name,pred in preds.items():
                vals=metrics(pred,y,x)
                for j in range(len(x)):
                    rows[name].append({'source_row':d['rows'][start+j],**{k:float(v[j]) for k,v in vals.items()}})
            print(split,'rows',start+len(x),flush=True)
        split_result={}
        moving=[i for i,r in enumerate(motion['selected'][split]['rows']) if r['max_arm_endpoint_m']>=.05]
        assert [(r['episode'],r['index']) for r in motion['selected'][split]['rows']]==[(r['episode'],r['index']) for r in d['rows']]
        rng=np.random.default_rng(20261113);draws=rng.integers(0,len(d['rows']),(20000,len(d['rows'])))
        for name,rs in rows.items():
            means={k:float(np.mean([r[k] for r in rs])) for k in protocol['metrics']}
            differences={}
            for metric in protocol['metrics']:
                diff=np.array([r[metric]-b[metric] for r,b in zip(rs,rows['original_native'])])
                differences[metric]={'difference':float(diff.mean()),'conditional_paired_ci95':np.quantile(diff[draws].mean(1),[.025,.975]).tolist()}
            split_result[name]={'metrics':means,'moving_at_least_5cm':{'count':len(moving),'metrics':{k:float(np.mean([rs[i][k] for i in moving])) for k in protocol['metrics']}},
                'minus_original_native':differences,'rows':rs}
        assert abs(split_result['original_native']['metrics']['mse']-native_audit['results'][split]['original_model_mse'])<1e-6
        report['splits'][split]=split_result
        print(split,{n:v['metrics'] for n,v in split_result.items()},flush=True)
        (out/'partial-results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    report['complete']=True
    report['interval_scope']='Fixed-model episode bootstrap conditional on trained weights; excludes training and adaptive selection uncertainty.'
    (out/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
