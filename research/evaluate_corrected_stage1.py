"""Revalidate a corrected LeRobot Stage-1 checkpoint without legacy caches."""
import argparse
import functools
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['LAM_ENABLE_MANUAL_WANDB']='0'
import torch
from torch.utils.data import DataLoader, Subset
from train_stage1 import load_config, check_resume_contract
from vendor.dataset.lerobot_robomind import LeRobotSource, LeRobotRoboMINDDataset, lerobot_collate
from latent_action_model.core.lam_lightinng import VJEPA_LAM
from research.evaluate_prediction_metrics import metrics
from research.train_early_condition import file_hash


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=0,help='Smoke only: first N pairs per split; 0 evaluates all')
    a=p.parse_args()
    if a.output.exists():p.error('Output must be new')
    torch.set_num_threads(4);torch.manual_seed(42)
    cfg=load_config(a.run/'resolved_config.yaml');mc=cfg['model'];dc=cfg['data']
    assert dc['format']=='lerobot_v3'
    source=LeRobotSource(dc['root'])
    opts={k:mc[k] for k in ('image_hw','max_state_dim','num_embodiments')}
    opts.update({k:dc[k] for k in ('horizon_seconds','tolerance_seconds')})
    train=LeRobotRoboMINDDataset(source,'train',**opts)
    ck=a.run/'checkpoints/final.ckpt'
    saved=torch.load(ck,map_location='cpu',weights_only=True,mmap=True)
    check_resume_contract(saved,train.contract)
    model=VJEPA_LAM(**mc).eval().requires_grad_(False)
    model.load_state_dict(saved['state_dict'],strict=True)
    step=saved['global_step'];del saved
    model=model.cuda();torch.set_float32_matmul_precision('highest')
    report={'checkpoint_sha256':file_hash(ck),'global_step':step,
            'script_sha256':file_hash(Path(__file__)),'data_contract':train.contract,
            'precision':'float32 highest, no augmentation','limit':a.limit,
            'scope':'Corrected raw-data reconstruction with true-future posterior mean; zero and cyclic code permutation are diagnostic interventions. No legacy-cache numerical comparison or deployment claim. Test split has prior research exposure.',
            'splits':{}}
    for split in ('validation','test'):
        ds=LeRobotRoboMINDDataset(source,split,**opts)
        indices=list(range(min(a.limit,len(ds)) if a.limit else len(ds)))
        loader=DataLoader(Subset(ds,indices),batch_size=2,num_workers=2,
                          collate_fn=functools.partial(lerobot_collate,max_state_dim=mc['max_state_dim']))
        cache={k:[] for k in ('current','future','codes')}
        native=[]
        for i,batch in enumerate(loader):
            batch=model.transfer_batch_to_device(batch,torch.device('cuda'),0)
            out=model.lam.get_latent_action(batch['videos'],batch['states'],batch['dec_videos'],
                    state_mask=batch['state_mask'],embodiment_ids=batch['embodiment_ids'],predict_future_frame=True)
            for k,key in [('current','dec_in'),('future','tgt'),('codes','quantized')]:
                cache[k].append(out[key].cpu())
            vals=metrics(out['recon'],out['tgt'],out['dec_in'])
            native.extend([{k:float(v[j]) for k,v in vals.items()} for j in range(len(out['recon']))])
            if i%100==0:print(split,'encoded',len(native),'/',len(indices),flush=True)
        cache={k:torch.cat(v) for k,v in cache.items()}
        rows=[]
        for start in range(0,len(indices),2):
            sl=slice(start,start+2);x,y,z=[cache[k][sl].cuda() for k in ('current','future','codes')]
            conditions={'zero':torch.zeros_like(z),'permuted':cache['codes'].roll(1,0)[sl].cuda()}
            measures={name:metrics(model.lam.decoder(x,code),y,x) for name,code in conditions.items()}
            for j in range(len(x)):
                idx=indices[start+j];episode,first,last,tail=ds.pairs[idx]
                rows.append({'pair_index':idx,'episode':int(ds.rows[episode]['episode_index']),
                    'first':first,'last':last,'tail':bool(tail),'native':native[start+j],
                    **{name:{k:float(v[j]) for k,v in values.items()} for name,values in measures.items()}})
        assert all(torch.isfinite(t).all() for t in cache.values())
        means={name:{k:sum(row[name][k] for row in rows)/len(rows) for k in native[0]}
               for name in ('native','zero','permuted')}
        z=cache['codes'].double().flatten(1)
        report['splits'][split]={'count':len(rows),'metrics':means,
            'z_mean':z.mean(0).tolist(),'z_variance':z.var(0,unbiased=False).tolist(),
            'zero_relative_mse_gain_percent':100*(1-means['native']['mse']/means['zero']['mse']),
            'rows':rows}
        print(split,means,flush=True)
    report['complete']=True
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
