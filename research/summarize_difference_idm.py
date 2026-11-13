"""Audit the completed paired difference-IDM experiment and reload final models."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import numpy as np
import torch
from research.train_early_condition import file_hash,load_inputs
from research.evaluate_prediction_metrics import metrics
from research.idm.load_difference_idm import load_trained_model


def write(path,obj):
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,default=ROOT/'runs/difference-idm-joint-20261113')
    p.add_argument('--source',type=Path,default=ROOT/'runs/layerwise-gpu-inputs-20261113')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    torch.cuda.set_device(a.device);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    runs={};hashes={};protocols={}
    for arm in ('joint','frozen_encoder'):
        runs[arm]=[];protocols[arm]=[]
        for seed in (42,43,44):
            root=a.runs/f'{arm}-{seed}'
            result=json.loads((root/'results.json').read_text())
            protocol=json.loads((root/'protocol.json').read_text())
            assert result['complete'] and not result['smoke_only']
            assert protocol['steps']==400 and protocol['seed']==seed
            assert result['changed']=={'encoder':arm=='joint','head':True,'decoder':True}
            assert file_hash(root/'model.pt')==result['checkpoint_sha256']
            assert file_hash(root/'executed_source.py')==protocol['source_sha256'][str(ROOT/'research/train_difference_idm.py')]
            hashes[str(root/'results.json')]=file_hash(root/'results.json')
            runs[arm].append(result);protocols[arm].append(protocol)
    for i in range(3):
        left,right=protocols['joint'][i],protocols['frozen_encoder'][i]
        for key in ('initial_hashes','batch_hash','lr','batch_size','microbatch','objective','source_sha256'):
            assert left[key]==right[key],key
    data,original,input_hashes,_=load_inputs(a.source)
    hashes.update(input_hashes)
    original=original.to(a.device).eval().requires_grad_(False)
    original_rows={}
    for split,d in data.items():
        rows=[]
        for start in range(0,len(d['rows']),8):
            sl=slice(start,start+8)
            x,y,z=[d[k][sl].to(a.device) for k in ('current','future','native_codes')]
            vals=metrics(original(x,z),y,x)
            for j in range(len(x)):
                rows.append({'source_row':d['rows'][start+j],**{k:float(v[j]) for k,v in vals.items()}})
        original_rows[split]=rows
    del original
    report={'complete':False,'protocols':protocols,'source_sha256':hashes,
        'script_sha256':file_hash(Path(__file__)),'splits':{},
        'scope':protocols['joint'][0]['scope'],
        'interval_scope':'20000 paired episode resamples of seed-mean errors; conditional on pretrained weights and these minibatch seeds. No multiplicity correction or unseen-test claim.'}
    for split,d in data.items():
        modelrows={'original_native':[original_rows[split]]}
        modelrows.update({arm:[r['evaluation'][split]['rows'] for r in results] for arm,results in runs.items()})
        for rowsets in modelrows.values():
            for rows in rowsets:assert [r['source_row'] for r in rows]==d['rows']
        # Historical local GPU runs share the features, but used a frozen head,
        # different learning rates/capacity/objectives; engineering references.
        for arm in ('external','layerwise','continuation'):
            rows=[]
            for seed in (42,43,44):
                path=ROOT/f'runs/layerwise-gpu-20261113/{arm}-{seed}/results.json'
                result=json.loads(path.read_text());hashes[str(path)]=file_hash(path)
                rr=result['evaluation'][split]['rows']
                assert [r['source_row'] for r in rr]==d['rows']
                rows.append(rr)
            modelrows['previous_'+arm]=rows
        modelmetrics={};perrow={}
        for name,rowsets in modelrows.items():
            metric_names=[k for k in rowsets[0][0] if k!='source_row']
            values={k:np.array([[r[k] for r in rows] for rows in rowsets]) for k in metric_names}
            perrow[name]=values['mse'].mean(0)
            seed_mse=values['mse'].mean(1).tolist()
            modelmetrics[name]={'metrics':{k:float(v.mean()) for k,v in values.items()},
                'seed_mse':seed_mse,'seed_sd':statistics.stdev(seed_mse) if len(seed_mse)>1 else None}
        rng=np.random.default_rng(20261113);draws=rng.integers(len(d['rows']),size=(20000,len(d['rows'])))
        comparisons={}
        for ref in ('frozen_encoder','original_native','previous_external','previous_layerwise','previous_continuation'):
            diff=perrow['joint']-perrow[ref]
            comparisons['joint_minus_'+ref]={'mse_difference':float(diff.mean()),
                'relative_reduction':float(1-perrow['joint'].mean()/perrow[ref].mean()),
                'conditional_paired_ci95':np.quantile(diff[draws].mean(1),[.025,.975]).tolist()}
        report['splits'][split]={'models':modelmetrics,'comparisons':comparisons,
            'rows':d['rows'],'per_example_seed_mean_mse':{k:v.tolist() for k,v in perrow.items()}}
    # Standalone reload of each final checkpoint and exact-shape prediction replay.
    reloads={};timing={}
    for arm in ('joint','frozen_encoder'):
        reloads[arm]=[]
        for i,seed in enumerate((42,43,44)):
            model,protocol=load_trained_model(a.runs/f'{arm}-{seed}/model.pt',a.device)
            d=data['observed96'];x,y=[d[k][:8].to(a.device) for k in ('current','future')]
            pred,z=model(x,y)
            actual=metrics(pred,y,x)['mse'].cpu()
            expected=torch.tensor([r['mse'] for r in runs[arm][i]['evaluation']['observed96']['rows'][:8]])
            error=float((actual-expected).abs().max());assert error<1e-6,error
            reloads[arm].append({'seed':seed,'first8_mse_max_abs':error})
            if seed==42:
                timing[arm]={}
                for batch in (1,8):
                    bx,by=x[:batch],y[:batch];bz=model.idm(bx,by)
                    calls={'idm_and_decoder':lambda:model(bx,by),'decoder_only':lambda:model.predict(bx,bz)}
                    timing[arm][str(batch)]={}
                    for name,call in calls.items():
                        for _ in range(5):call()
                        durations=[]
                        for _ in range(20):
                            torch.cuda.synchronize();started=time.perf_counter();call();torch.cuda.synchronize()
                            durations.append((time.perf_counter()-started)*1000)
                        timing[arm][str(batch)][name]={'median_ms':statistics.median(durations),'trials_ms':durations}
            del model
            torch.cuda.empty_cache()
    report['checkpoint_reload']=reloads
    report['timing']={'device':torch.cuda.get_device_name(),'data':timing,
        'scope':'GPU float32 highest, warmup5 and20 synchronized wall measurements. No competing fit jobs. DINO/data loading excluded; full path consumes true future.'}
    report['complete']=True
    write(a.output/'results.json',report)
    print(json.dumps({s:{k:v['metrics']['mse'] for k,v in ss['models'].items()} for s,ss in report['splits'].items()},indent=2))


if __name__=='__main__':main()
