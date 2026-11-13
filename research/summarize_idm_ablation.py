"""Compare all fixed milestones, then score the untouched test at terminal1600 only."""
import argparse
import json
from pathlib import Path
import statistics
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import numpy as np
import torch
from research.train_early_condition import file_hash,load_inputs
from research.train_idm_ablation import evaluate,write
from research.idm.load_difference_idm import load_trained_model


def verify_checkpoint_protocol(checkpoint_protocol, recorded_protocol):
    # torch.save preserves tuples; the JSON record serializes them as lists.
    # Compare the same serialized representation without dropping any fields.
    if json.loads(json.dumps(checkpoint_protocol)) != recorded_protocol:
        raise ValueError('Checkpoint protocol differs from recorded protocol')


def paired_comparison(left,right,groups):
    """Paired risk difference; seed and finite-cluster uncertainty remain distinct."""
    difference=left-right;delta=difference.mean(0)
    rng=np.random.default_rng(20261113)
    draws=rng.integers(len(delta),size=(20000,len(delta)))
    unique=sorted(set(groups));sums=np.array([delta[np.array(groups)==g].sum() for g in unique])
    counts=np.array([sum(x==g for x in groups) for g in unique])
    grouped=rng.integers(len(unique),size=(20000,len(unique)))
    return {'mse_difference':float(delta.mean()),'relative_reduction':float(1-left.mean()/right.mean()),
        'seed_differences':difference.mean(1).tolist(),
        'episode_paired_ci95':np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
        'video_cluster_paired_ci95':np.quantile(sums[grouped].sum(1)/counts[grouped].sum(1),[.025,.975]).tolist(),
        'video_cluster_count':len(unique)}


def aggregate(evaluations,episode_groups):
    out={}
    for split in evaluations['difference'][0]:
        rows=evaluations['difference'][0][split]['rows'];source=[r['source_row'] for r in rows]
        assert len({r['episode'] for r in source})==len(source), 'Episode bootstrap requires one row per episode'
        values={};models={}
        for arm,results in evaluations.items():
            rr=[r[split]['rows'] for r in results]
            assert all([r['source_row'] for r in rowset]==source for rowset in rr)
            arrays={k:np.array([[r[k] for r in rowset] for rowset in rr]) for k in rows[0] if k!='source_row'}
            values[arm]=arrays['mse'];seed_mse=arrays['mse'].mean(1).tolist()
            models[arm]={'metrics':{k:float(v.mean()) for k,v in arrays.items()},'seed_mse':seed_mse,
                'seed_sd':statistics.stdev(seed_mse),'per_example_mse':arrays['mse'].tolist()}
        groups=[episode_groups[r['episode']] for r in source]
        out[split]={'models':models,'difference_minus_raw':paired_comparison(values['difference'],values['raw'],groups),'rows':source}
    return out


def verify_runs(runs):
    protocols={};evaluations={step:{} for step in (400,800,1600)};hashes={}
    for arm in ('difference','raw'):
        protocols[arm]=[]
        for seed in (42,43,44):
            root=runs/f'{arm}-{seed}'
            result=json.loads((root/'results.json').read_text());protocol=json.loads((root/'protocol.json').read_text())
            assert result['complete'] and not result['smoke_only'] and all(result['changed'].values())
            assert protocol['representation']==arm and protocol['seed']==seed and protocol['steps']==1600
            assert protocol['eval_steps']==[400,800,1600] and not protocol['smoke_only']
            assert file_hash(root/'executed_source.py')==protocol['source_sha256'][str(ROOT/'research/train_idm_ablation.py')]
            for step in (400,800,1600):
                p=root/f'evaluation-step{step}.json';ev=json.loads(p.read_text())
                assert file_hash(p)==result['evaluations'][str(step)]
                assert file_hash(root/f'model-step{step}.pt')==ev['checkpoint_sha256']
                assert ev['step']==step and not ev['smoke_only']
                evaluations[step].setdefault(arm,[]).append(ev['evaluation']);hashes[str(p)]=file_hash(p)
            protocols[arm].append(protocol)
    for left,right in zip(protocols['difference'],protocols['raw']):
        for key in ('initial_hashes','batch_hash','lr','batch_size','microbatch','objective','source_sha256','test_selection_sha256','options'):
            assert left[key]==right[key],key
    return protocols,evaluations,hashes


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--source',type=Path,default=ROOT/'runs/layerwise-gpu-inputs-20261113')
    p.add_argument('--test-cache',type=Path,default=ROOT/'runs/difference-idm-ablation-inputs-20261113/test.pt')
    p.add_argument('--selection',type=Path,default=ROOT/'research/results/difference-idm-ablation-20261113/test-selection.json')
    p.add_argument('--output',type=Path,default=ROOT/'research/results/difference-idm-ablation-20261113')
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if (a.output/'results.json').exists():raise FileExistsError('Summary already exists; test is not automatically rescored')
    # Fail before opening held-out features unless ALL six runs completed as planned.
    protocols,evaluations,hashes=verify_runs(a.runs)
    plan=json.loads(a.selection.read_text());selection_hash=file_hash(a.selection)
    for pp in protocols.values():
        for protocol in pp:assert protocol['test_selection_sha256']==selection_hash
    groups={}
    for path in plan['manifest_paths'].values():
        for line in Path(path).read_text().splitlines():
            r=json.loads(line);groups[str(r['episode_id'])]=r['split_group']
    torch.cuda.set_device(a.device);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    data,base,_,_=load_inputs(a.source);del base
    testmeta=json.loads(a.test_cache.with_suffix('.json').read_text())
    assert file_hash(a.test_cache)==testmeta['cache_sha256'] and testmeta['selection_sha256']==selection_hash
    for pp in protocols.values():
        for protocol in pp:
            source_checkpoint=[v for k,v in protocol['source_sha256'].items() if k.endswith('/final.ckpt')]
            assert source_checkpoint==[testmeta['checkpoint_sha256']]
    test=torch.load(a.test_cache,weights_only=True,map_location='cpu')
    assert test['rows']==plan['selected']
    testresults={};replays=[]
    for arm in ('difference','raw'):
        testresults[arm]=[]
        for i,seed in enumerate((42,43,44)):
            root=a.runs/f'{arm}-{seed}'
            for step in (400,800,1600):
                model,protocol=load_trained_model(root/f'model-step{step}.pt',a.device)
                verify_checkpoint_protocol(protocol,protocols[arm][i])
                x,y=[data['observed96'][k][:8].to(a.device) for k in ('current','future')]
                actual=(model(x,y)[0]-y).square().flatten(1).mean(1).cpu().numpy()
                expected=np.array([r['mse'] for r in evaluations[step][arm][i]['observed96']['rows'][:8]])
                error=float(np.max(abs(actual-expected)));assert error<1e-6,error
                replays.append({'arm':arm,'seed':seed,'step':step,'mse_max_abs':error})
                if step==1600:
                    testpath=root/'terminal-test.json'
                    if testpath.exists():
                        scored=json.loads(testpath.read_text())
                        assert scored['checkpoint_sha256']==file_hash(root/'model-step1600.pt')
                        assert scored['cache_sha256']==testmeta['cache_sha256']
                    else:
                        scored={'evaluation':evaluate(model,{'test170':test},a.device),
                            'checkpoint_sha256':file_hash(root/'model-step1600.pt'),'cache_sha256':testmeta['cache_sha256']}
                        write(testpath,scored)
                    testresults[arm].append(scored['evaluation']);hashes[str(testpath)]=file_hash(testpath)
                del model;torch.cuda.empty_cache()
    report={'complete':True,'protocols':protocols,'source_sha256':hashes,'script_sha256':file_hash(Path(__file__)),
        'test_selection_sha256':selection_hash,'test_cache_meta':testmeta,'checkpoint_replay':replays,
        'milestones':{str(step):aggregate(ev,groups) for step,ev in evaluations.items()},
        'terminal_test':aggregate(testresults,groups),
        'scope':'Posterior reconstruction on fixed DINO features. Fixed terminal1600 test scoring after all runs; test never used for model/step selection. Same task families and camera; not task/domain transfer.',
        'interval_scope':'20000 paired resamples of seed-mean errors. Episode and video-shard cluster intervals conditional on shared pretrained weights; only seven validation/test shards, so cluster uncertainty is itself imprecise.'}
    a.output.mkdir(parents=True,exist_ok=True);write(a.output/'results.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True)
    for ax,split in zip(axes,('train','validation','observed96')):
        for arm in ('difference','raw'):
            vals=np.array([report['milestones'][str(s)][split]['models'][arm]['seed_mse'] for s in (400,800,1600)])
            line=ax.plot([400,800,1600],vals.mean(1),'-o',label=arm)[0]
            for i in range(3):ax.plot([400,800,1600],vals[:,i],alpha=.25,color=line.get_color())
        ax.set(title=split,xlabel='Optimizer updates',ylabel='Feature MSE');ax.grid(alpha=.2);ax.legend()
    fig.savefig(a.output/'learning-curves.png',dpi=180);plt.close(fig)
    for step,splits in report['milestones'].items():
        print('STEP',step,{s:{a:r['metrics']['mse'] for a,r in v['models'].items()} for s,v in splits.items()},flush=True)
    print('TEST',report['terminal_test']['test170']['difference_minus_raw'],flush=True)


if __name__=='__main__':main()
