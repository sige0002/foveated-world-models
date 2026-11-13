"""Freeze a metadata-only held-out selection, then cache DINO features without scoring it."""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ['HF_HUB_OFFLINE']='1'
os.environ['TRANSFORMERS_OFFLINE']='1'
os.environ['LAM_ENABLE_MANUAL_WANDB']='0'
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import torch
import yaml
from research.train_early_condition import file_hash
from research.train_difference_idm import write,atomic_save
from vendor.dataset.robomind import RoboMINDDataset,robomind_collate,check_disjoint


def audit(run,output):
    cfg=yaml.safe_load((run/'resolved_config.yaml').read_text())
    options=json.loads((ROOT/'research/results/nested-train512-selection-20261113.json').read_text())['options']
    paths={s:Path(cfg['data'][s+'_manifest']) for s in ('train','val')}
    paths['test']=paths['val'].with_name('test.jsonl')
    datasets={s:RoboMINDDataset(p,**options) for s,p in paths.items()}
    check_disjoint(*datasets.values())
    identities={s:{str(Path(r['video_file']).resolve()) for r in ds.rows} for s,ds in datasets.items()}
    arrayids={s:{str((ds.path.parent/r['array_file']).resolve()) for r in ds.rows} for s,ds in datasets.items()}
    for a,b in (('train','val'),('train','test'),('val','test')):
        assert not identities[a]&identities[b]
        assert not arrayids[a]&arrayids[b]
    # One deterministic middle full-horizon pair per episode; no model scores.
    test=datasets['test'];by_episode={}
    for index,(ep,i,j,tail) in enumerate(test.pairs):
        if not tail:by_episode.setdefault(ep,[]).append(index)
    rows=[]
    for ep in sorted(by_episode):
        indices=by_episode[ep];index=indices[len(indices)//2]
        _,i,j,_=test.pairs[index];source=test.rows[ep];arrays=test._arrays(ep)
        rows.append({'index':index,'episode':str(source['episode_id']),
            'dt':float(arrays['timestamps'][j]-arrays['timestamps'][i]),
            'split_group':source['split_group'],'tasks':source['tasks'],
            'array_sha256':file_hash(test.path.parent/source['array_file'])})
    assert len(rows)==len(test.rows), 'Some episodes have no eligible full-horizon pair'
    video_hashes={s:{p:file_hash(Path(p)) for p in sorted(ids)} for s,ids in identities.items()}
    for a,b in (('train','val'),('train','test'),('val','test')):
        assert not set(video_hashes[a].values())&set(video_hashes[b].values())
    report={'selection_rule':'One middle non-tail eligible pair per test episode; no model-output selection',
        'config_sha256':file_hash(run/'resolved_config.yaml'),'options':options,
        'manifest_sha256':{s:file_hash(p) for s,p in paths.items()},
        'manifest_paths':{s:str(p) for s,p in paths.items()},
        'counts':{s:{'episodes':len(ds.rows),'groups':len(ds.groups),'pairs':len(ds)} for s,ds in datasets.items()},
        'video_sha256':video_hashes,'selected':rows,
        'disjoint_episode_group_video_path_array_path_video_content':True,
        'prior_usage_audit':'Base run config uses train and validation only. Repository search found no test-scoring experiment. Preprocessing split/loader verification exists; complete external usage history is not knowable.',
        'limitations':'Same six task families and camera/domain. Grouped by video shard, original recording IDs unavailable; not a held-out-task or camera-generalization test.',
        'evaluation_policy':'Only fixed terminal1600 checkpoints, after all six training runs finish. No test access in trainer.',
        'script_sha256':file_hash(Path(__file__))}
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():raise FileExistsError(output)
    write(output,report)
    print('FROZEN',report['counts'],'selected',len(rows),flush=True)


@torch.no_grad()
def extract(run,selection,checkpoint,output,device,source):
    from latent_action_model.core.lam_lightinng import VJEPA_LAM
    if output.exists():raise FileExistsError(output)
    cfg=yaml.safe_load((run/'resolved_config.yaml').read_text())
    plan=json.loads(selection.read_text())
    assert file_hash(run/'resolved_config.yaml')==plan['config_sha256']
    for s,path in plan['manifest_paths'].items():assert file_hash(Path(path))==plan['manifest_sha256'][s]
    base_split=json.loads((run/'split_verification.json').read_text())
    for ours,original in (('train','train'),('val','validation')):
        assert plan['manifest_sha256'][ours]==base_split[original]['sha256']
    torch.cuda.set_device(device);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32=False
    model=VJEPA_LAM(**cfg['model']).eval().requires_grad_(False)
    # VJEPA_LAM.__init__ sets matmul precision to medium; override AFTER construction,
    # exactly as validate_layerwise_gpu.prepare does for the existing feature cache.
    torch.set_float32_matmul_precision('highest')
    saved=torch.load(checkpoint,weights_only=True,mmap=True,map_location='cpu')
    assert saved['global_step']==10000
    model.load_state_dict(saved['state_dict'],strict=True);del saved
    model.to(device)
    def features(ds,row):
        sample=ds[row['index']]
        assert str(sample['trajectory_id'])==row['episode']
        assert abs(sample['dt']-row['dt'])<1e-6
        batch=model.transfer_batch_to_device(robomind_collate([sample],max_state_dim=32),torch.device(device),0)
        # Same four-frame shape/layer/normalization as the existing feature cache.
        feat=model.lam.vision_encoder.encode(torch.cat((batch['videos'],batch['dec_videos']),1),n=cfg['model']['latent_layer_to_use'])
        return feat[:,2:3].cpu(),feat[:,-1:].cpu()
    # Validate extraction against an existing training pair, without test scoring.
    train=RoboMINDDataset(plan['manifest_paths']['train'],**plan['options'])
    cached=torch.load(source/'train-features.pt',weights_only=True,map_location='cpu')
    c,f=features(train,cached['rows'][0])
    delta=max(float((c-cached['current'][:1]).abs().max()),float((f-cached['future'][:1]).abs().max()))
    assert delta<1e-5,delta
    del cached
    ds=RoboMINDDataset(plan['manifest_paths']['test'],**plan['options'])
    current=[];future=[]
    for n,row in enumerate(plan['selected'],1):
        ep=ds.pairs[row['index']][0]
        assert file_hash(ds.path.parent/ds.rows[ep]['array_file'])==row['array_sha256']
        c,f=features(ds,row);current.append(c);future.append(f)
        if n%20==0:print('CACHE TEST',n,'/',len(plan['selected']),flush=True)
    d={'current':torch.cat(current),'future':torch.cat(future),'rows':plan['selected']}
    assert torch.isfinite(d['current']).all() and torch.isfinite(d['future']).all()
    output.parent.mkdir(parents=True,exist_ok=True);atomic_save(output,d)
    write(output.with_suffix('.json'),{'selection_sha256':file_hash(selection),'checkpoint_sha256':file_hash(checkpoint),
        'cache_sha256':file_hash(output),'train_extraction_max_abs':delta,'script_sha256':file_hash(Path(__file__)),
        'precision':'float32 highest; cudnn.allow_tf32=False; no autocast',
        'base_run_manifest_hashes_match':True,'base_split_verification_sha256':file_hash(run/'split_verification.json'),
        'scope':'DINO features only; no test prediction metrics or checkpoint selection'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['audit','extract'])
    p.add_argument('--run',type=Path,default=Path('/data2/yuki_sadasue/runs/foveated-world-models/stage1-robomind-main'))
    p.add_argument('--selection',type=Path,default=ROOT/'research/results/difference-idm-ablation-20261113/test-selection.json')
    p.add_argument('--output',type=Path,default=ROOT/'runs/difference-idm-ablation-inputs-20261113/test.pt')
    p.add_argument('--source',type=Path,default=ROOT/'runs/layerwise-gpu-inputs-20261113')
    p.add_argument('--checkpoint',type=Path,default=Path('/data2/yuki_sadasue/runs/foveated-world-models/stage1-robomind-main/checkpoints/final.ckpt'))
    p.add_argument('--device',default='cuda:0');a=p.parse_args()
    if a.command=='audit':audit(a.run,a.selection)
    else:extract(a.run,a.selection,a.checkpoint,a.output,a.device,a.source)
