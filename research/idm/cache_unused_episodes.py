"""Extract prespecified research-unused validation episodes, without fitting anything.

CPU-only, frozen step2900 query/head and step6501 original predictor. This is
research-unused validation, not an external test set or an executable prior z.
"""
import argparse,hashlib,json,os,sys,time
from pathlib import Path
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['LAM_ENABLE_MANUAL_WANDB']='0'
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import torch
from torch import nn
import yaml
from vendor.dataset.robomind import RoboMINDDataset,robomind_collate
from latent_action_model.core.lam_lightinng import VJEPA_LAM
from research.idm.train_reference_heads import representation
from research.cache_stage1_features import sha256

def state_hash(state,prefix):
    h=hashlib.sha256();count=0
    for key in sorted(state):
        if key.startswith(prefix):
            value=state[key].detach().cpu().contiguous();count+=1
            h.update(key.encode());h.update(str(value.dtype).encode());h.update(str(tuple(value.shape)).encode());h.update(value.numpy().tobytes())
    if not count:raise ValueError('Empty state prefix: '+prefix)
    return h.hexdigest()

@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selection',type=Path,default=ROOT/'research/results/research-unused-episode-selection-20261113.json')
    p.add_argument('--old-cache',type=Path,default=ROOT/'runs/frozen-idm-step2900')
    p.add_argument('--prior-cache',type=Path,default=ROOT/'runs/frozen-idm-step6501')
    p.add_argument('--head-manifest',type=Path,default=ROOT/'research/idm/reference-codes-seed42/cache_manifest.json')
    p.add_argument('--calibration',type=Path,default=ROOT/'research/idm/reference-query-full/train_calibration.pt')
    p.add_argument('--output',type=Path,default=ROOT/'runs/research-unused-96-prior6501')
    a=p.parse_args()
    if a.output.exists():p.error('Output must be new')
    torch.set_num_threads(4);torch.set_float32_matmul_precision('medium')
    selection=json.loads(a.selection.read_text());selection_hash=sha256(a.selection)
    hm=json.loads(a.head_manifest.read_text());export=hm['export']
    if (export['variant'],export['seed'],export['steps'])!=('reference',42,2000):raise ValueError('Unexpected frozen head')
    head_path=Path(export['head_checkpoint'])
    if sha256(head_path)!=export['head_checkpoint_sha256'] or sha256(a.calibration)!=export['calibration_sha256']:raise ValueError('Head/calibration hash mismatch')
    if sha256(selection['manifest'])!=selection['manifest_sha256']:raise ValueError('Dataset manifest hash mismatch')
    picked=selection['selected'];episodes={r['episode'] for r in picked};excluded=set(selection['excluded_episode_ids'])
    if len(picked)!=96 or len(episodes)!=96 or episodes&excluded:raise ValueError('Selection overlap/count failure')
    old_manifest=json.loads((a.old_cache/'cache_manifest.json').read_text())
    prior_manifest=json.loads((a.prior_cache/'cache_manifest.json').read_text())
    old_research=set();cache_checks={}
    for cache in (a.old_cache,a.prior_cache):
        manifest=json.loads((cache/'cache_manifest.json').read_text())
        for split in ('train','validation'):
            file=cache/f'{split}.pt';digest=sha256(file)
            if digest!=manifest['cache_sha256'][split]:raise ValueError('Source cache hash mismatch')
            data=torch.load(file,weights_only=True)
            old_research.update(r['episode'] for r in data['rows']);cache_checks[str(file)]=digest
    if old_research!=excluded or episodes&old_research:raise ValueError('Excluded episodes do not match research cache history')
    configs=[yaml.safe_load((cache/'resolved_config.yaml').read_text()) for cache in (a.old_cache,a.prior_cache)]
    if configs[0]['model']!=configs[1]['model']:raise ValueError('Model config mismatch')
    for cfg in configs:
        if cfg['model']['enc_add_state'] or cfg['model']['latent_layer_to_use']!=-2:raise ValueError('Unsupported encoder feature route')
        if Path(cfg['data']['val_manifest']).resolve()!=Path(selection['manifest']).resolve():raise ValueError('Selection manifest path mismatch')
    options=dict(image_hw=configs[0]['model']['image_hw'],max_state_dim=configs[0]['model']['max_state_dim'],
        num_embodiments=configs[0]['model']['num_embodiments'],horizon_seconds=configs[0]['data']['horizon_seconds'],
        tolerance_seconds=configs[0]['data']['tolerance_seconds'],require_observed_endpoints=True)
    dataset=RoboMINDDataset(selection['manifest'],**options)
    # Check all selection metadata before any images/errors are inspected.
    for row in picked:
        i,first,last,_=dataset.pairs[row['index']];arrays=dataset._arrays(i)
        dt=float(arrays['timestamps'][last]-arrays['timestamps'][first])
        if dataset.rows[i]['episode_id']!=row['episode'] or [first,last]!=row['endpoint_indices'] or abs(dt-row['dt'])>1e-9:raise ValueError('Selection pair metadata mismatch')
        if not arrays['original_valid'][[first,last]].all():raise ValueError('Unobserved endpoint')
    print('Selection/hash/episode/pair checks passed',flush=True)
    saved=[torch.load(cache/'reference_weights.pt',weights_only=True,mmap=True) for cache in (a.old_cache,a.prior_cache)]
    if [s['global_step'] for s in saved]!=[2900,6501]:raise ValueError('Unexpected checkpoint steps')
    visual_hashes=[state_hash(s['state_dict'],'lam.vision_encoder.') for s in saved]
    if visual_hashes[0]!=visual_hashes[1]:raise ValueError('Frozen visual weight mismatch')
    encoder_hash=state_hash(saved[0]['state_dict'],'lam.encoder.')
    if encoder_hash!=export['encoder_state_sha256']:raise ValueError('Original encoder hash mismatch')
    head_saved=torch.load(head_path,weights_only=True)
    if state_hash(head_saved['state_dict'],'head.')!=export['head_state_sha256']:raise ValueError('Head state mismatch')
    head=nn.Sequential(nn.Linear(1024,256),nn.GELU(),nn.Linear(256,64)).eval()
    head.load_state_dict({k[len('head.'):]:v for k,v in head_saved['state_dict'].items() if k.startswith('head.')})
    cal=torch.load(a.calibration,weights_only=True)
    models=[]
    for cfg,weights in zip(configs,saved):
        model=VJEPA_LAM(**cfg['model']).cpu().eval().requires_grad_(False)
        model.load_state_dict(weights['state_dict'],strict=True)
        if any(p.device.type!='cpu' for p in model.parameters()):raise ValueError('Non-CPU model')
        models.append(model)
    del saved,head_saved
    old,new=models;capture={};original=old.lam.encoder.out_proj.forward
    def capture_forward(query):capture['query']=query.detach().clone();return original(query)
    old.lam.encoder.out_proj.forward=capture_forward
    current=[];future=[];baseline=[];original_codes=[];queries=[];references=[];rows=[]
    started=time.perf_counter()
    for number,row in enumerate(picked):
        sample=dataset[row['index']]
        if str(sample['trajectory_id'])!=row['episode'] or abs(sample['dt']-row['dt'])>1e-9:raise ValueError('Decoded pair mismatch')
        raw_batch=robomind_collate([sample],max_state_dim=old.lam.encoder.max_state_dim)
        outputs=[]
        for model in models:
            batch=model.transfer_batch_to_device(raw_batch,torch.device('cpu'),0)
            outputs.append(model.lam._run(batch['videos'],batch['states'],batch['dec_videos'],
                state_mask=batch['state_mask'],embodiment_ids=batch['embodiment_ids'],vq_training=False,predict_future_frame=False))
            if model is old:true_query=capture['query'].clone()
        h,y,z=outputs[1][1],outputs[1][2],outputs[1][7]
        if not torch.equal(h,outputs[0][1]) or not torch.equal(y,outputs[0][2]):raise ValueError('Old/new visual features differ')
        old.lam.encoder(torch.cat((h,h),dim=1));ref_query=capture['query'].clone()
        pred=new.lam.decoder(h,z);prior=new.lam.decoder(h,torch.zeros_like(z))
        current.append(h.clone());future.append(y.clone());baseline.append(prior.clone());original_codes.append(z.clone())
        queries.append(true_query);references.append(ref_query)
        rows.append(dict(row,original6501_mse=float((pred-y).square().mean()),reference_mse=float((pred-y).square().mean()),
            baseline_mse=float((prior-y).square().mean()),reference_zero_z_mse=float((prior-y).square().mean())))
        if (number+1)%8==0:print('Extracted',number+1,'/96; visual features exactly equal',flush=True)
    old.lam.encoder.out_proj.forward=original
    q=dict(query=torch.cat(queries),reference=torch.cat(references))
    normalized=representation(q,cal,'reference');code_batches=[]
    # Original exported validation head used B64. Keep B64 chunks (last chunk B32).
    for start in range(0,len(rows),64):code_batches.append(head(normalized[start:start+64]).chunk(2,-1)[0])
    tensors=dict(current=torch.cat(current),future=torch.cat(future),codes=torch.cat(code_batches),baseline=torch.cat(baseline),original_codes=torch.cat(original_codes))
    if not all(torch.isfinite(v).all() for v in tensors.values()):raise ValueError('Nonfinite cached tensor')
    if sha256(a.selection)!=selection_hash or sha256(selection['manifest'])!=selection['manifest_sha256']:raise ValueError('Selection changed during extraction')
    a.output.mkdir(parents=True)
    torch.save(dict(**tensors,rows=rows),a.output/'validation.pt')
    report=dict(model_config=configs[0]['model'],checkpoint_step=2900,prior_checkpoint_step=6501,
        augmentation=False,cache_sha256={'validation':sha256(a.output/'validation.pt')},
        selection=selection['rule'],selection_sha256=selection_hash,selection_path=str(a.selection.resolve()),scope=selection['scope'],
        source_cache_sha256=cache_checks,manifests={'validation':{'path':selection['manifest'],'sha256':selection['manifest_sha256']}},
        visual_state_sha256=visual_hashes[0],visual_features_exactly_equal_all_pairs=True,encoder_state_sha256=encoder_hash,
        head_checkpoint=str(head_path),head_checkpoint_sha256=export['head_checkpoint_sha256'],head_state_sha256=export['head_state_sha256'],
        calibration_sha256=export['calibration_sha256'],head_manifest_sha256=sha256(a.head_manifest),
        source_checkpoint_manifest_sha256={str(a.old_cache):sha256(a.old_cache/'cache_manifest.json'),str(a.prior_cache):sha256(a.prior_cache/'cache_manifest.json')},
        conditioning_mean='Fixed step2900 reference-query Gaussian head mu using true future; no fitting/recalibration; not executable prior',
        baseline_source='Frozen original step6501 decoder(current,zero code)',original6501_source='Step6501 encoder+VAE posterior mean from actual pair, decoded by step6501 decoder',
        count=len(rows),episode_overlap_with_research_caches=0,precision='float32 medium matmul; CPU4; original models B1; adapted head B64/B32',
        extraction_seconds=time.perf_counter()-started,script_sha256=sha256(Path(__file__)),torch_version=torch.__version__)
    (a.output/'cache_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Completed',a.output,'seconds',report['extraction_seconds'],flush=True)
if __name__=='__main__':main()
