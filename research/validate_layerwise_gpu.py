"""Rebuild missing research inputs and compare integrated decoders on CUDA.

Uses the local final10000 encoder, rather than the unavailable historical
encoder2900/head. All new arms share one train-only calibrated reference head.
Run prepare once, fit once per arm/seed, then summarize. Outputs must be new.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import statistics
import sys
import time

os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'
os.environ['LAM_ENABLE_MANUAL_WANDB'] = '0'
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'vendor/lawam')]
import numpy as np
import torch
import yaml
from research.train_early_condition import file_hash, load_inputs
from research.early_condition_adapter import install_adapter, frozen_hash, state_hash
from research.layerwise_condition import install_layerwise_adapter
from research.evaluate_prediction_metrics import metrics
from research.train_cached_decoders import make_decoder


def write(path, obj):
    path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')


def sync():
    torch.cuda.synchronize()


def prepare(a):
    from latent_action_model.core.lam_lightinng import VJEPA_LAM
    from vendor.dataset.robomind import RoboMINDDataset, robomind_collate, check_disjoint
    from research.idm.train_reference_heads import Model
    a.output.mkdir(parents=True, exist_ok=False)
    cfg = yaml.safe_load((a.run/'resolved_config.yaml').read_text())
    selection = json.loads((ROOT/'research/results/nested-train512-selection-20261113.json').read_text())
    extra = json.loads((ROOT/'research/results/research-unused-episode-selection-20261113.json').read_text())
    train = RoboMINDDataset(cfg['data']['train_manifest'], **selection['options'])
    val = RoboMINDDataset(cfg['data']['val_manifest'], **selection['options'])
    check_disjoint(train, val)
    trainrows = selection['retained'] + selection['added']
    valrows = []
    for index in np.linspace(0, len(val)-1, 64, dtype=int):
        ep, i, j, _ = val.pairs[int(index)]
        timestamps = val._arrays(ep)['timestamps']
        valrows.append(dict(index=int(index), episode=str(val.rows[ep]['episode_id']), dt=float(timestamps[j]-timestamps[i])))
    selections = {'train': trainrows, 'validation': valrows, 'observed96': extra['selected']}
    assert [len(selections[s]) for s in selections] == [512, 64, 96]
    assert not {r['episode'] for r in valrows} & {r['episode'] for r in extra['selected']}
    model = VJEPA_LAM(**cfg['model']).eval().requires_grad_(False)
    torch.set_float32_matmul_precision('highest')
    ck = torch.load(a.checkpoint, weights_only=True, mmap=True, map_location='cpu')
    assert ck['global_step'] == 10000
    model.load_state_dict(ck['state_dict'], strict=True)
    weightsdir = a.output/'runs/frozen-model-step10000'
    weightsdir.mkdir(parents=True)
    torch.save({'global_step': 10000, 'state_dict': {
        k: v for k, v in ck['state_dict'].items() if k.startswith('lam.decoder.')}}, weightsdir/'reference_weights.pt')
    (weightsdir/'resolved_config.yaml').write_text(yaml.safe_dump(cfg))
    del ck
    model.to(a.device)
    capture = {}
    original = model.lam.encoder.out_proj.forward
    def forward(q):
        capture['query'] = q.detach().clone()
        return original(q)
    model.lam.encoder.out_proj.forward = forward
    protocol = {'scope': 'NEW final10000 encoder + freshly trained reference head; not historical encoder2900 reproduction. True-future posterior reconstruction, reused evaluation episodes.',
        'checkpoint_sha256': file_hash(a.checkpoint), 'checkpoint': str(a.checkpoint),
        'config_sha256': file_hash(a.run/'resolved_config.yaml'),
        'script_sha256': file_hash(Path(__file__)), 'torch': str(torch.__version__),
        'device': torch.cuda.get_device_name(), 'precision': 'float32 highest, no autocast',
        'selections': selections, 'head_seed': 42, 'head_steps': 2000, 'head_training_rows': 128,
        'head_lr': 3e-4, 'head_beta': 5e-5,
        'manifest_sha256': {s:file_hash(Path(cfg['data'][s+'_manifest'])) for s in ('train','val')}}
    write(a.output/'prepare_protocol.json', protocol)
    data = {}
    for split, rows in selections.items():
        ds = train if split == 'train' else val
        chunks = []
        with torch.no_grad():
            for n, row in enumerate(rows):
                sample = ds[row['index']]
                assert str(sample['trajectory_id']) == row['episode']
                assert abs(sample['dt']-row['dt']) < 1e-6
                batch = model.transfer_batch_to_device(robomind_collate([sample], max_state_dim=32), torch.device(a.device), 0)
                out = model.lam._run(batch['videos'], batch['states'], batch['dec_videos'],
                    state_mask=batch['state_mask'], embodiment_ids=batch['embodiment_ids'],
                    vq_training=False, predict_future_frame=False)
                h, y, native = out[1], out[2], out[7]
                q = capture['query'].clone()
                model.lam.encoder(torch.cat((h,h),1))
                reference = capture['query']
                chunks.append({k:v.cpu() for k,v in dict(current=h, future=y, native_codes=native,
                    query=q, reference=reference, baseline=model.lam.decoder(h,torch.zeros_like(native))).items()})
                if (n+1)%32 == 0:
                    print('EXTRACT', split, n+1, '/', len(rows), flush=True)
        d = {k:torch.cat([c[k] for c in chunks]) for k in chunks[0]}
        d['rows'] = rows
        assert all(torch.isfinite(v).all() for v in d.values() if isinstance(v,torch.Tensor))
        torch.save(d, a.output/(split+'-features.pt'))
        data[split] = d
    del model
    torch.cuda.empty_cache()
    # Identical reference-head objective to the research pilot, trained only on
    # the retained128 rows. No validation is used to select steps or calibration.
    scale = (data['train']['query'][:128]-data['train']['reference'][:128]).square().mean().sqrt().clamp_min(1e-3)
    torch.manual_seed(42)
    head = Model().to(a.device)
    opt = torch.optim.AdamW([
        {'params':[p for p in head.parameters() if p.ndim>=2], 'weight_decay':.01},
        {'params':[p for p in head.parameters() if p.ndim<2], 'weight_decay':0.}], lr=3e-4)
    h,y=[data['train'][k][:128].to(a.device) for k in ('current','future')]
    representation = {s:((d['query']-d['reference'])/scale).to(a.device) for s,d in data.items()}
    rng = torch.Generator().manual_seed(1042)
    noise = torch.Generator().manual_seed(2042)
    for step in range(1,2001):
        idx=torch.randint(128,(8,),generator=rng)
        mu,lv=head.posterior(representation['train'][idx])
        eps=torch.randn(mu.shape,generator=noise).to(a.device)
        pred=head.decoder(h[idx],mu+(.5*lv).exp()*eps)
        mse=(pred-y[idx]).square().mean()
        kl=.5*(mu.square()+lv.exp()-1-lv).sum(-1).mean()
        loss=mse+5e-5*kl
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(),1.,error_if_nonfinite=True);opt.step()
        if step%200==0: print('HEAD',step,float(mse.detach()),float(kl.detach()),flush=True)
    torch.save({'state_dict':{k:v.cpu() for k,v in head.state_dict().items()},'scale':scale,'seed':42,'steps':2000},a.output/'reference-head.pt')
    wh=file_hash(weightsdir/'reference_weights.pt')
    for split,d in data.items():
        with torch.no_grad():
            d['codes']=head.posterior(representation[split])[0].cpu()
        root=a.output/('runs/reference-idm42-observed96-prior10000' if split=='observed96' else 'runs/reference-idm42-nested512-prior10000')
        root.mkdir(parents=True,exist_ok=True)
        disk='validation' if split=='observed96' else split
        torch.save(d,root/(disk+'.pt'))
        mp=root/'cache_manifest.json'
        manifest=json.loads(mp.read_text()) if mp.exists() else {'cache_sha256':{},'prior_checkpoint_step':10000,'prior_weights_sha256':wh,'scope':protocol['scope']}
        manifest['cache_sha256'][disk]=file_hash(root/(disk+'.pt'))
        write(mp,manifest)
    write(a.output/'prepare_complete.json',{'complete':True,'protocol_sha256':file_hash(a.output/'prepare_protocol.json')})


@torch.no_grad()
def evaluate(model, data, constant, device, original=None):
    out={}
    for split,d in data.items():
        rows=[]
        for start in range(0,len(d['rows']),8):
            sl=slice(start,start+8)
            x,y,z=[d[k][sl].to(device) for k in ('current','future','codes')]
            b=d['baseline'][sl].to(device)
            if constant is not None:z=constant.expand_as(z)
            pred=model(x,z,baseline=b) if original is not None else model(x,z)
            vals=metrics(pred,y,x)
            wrong=d['codes'].roll(1,0)[sl].to(device) if constant is None else z
            swapped=model(x,wrong,baseline=b) if original is not None else model(x,wrong)
            vals['swapped_mse']=(swapped-y).square().flatten(1).mean(1)
            if original is None:
                zero=model(x,torch.zeros_like(z))
                vals['zero_vs_cached_max_abs']=(zero-b).abs().flatten(1).max(1).values
            for j in range(len(x)):
                rows.append({'source_row':d['rows'][start+j],**{k:float(v[j]) for k,v in vals.items()}})
        out[split]={'metrics':{k:float(np.mean([r[k] for r in rows])) for k in vals},'rows':rows}
    return out


def fit(a):
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'executed_source.py').write_bytes(Path(__file__).read_bytes())
    torch.manual_seed(a.seed)
    data,base,hashes,options=load_inputs(a.source)
    original=copy.deepcopy(base).to(a.device).requires_grad_(False)
    if a.arm=='external':model=make_decoder('prior_residual',128,2,768,32,16)
    elif a.arm in ('shared','continuation'):model=install_adapter(base)
    else:model=install_layerwise_adapter(base)
    continuation=a.arm=='continuation'
    if continuation:model.requires_grad_(True)
    model.to(a.device).eval()
    frozen=frozen_hash(model)
    constant=data['train']['codes'].mean(0,keepdim=True).to(a.device)
    constant=constant*(data['train']['codes'].square().sum(-1).mean().sqrt().to(a.device)/constant.norm())
    constant=constant if a.arm=='constant' else None
    g=torch.Generator().manual_seed(a.seed+1000)
    batches=torch.stack([torch.randint(512,(8,),generator=g) for _ in range(a.steps)])
    train={k:data['train'][k].to(a.device) for k in ('current','future','codes','baseline')}
    protocol={'arm':a.arm,'seed':a.seed,'steps':a.steps,'batch_size':8,'lr':3e-4,'weight_decay':.01,
        'source_hashes':hashes,'script_sha256':file_hash(Path(__file__)),
        'layerwise_sha256':file_hash(ROOT/'research/layerwise_condition.py'),
        'batch_hash':state_hash({'batches':batches}),'options':options,
        'precision':'float32 highest, no autocast','torch':str(torch.__version__),'device':torch.cuda.get_device_name(),
        'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'scope':'New final10000 encoder, shared freshly trained head42; decoder seeds only. Reused64/96 true-future posterior reconstruction.'}
    write(a.output/'protocol.json',protocol)
    with torch.no_grad():
        x=train['current'][:8];z=train['codes'][:8]
        prior=original(x,torch.zeros_like(z))
        initial=model(x,z,baseline=prior) if a.arm=='external' else model(x,z)
        initial_error=float((initial-prior).abs().max())
        assert initial_error<=1e-5
    opt=torch.optim.AdamW([
        {'params':[p for p in model.parameters() if p.requires_grad and p.ndim>=2],'weight_decay':.01},
        {'params':[p for p in model.parameters() if p.requires_grad and p.ndim<2],'weight_decay':0.}],lr=3e-4)
    sync();started=time.perf_counter();logs=[]
    for step,idx in enumerate(batches,1):
        code=train['codes'][idx] if constant is None else constant.expand_as(train['codes'][idx])
        pred=model(train['current'][idx],code,baseline=train['baseline'][idx]) if a.arm=='external' else model(train['current'][idx],code)
        loss=(pred-train['future'][idx]).square().mean()
        assert torch.isfinite(loss)
        opt.zero_grad(set_to_none=True);loss.backward()
        if step==1 and a.arm in ('layerwise','constant'):
            grad=model.action_in_proj.U.weight.grad.reshape(12,1024,32)
            norms=grad.flatten(1).norm(dim=1)
            assert torch.all(norms>0)
            write(a.output/'gradient_check.json',{'norms':norms.tolist(),'cancellation_ratio':float(grad.sum(0).norm()/norms.sum())})
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
        opt.step()
        if step in (1,10) or step%100==0 or step==a.steps:
            logs.append({'step':step,'mse':float(loss.detach()),'grad_norm':float(norm)})
            print(a.arm,a.seed,logs[-1],flush=True)
    sync();seconds=time.perf_counter()-started
    if not continuation and a.arm!='external':
        assert frozen_hash(model)==frozen
    ev=evaluate(model,data,constant,a.device,original if a.arm=='external' else None)
    anchor_error=None
    if a.arm!='external':
        with torch.no_grad():anchor_error=float((model(x,torch.zeros_like(z))-prior).abs().max())
        if not continuation:assert anchor_error<=1e-5
    # Save trainable state only; immutable base and source hashes are in protocol.
    state={k:v.detach().cpu() for k,v in model.named_parameters() if v.requires_grad}
    torch.save({'trainable_state':state,'protocol':protocol,'constant':constant.cpu() if constant is not None else None},a.output/'model.pt')
    write(a.output/'results.json',{'complete':True,'evaluation':ev,'logs':logs,'training_seconds':seconds,
        'initial_prior_max_abs':initial_error,'zero_anchor_after_max_abs':anchor_error,
        'frozen_unchanged':not continuation if a.arm!='external' else None,'checkpoint_sha256':file_hash(a.output/'model.pt')})
    print('DONE',a.arm,a.seed,{s:v['metrics'] for s,v in ev.items()},flush=True)


@torch.inference_mode()
def summarize(a):
    a.output.mkdir(parents=True,exist_ok=False)
    data,base,hashes,_=load_inputs(a.source)
    base=base.to(a.device).eval().requires_grad_(False)
    arms=['layerwise','shared','constant','external','continuation']
    runs={arm:[] for arm in arms}
    for arm in arms:
        for seed in (42,43,44):
            path=a.experiments/f'{arm}-{seed}'
            r=json.loads((path/'results.json').read_text())
            protocol=json.loads((path/'protocol.json').read_text())
            assert r['complete'] and protocol['steps']==400
            assert file_hash(path/'model.pt')==r['checkpoint_sha256']
            runs[arm].append((r,protocol))
            hashes[str(path/'results.json')]=file_hash(path/'results.json')
    for i in range(3):
        assert len({runs[arm][i][1]['batch_hash'] for arm in arms})==1
    integrity={}
    for arm in ('layerwise','shared','constant'):
        integrity[arm]=[]
        for result,protocol in runs[arm]:
            assert result['frozen_unchanged']
            assert result['initial_prior_max_abs']<=1e-5
            assert result['zero_anchor_after_max_abs']<=1e-5
            maximum=max(row['zero_vs_cached_max_abs'] for split in result['evaluation'].values() for row in split['rows'])
            assert maximum<=1e-4
            integrity[arm].append({'seed':protocol['seed'],'initial_prior_max_abs':result['initial_prior_max_abs'],
                'zero_anchor_after_max_abs':result['zero_anchor_after_max_abs'],
                'zero_vs_cache_all_rows_max_abs':maximum,'frozen_unchanged':True})
    baseline={}
    for split,d in data.items():
        baseline[split]={'original_native':[],'zero_prior':[]}
        for start in range(0,len(d['rows']),8):
            sl=slice(start,start+8)
            x,y,z=[d[k][sl].to(a.device) for k in ('current','future','native_codes')]
            preds={'original_native':base(x,z),'zero_prior':base(x,torch.zeros_like(z))}
            for name,pred in preds.items():
                vals=metrics(pred,y,x)
                for j in range(len(x)):
                    baseline[split][name].append({k:float(v[j]) for k,v in vals.items()})
    report={'scope':runs['layerwise'][0][1]['scope'], 'source_sha256':hashes,'splits':{},'integrity':integrity,
        'prepare_protocol':json.loads((a.source/'prepare_protocol.json').read_text()),
        'script_sha256':file_hash(Path(__file__)),
        'bootstrap_scope':'20000 episode resamples of paired seed-mean errors; conditional on one shared trained head and these decoder seeds. Excludes adaptive evaluation/history uncertainty.',
        'protocol':{arm:[p for _,p in rs] for arm,rs in runs.items()}}
    for split in data:
        summary={}
        metric_names=list(baseline[split]['original_native'][0])
        perrow={}
        for name,rows in baseline[split].items():
            summary[name]={'metrics':{k:float(np.mean([r[k] for r in rows])) for k in metric_names}}
            perrow[name]=np.array([r['mse'] for r in rows])
        for arm,rs in runs.items():
            values=[r['evaluation'][split] for r,_ in rs]
            for v in values:
                assert [r['source_row'] for r in v['rows']]==data[split]['rows']
            summary[arm]={'metrics':{k:float(np.mean([v['metrics'][k] for v in values])) for k in metric_names},
                'swapped_mse':float(np.mean([v['metrics']['swapped_mse'] for v in values])),
                'seed_mse':[v['metrics']['mse'] for v in values],
                'seed_sd_mse':statistics.stdev(v['metrics']['mse'] for v in values),
                'training_seconds':[r['training_seconds'] for r,_ in rs],
                'trainable_parameters':rs[0][1]['trainable_parameters']}
            perrow[arm]=np.mean([[r['mse'] for r in v['rows']] for v in values],axis=0)
        rng=np.random.default_rng(20261113)
        draws=rng.integers(0,len(data[split]['rows']),(20000,len(data[split]['rows'])))
        comparisons={}
        for reference in ('original_native','zero_prior','shared','constant','external','continuation'):
            diff=perrow['layerwise']-perrow[reference]
            comparisons['layerwise_minus_'+reference]={'mse_difference':float(diff.mean()),
                'relative_reduction':float(1-perrow['layerwise'].mean()/perrow[reference].mean()),
                'conditional_paired_ci95':np.quantile(diff[draws].mean(1),[.025,.975]).tolist()}
        report['splits'][split]={'models':summary,'comparisons':comparisons}
    # Compare the full prediction path with B1/B8, including a freshly computed
    # prior for external correction. Run only after all fitting jobs have ended.
    models={'original_native':base}
    for arm in arms:
        # Earlier artifacts stored torch.__version__ as its TorchVersion string
        # subclass. Keep weights-only loading and allow just that known type.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            saved=torch.load(a.experiments/f'{arm}-42/model.pt',weights_only=True,map_location='cpu')
        if arm=='external':m=make_decoder('prior_residual',128,2,768,32,16)
        elif arm in ('shared','continuation'):m=install_adapter(copy.deepcopy(base))
        else:m=install_layerwise_adapter(copy.deepcopy(base))
        state=m.state_dict();state.update(saved['trainable_state']);m.load_state_dict(state,strict=True)
        models[arm]=m.to(a.device).eval().requires_grad_(False)
    constant=data['train']['codes'].mean(0,keepdim=True).to(a.device)
    constant*=data['train']['codes'].square().sum(-1).mean().sqrt().to(a.device)/constant.norm()
    timing={}
    for batch in (1,8):
        x=data['validation']['current'][:batch].to(a.device)
        z=data['validation']['codes'][:batch].to(a.device)
        native=data['validation']['native_codes'][:batch].to(a.device)
        def call(name):
            if name=='original_native':return base(x,native)
            if name=='external':return models[name](x,z,baseline=base(x,torch.zeros_like(z)))
            return models[name](x,constant.expand_as(z) if name=='constant' else z)
        for name in models:
            for _ in range(10):call(name)
        sync()
        measurements={name:[] for name in models}
        random=np.random.default_rng(20261113)
        for trial in range(40):
            for name in random.permutation(list(models)):
                sync();started=time.perf_counter();call(name);sync()
                measurements[name].append((time.perf_counter()-started)*1000)
        timing[str(batch)]={}
        for name,m in models.items():
            sync();torch.cuda.reset_peak_memory_stats();before=torch.cuda.memory_allocated()
            pred=call(name);sync()
            memory=torch.cuda.max_memory_allocated()-before
            del pred
            params=sum(p.numel()*p.element_size() for p in m.parameters())
            if name=='external':params+=sum(p.numel()*p.element_size() for p in base.parameters())
            timing[str(batch)][name]={'median_ms':statistics.median(measurements[name]),
                'trials_ms':measurements[name],'incremental_peak_activation_bytes':memory,
                'full_path_parameter_bytes':params}
    report['gpu_timing']={'device':torch.cuda.get_device_name(),'precision':'float32 highest',
        'warmup':10,'repetitions':40,'batches':timing,
        'scope':'Synchronized warm wall latency; full decoder path including original zero-code prior for external. Excludes DINO, IDM, host transfer, loading. Models resident together; incremental activation allocation and parameter bytes reported separately.'}
    report['complete']=True
    write(a.output/'results.json',report)
    print(json.dumps({s:{n:m['metrics']['mse'] for n,m in v['models'].items()} for s,v in report['splits'].items()},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode',choices=['prepare','fit','summarize'])
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--source',type=Path)
    p.add_argument('--experiments',type=Path)
    p.add_argument('--run',type=Path,default=Path('/data2/yuki_sadasue/runs/foveated-world-models/stage1-robomind-main'))
    p.add_argument('--checkpoint',type=Path,default=Path('/data2/yuki_sadasue/runs/foveated-world-models/stage1-robomind-main/checkpoints/final.ckpt'))
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--arm',choices=['layerwise','shared','constant','external','continuation'],default='layerwise')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--steps',type=int,default=400)
    a=p.parse_args()
    if not a.device.startswith('cuda') or not torch.cuda.is_available():p.error('CUDA required')
    if a.steps<1:p.error('steps must be positive')
    torch.cuda.set_device(a.device)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    torch.backends.cudnn.allow_tf32=False
    if a.mode=='prepare':prepare(a)
    else:
        if a.source is None:p.error('--source required for fit')
        if a.mode=='fit':fit(a)
        else:
            if a.experiments is None:p.error('--experiments required for summarize')
            summarize(a)


if __name__=='__main__':main()
