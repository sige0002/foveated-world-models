"""Fixed-weight z interventions, output sensitivity and matched four-arm risks."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import numpy as np
import torch
from research.z_utility_matched import ARMS,OLD,SOURCE,CHECKPOINT,make_model,load_state,MatchedModel
from research.train_early_condition import file_hash,load_inputs
from research.train_difference_idm import write
from research.idm.load_difference_idm import load_trained_model
from research.summarize_idm_ablation import verify_checkpoint_protocol,paired_comparison
from research.evaluate_prediction_metrics import metrics

TEST=ROOT/'runs/difference-idm-ablation-inputs-20261113/test.pt'
SELECTION=ROOT/'research/results/difference-idm-ablation-20261113/test-selection.json'


def measurement_hash():
    source=Path(__file__).read_text();tree=ast.parse(source)
    names={'NativeModel','code_geometry','summarize_rows','evaluate','aggregate'}
    core='\n'.join(ast.get_source_segment(source,n) for n in tree.body
                   if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)
    return hashlib.sha256(core.encode()).hexdigest()


def compatible_cached_diagnostic(saved):
    if saved.get('measurement_sha256')==measurement_hash():return True
    if saved['script_sha256']==file_hash(Path(__file__)):return True
    # This revision changes replay tolerance/scheduling only, not measurements.
    return (saved['script_sha256']=='58946502e2ddbbcac10d298a8b49f0531cb50cd846f156c71aabc59b63b643b6'
            and measurement_hash()=='075326f7cb6bead750ff9311f9556d6cad4fb09c4e4b091b855efe0bffc1515c')


def verify_replayed_mse(actual,expected):
    actual=np.asarray(actual,dtype=np.float64);expected=np.asarray(expected,dtype=np.float64)
    np.testing.assert_allclose(actual,expected,rtol=1e-4,atol=1e-6)
    difference=np.abs(actual-expected)
    assert abs(float(actual.mean()-expected.mean()))<1e-6
    return {'max_abs':float(difference.max()),'max_relative':float((difference/np.maximum(np.abs(expected),1e-12)).max()),
            'mean_abs':float(difference.mean()),'mean_drift':float(actual.mean()-expected.mean()),'rtol':1e-4,'atol':1e-6}


class NativeModel(torch.nn.Module):
    """Original final10000 native encoder/VAE mean/decoder, entirely frozen."""
    prior=None

    def __init__(self):
        super().__init__()
        import yaml
        from latent_action_model.core.utils.lam_encoder import LAMEncoder
        from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
        from latent_action_model.core.vq import VAEQuantizer
        c=yaml.safe_load((SOURCE/'runs/frozen-model-step10000/resolved_config.yaml').read_text())['model']
        self.encoder=LAMEncoder(context_dim=c['dim'],input_dim=768,num_heads=c['num_heads'],num_queries=c['num_queries'],
            grid_hw=tuple(v//c['patch_size'] for v in c['image_hw']),ffn_expansion_factor=c['ffn_expansion_factor'],
            dropout=c['dropout'],code_dim=c['code_dim'],num_layers=c['enc_layers'],num_frames=c['num_frames'],
            modal_mask=c['enc_modal_mask'],add_state=False,max_state_dim=c['max_state_dim'],num_embodiments=c['num_embodiments'])
        self.vq=VAEQuantizer(code_dim=c['code_dim'],**c['vq_kwargs'])
        self.decoder=LAMDecoder_v2(context_dim=c['dim'],input_dim=768,code_dim=c['code_dim'],num_queries=c['num_queries'],
            num_layers=c['dec_layers'],num_heads=c['num_heads'],dropout=c['dropout'],grid_hw=tuple(v//c['patch_size'] for v in c['image_hw']),
            ffn_expansion_factor=c['ffn_expansion_factor'],last_ln=c.get('decoder_last_ln',True))
        saved=torch.load(CHECKPOINT,weights_only=True,map_location='cpu',mmap=True)
        assert saved['global_step']==10000
        for name,module in [('encoder',self.encoder),('vq',self.vq),('decoder',self.decoder)]:
            prefix='lam.'+name+'.'
            module.load_state_dict({k[len(prefix):]:v for k,v in saved['state_dict'].items() if k.startswith(prefix)},strict=True)
        self.eval().requires_grad_(False)

    def code(self,x,y):return self.vq._encode(self.encoder(torch.cat((x,y),1)))[0]
    def null_code(self,x):return self.code(x,x)
    def predict(self,x,z,baseline=None):return self.decoder(x,z)


def code_geometry(z,train_mean):
    z=z.double().flatten(1);center=z-z.mean(0)
    vals=torch.linalg.eigvalsh(center.T@center/len(z)).clamp_min(0)
    total=float(vals.sum());squares=float(vals.square().sum())
    return {'variance_sum':total,'effective_rank':total**2/squares if squares>1e-24 else 0.,
            'mean_squared_norm':float(z.square().sum(1).mean()),
            'mean_shift_from_train_squared':float((z.mean(0)-train_mean.double().flatten()).square().sum())}


def summarize_rows(rows):
    keys=[k for k,v in rows[0].items() if isinstance(v,(float,int)) and k not in ['motion_bin','neighbor_index']]
    return {k:float(np.mean([r[k] for r in rows])) for k in keys}


@torch.no_grad()
def evaluate(model,data,device,groups):
    model.eval();codes={}
    for split,d in data.items():
        codes[split]=torch.cat([model.code(d['current'][i:i+8].to(device),d['future'][i:i+8].to(device)).cpu()
                               for i in range(0,len(d['rows']),8)])
    mean=codes['train'].mean(0,keepdim=True)
    trainchange=(data['train']['future']-data['train']['current']).square().flatten(1).mean(1)
    thresholds=torch.quantile(trainchange,torch.tensor([1/3,2/3]))
    result={}
    for split,d in data.items():
        z=codes[split];rows=[]
        pooled=d['current'].mean((1,2)).double()
        distances=torch.cdist(pooled,pooled)
        episodes=[r['episode'] for r in d['rows']]
        for i,ep in enumerate(episodes):
            distances[i,torch.tensor([e==ep for e in episodes])]=float('inf')
        neighbor=distances.argmin(1)
        for i in range(0,len(z),8):
            sl=slice(i,i+8);x,y=[d[k][sl].to(device) for k in ['current','future']]
            b=d['baseline'][sl].to(device) if model.prior is not None else None
            zi=z[sl].to(device);pred=model.predict(x,zi,b)
            vals=metrics(pred,y,x)
            vals['copy_mse']=(x-y).square().flatten(1).mean(1)
            vals['prediction_change_mse']=(pred-x).square().flatten(1).mean(1)
            interventions={'null':model.null_code(x),'mean':mean.expand_as(z[sl]).to(device),
                           'zero':torch.zeros_like(zi),'swapped':z.roll(1,0)[sl].to(device),
                           'neighbor':z[neighbor[sl]].to(device)}
            for name,code in interventions.items():
                changed=model.predict(x,code,b)
                vals[name+'_mse']=(changed-y).square().flatten(1).mean(1)
                vals[name+'_output_delta_mse']=(changed-pred).square().flatten(1).mean(1)
                vals[name+'_code_delta_squared']=(code-zi).square().flatten(1).sum(1)
            bins=torch.bucketize(vals['copy_mse'].cpu(),thresholds)
            for j in range(len(x)):
                r=d['rows'][i+j]
                rows.append({'source_row':r,'group':groups[r['episode']],
                             'motion_bin':int(bins[j]),'neighbor_index':int(neighbor[i+j]),
                             'neighbor_current_distance':float(distances[i+j,neighbor[i+j]]),
                             **{k:float(v[j]) for k,v in vals.items()}})
        bybin={str(k):{'count':sum(r['motion_bin']==k for r in rows),'metrics':summarize_rows([r for r in rows if r['motion_bin']==k])}
               for k in range(3) if any(r['motion_bin']==k for r in rows)}
        bygroup={k:{'count':sum(r['group']==k for r in rows),'metrics':summarize_rows([r for r in rows if r['group']==k])}
                 for k in sorted({r['group'] for r in rows})}
        result[split]={'metrics':summarize_rows(rows),'rows':rows,'code_geometry':code_geometry(z,mean),
                       'train_change_tertiles':thresholds.tolist(),'by_change_bin':bybin,'by_video_group':bygroup,
                       'dt':{'min':min(r['dt'] for r in d['rows']),'mean':float(np.mean([r['dt'] for r in d['rows']])),
                             'max':max(r['dt'] for r in d['rows'])}}
        print('EVALUATED',split,{k:result[split]['metrics'][k] for k in ['mse','null_mse','mean_mse','copy_mse','null_output_delta_mse']},flush=True)
    return result


def aggregate(evaluations):
    report={}
    for split in ('train','validation','observed96','reused_test170'):
        arrays={};models={}
        first=evaluations[ARMS[0]][0][split]['rows']
        groups=[r['group'] for r in first];source=[r['source_row'] for r in first]
        assert len({r['episode'] for r in source})==len(source)
        for arm in ARMS:
            rr=[e[split]['rows'] for e in evaluations[arm]]
            assert all([r['source_row'] for r in rows]==source for rows in rr)
            arrays[arm]=np.array([[r['mse'] for r in rows] for rows in rr])
            models[arm]={'seed_mse':arrays[arm].mean(1).tolist(),
                'metrics':{k:float(np.mean([e[split]['metrics'][k] for e in evaluations[arm]])) for k in evaluations[arm][0][split]['metrics']},
                'geometry_per_seed':[e[split]['code_geometry'] for e in evaluations[arm]]}
        comparisons={}
        for a,b in [('direct-real','direct-constant'),('external-real','external-constant'),
                    ('external-real','direct-real'),('external-constant','direct-constant')]:
            comparisons[a+'_minus_'+b]=paired_comparison(arrays[a],arrays[b],groups)
        interventions={}
        for arm in ['direct-real','external-real']:
            interventions[arm]={}
            for name in ['null','mean','zero','swapped','neighbor']:
                altered=np.array([[r[name+'_mse'] for r in e[split]['rows']] for e in evaluations[arm]])
                interventions[arm][name+'_minus_real']=paired_comparison(altered,arrays[arm],groups)
        report[split]={'models':models,'comparisons':comparisons,'interventions':interventions}
    return report


@torch.no_grad()
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--device',default='cuda:0')
    p.add_argument('--worker-arms',nargs='+',choices=ARMS);a=p.parse_args()
    if (a.output/'results.json').exists():raise FileExistsError('Report exists')
    # All prespecified runs must complete before any diagnostic evaluation.
    for arm in ARMS[1:]:
        for seed in (42,43,44):
            r=json.loads((a.runs/f'{arm}-{seed}'/'results.json').read_text())
            assert r['complete'] and not r['smoke_only'] and r['history'][-1]['step']==400
    torch.cuda.set_device(a.device);torch.set_num_threads(4);torch.set_float32_matmul_precision('highest');torch.backends.cudnn.allow_tf32=False
    data,base,hashes,_=load_inputs(SOURCE)
    meta=json.loads(TEST.with_suffix('.json').read_text());assert file_hash(TEST)==meta['cache_sha256']
    assert meta['selection_sha256']==file_hash(SELECTION)
    test=torch.load(TEST,weights_only=True,map_location='cpu');plan=json.loads(SELECTION.read_text())
    assert test['rows']==plan['selected'];groups={}
    for path in plan['manifest_paths'].values():
        for line in Path(path).read_text().splitlines():
            r=json.loads(line);groups[str(r['episode_id'])]=r['split_group']
    base.to(a.device).eval();test['baseline']=torch.cat([base(test['current'][i:i+8].to(a.device),torch.zeros((len(test['current'][i:i+8]),1,32),device=a.device)).cpu() for i in range(0,len(test['rows']),8)])
    del base;torch.cuda.empty_cache();data['reused_test170']=test
    evaluations={arm:[] for arm in ARMS};records=[];costs={}
    for arm in (a.worker_arms or ARMS):
        costs[arm]=[]
        for seed in (42,43,44):
            out=a.runs/f'{arm}-{seed}';out.mkdir(exist_ok=True)
            checkpoint=(OLD/f'difference-{seed}'/'model-step400.pt') if arm=='direct-real' else out/'model.pt'
            digest=file_hash(checkpoint);ep=out/'diagnostics.json'
            if ep.exists():
                saved=json.loads(ep.read_text());assert saved['checkpoint_sha256']==digest and compatible_cached_diagnostic(saved)
            else:
                if arm=='direct-real':
                    # Re-run the initial-weight/data/prefix checks used for new arms.
                    _,check,provenance=make_model(arm,seed);del check
                    m,protocol=load_trained_model(checkpoint,a.device)
                    verify_checkpoint_protocol(protocol,json.loads((checkpoint.parent/'protocol.json').read_text()))
                    assert protocol['initial_hashes']==provenance['initial_backbone_hashes']
                    model=MatchedModel(m.idm,m.decoder)
                else:
                    _,model,provenance=make_model(arm,seed)
                    ck=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
                    protocol=json.loads((out/'protocol.json').read_text());verify_checkpoint_protocol(ck['protocol'],protocol)
                    assert ck['step']==400 and protocol['batch_hash']==provenance['batch_hash']
                    assert protocol['initial_predictor_hash']==provenance['initial_predictor_hash']
                    load_state(model,ck['model']);del ck;model.to(a.device)
                print('MODEL',arm,seed,flush=True)
                ev=evaluate(model,data,a.device,groups)
                if arm=='direct-real':
                    previous=json.loads((checkpoint.parent/'evaluation-step400.json').read_text())
                    for split in ('train','validation','observed96'):
                        verify_replayed_mse([r['mse'] for r in ev[split]['rows']],
                                            [r['mse'] for r in previous['evaluation'][split]['rows']])
                saved={'arm':arm,'seed':seed,'checkpoint_sha256':digest,'script_sha256':file_hash(Path(__file__)),
                       'measurement_sha256':measurement_hash(),
                       'provenance':provenance,'evaluation':ev,'scope':'All evaluation sets reused; exploratory fixed-weight interventions'}
                write(ep,saved);del model;torch.cuda.empty_cache()
            replay={}
            if arm=='direct-real':
                previous=json.loads((checkpoint.parent/'evaluation-step400.json').read_text())
                for split in ('train','validation','observed96'):
                    replay[split]=verify_replayed_mse([r['mse'] for r in saved['evaluation'][split]['rows']],
                                                     [r['mse'] for r in previous['evaluation'][split]['rows']])
            evaluations[arm].append(saved['evaluation']);records.append({'arm':arm,'seed':seed,'path':str(ep),'sha256':file_hash(ep),'replay':replay})
            if arm=='direct-real':
                rr=json.loads((checkpoint.parent/'results.json').read_text());pr=json.loads((checkpoint.parent/'protocol.json').read_text())
                secs=next(r['training_seconds'] for r in rr['history'] if r['step']==400)
            else:
                rr=json.loads((out/'results.json').read_text());pr=json.loads((out/'protocol.json').read_text());secs=rr['training_seconds']
            costs[arm].append({'seed':seed,'training_seconds_400':secs,'peak_allocated_bytes':rr['peak_allocated_bytes'],'trainable_parameters':pr['trainable_parameters']})
    if a.worker_arms:
        print('WORKER COMPLETE',a.worker_arms,flush=True);return
    native_path=a.runs/'original-native.json'
    native_hash=file_hash(CHECKPOINT)
    assert native_hash==meta['checkpoint_sha256']
    if native_path.exists():
        native=json.loads(native_path.read_text())
        assert native['checkpoint_sha256']==native_hash and compatible_cached_diagnostic(native)
    else:
        model=NativeModel().to(a.device)
        checks={}
        for split in ['train','validation','observed96']:
            x,y=[data[split][k][:8].to(a.device) for k in ['current','future']]
            if 'native_codes' in data[split]:
                error=float((model.code(x,y).cpu()-data[split]['native_codes'][:8]).abs().max())
                assert error<1e-3,error;checks[split]=error
        print('MODEL original-native',flush=True)
        native={'checkpoint_sha256':native_hash,'script_sha256':file_hash(Path(__file__)),
                'native_cache_max_abs':checks,'evaluation':evaluate(model,data,a.device,groups),
                'scope':'Local final10000 native posterior mean; not official distributed weights. null means C,C, not constant; mean and zero are constants.'}
        write(native_path,native);del model;torch.cuda.empty_cache()
    report={'complete':True,'scope':'Fixed400 matched four arms; reused test170, not an independent blind test. Shared pretrained weights, 3 seeds. Same steps, not FLOPs.',
            'splits':aggregate(evaluations),'costs':costs,'diagnostic_artifacts':records,
            'original_native':native,
            'input_hashes':{**hashes,str(TEST):meta['cache_sha256'],str(SELECTION):file_hash(SELECTION)},
            'script_sha256':file_hash(Path(__file__)),'plan_sha256':file_hash(ROOT/'docs/z-utility-matched-plan.md')}
    a.output.mkdir(parents=True,exist_ok=True)
    for rec in records:
        # Preserve compact, portable diagnostics, including per-example output distances.
        write(a.output/f"{rec['arm']}-{rec['seed']}.json",json.loads(Path(rec['path']).read_text()))
    write(a.output/'results.json',report)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    labels=['validation','observed96','reused_test170'];xx=np.arange(3)
    for i,arm in enumerate(ARMS):
        vals=np.array([report['splits'][s]['models'][arm]['seed_mse'] for s in labels])
        axes[0].errorbar(xx+(i-1.5)*.07,vals.mean(1),yerr=vals.std(1,ddof=1),marker='o',label=arm,capsize=3)
    for arm in ['direct-real','external-real']:
        vals=[report['splits'][s]['models'][arm]['metrics'] for s in labels]
        axes[1].plot(xx,[100*(v['null_mse']/v['mse']-1) for v in vals],'-o',label=arm)
    for ax in axes:ax.set_xticks(xx,labels);ax.grid(alpha=.2);ax.legend()
    axes[0].set_ylabel('Feature MSE (mean +/- seed SD)');axes[1].set_ylabel('MSE increase with constant z (%)')
    fig.savefig(a.output/'comparison.png',dpi=170);plt.close(fig)
    print('COMPLETE',json.dumps({s:{arm:v['metrics']['mse'] for arm,v in x['models'].items()} for s,x in report['splits'].items()}),flush=True)


if __name__=='__main__':main()
