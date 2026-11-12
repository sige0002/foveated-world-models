"""Fixed400 three-arm pilot. Default is real-data preflight only, never training."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import torch
from research.idm import train_local_repair as inherited
from research.train_cached_decoders import file_hash
from research.bounded_innovation import InnovationModel,fit_normalization,normalize,kl_summary


def write(path,value): path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def load(source):
    # The snapshotted loader validates source/cache/head/prior/query hashes,
    # selection rows, all split episodes and checkpoint identity. No source edits.
    inherited.ROOT=source
    data,delta,unused_head,decoder,hashes=inherited.load_inputs()
    del unused_head
    paths={'old':source/'research/idm/reference-query-full/train.pt',
           'added':source/'runs/reference-idm42-nested512-prior10000/additional_queries.pt',
           'validation':source/'research/idm/reference-query-full/validation.pt',
           'observed96':source/'research/idm/observed96-reference-queries2900/validation.pt'}
    q={k:torch.load(p,weights_only=True) for k,p in paths.items()}
    A={'train':torch.cat([q['old']['reference'],q['added']['reference']]),
       'validation':q['validation']['reference'],'observed96':q['observed96']['reference']}
    gen=torch.Generator().manual_seed(7001)
    T=torch.linalg.qr(torch.randn(1024,32,generator=gen,dtype=torch.float64),mode='reduced').Q.float()
    raw={arm:{s:torch.cat([A[s],d['codes'],delta[s] if arm=='future' else d['codes']@T.T],-1)
              for s,d in data.items()} for arm in ['future','recoding']}
    cal={arm:fit_normalization(v['train']) for arm,v in raw.items()}
    inputs={arm:{s:normalize(x,cal[arm]) for s,x in v.items()} for arm,v in raw.items()}
    for v in inputs.values():
        for x in v.values(): inherited.finite_tensor(x,'head input')
    return data,decoder,inputs,cal,T,hashes


def rng_stream(seed):
    return torch.Generator().manual_seed(seed+1000),torch.Generator().manual_seed(seed+2000)


@torch.no_grad()
def evaluate(model,data,inputs):
    model.eval();out={}
    for si,(split,d) in enumerate(data.items()):
        g=torch.Generator().manual_seed(30261113+si)
        rows=[];means=[]
        for start in range(0,len(d['rows']),8):
            sl=slice(start,start+8);n=len(d['current'][sl]);losses=[]
            for draw in range(4):
                eps=torch.randn(n,1,8,generator=g)
                pred,mu=model(d['current'][sl],d['codes'][sl],inputs[split][sl],eps,baseline=d['baseline'][sl])
                losses.append((pred-d['future'][sl]).square().flatten(1).mean(1))
            means.append(mu)
            values=torch.stack(losses)
            for i in range(n):rows.append({'source_row':d['rows'][start+i],
                'single_sample_risk_mc4':float(values[:,i].mean()),'draw_losses':values[:,i].tolist()})
        out[split]={'single_sample_risk_mc4':sum(r['single_sample_risk_mc4'] for r in rows)/len(rows),
                    'kl':kl_summary(torch.cat(means)),'rows':rows}
    return out


def preflight(model,data,inputs,old_predictions):
    model.eval();out={}
    with torch.no_grad():
        for split,d in data.items():
            maximum=0.;means=[]
            for start in range(0,len(d['rows']),8):
                sl=slice(start,start+8);mu=model.mean(inputs[split][sl]);means.append(mu)
                # Nonzero fixed epsilon: equality must hold for arbitrary new codes.
                eps=torch.ones_like(mu)*.37
                pred,_=model(d['current'][sl],d['codes'][sl],inputs[split][sl],eps,baseline=d['baseline'][sl])
                maximum=max(maximum,float((pred-old_predictions[split][sl]).abs().max()))
            if maximum!=0:raise RuntimeError(f'Initial prediction is not exact: {split} {maximum}')
            out[split]={'original_prediction_max_abs':maximum,'kl':kl_summary(torch.cat(means)),
                       'initial_mse':float((old_predictions[split]-d['future']).square().mean())}
    return out


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--arm',choices=['future','recoding','noise'],required=True)
    p.add_argument('--seed',type=int,choices=[42,43,44],default=42)
    p.add_argument('--threads',type=int,default=2);p.add_argument('--train',action='store_true')
    a=p.parse_args()
    if a.output.exists():p.error('Output must be new')
    torch.set_num_threads(a.threads);torch.set_float32_matmul_precision('highest');torch.manual_seed(42)
    data,decoder,inputs,cal,T,hashes=load(a.source.resolve())
    zhash={s:inherited.tensor_hash(d['codes']) for s,d in data.items()}
    fixedhash={s:{k:inherited.tensor_hash(d[k]) for k in ['current','future','baseline','codes']} for s,d in data.items()}
    for path in [Path(__file__),ROOT/'research/bounded_innovation.py',ROOT/'research/bounded-innovation-source-snapshot.json']:
        hashes[str(path)]=file_hash(path)
    snapshot=json.loads((ROOT/'research/bounded-innovation-source-snapshot.json').read_text())
    for name,digest in snapshot['files'].items():
        if file_hash(ROOT/name)!=digest: raise RuntimeError('Source snapshot changed: '+name)
    old={};decoder.eval()
    with torch.no_grad():
        for s,d in data.items():
            old[s]=torch.cat([decoder(d['current'][i:i+8],d['codes'][i:i+8],baseline=d['baseline'][i:i+8]) for i in range(0,len(d['rows']),8)])
    original_state={k:v.clone() for k,v in decoder.state_dict().items()}
    model=InnovationModel(decoder,a.arm,a.seed)
    # Verify all inherited tensors, including old action columns and bias.
    state=model.decoder.state_dict()
    for k,v in original_state.items():
        dest=k.replace('action_proj.','action_proj.original.') if k.startswith('action_proj.') else k
        assert torch.equal(v,state[dest]),k
    del original_state
    head_inputs=inputs['future' if a.arm=='future' else 'recoding']
    initial=preflight(model,data,head_inputs,old);del old
    a.output.mkdir(parents=True)
    bg,ng=rng_stream(a.seed)
    batches=torch.stack([torch.randint(512,(8,),generator=bg) for _ in range(400)])
    noises=torch.randn(400,8,1,8,generator=ng)
    protocol={'arm':a.arm,'seed':a.seed,'execute_training':a.train,'steps':400,'batch':8,
              'lr':3e-4,'weight_decay':{'ndim_ge_2':.01,'ndim_lt_2':0.},'clip':1.,'delta_nat':1.,'precision':'highest','threads':a.threads,
              'minibatch_seed':a.seed+1000,'noise_seed':a.seed+2000,
              'batch_hash':inherited.tensor_hash(batches),'noise_hash':inherited.tensor_hash(noises),
              'head_initial_hash':inherited.state_hash(model.head),'decoder_initial_hash':inherited.state_hash(model.decoder),
              'source_hashes':hashes,'snapshot_files':snapshot['files'],'z0_hashes':zhash,'fixed_input_hashes':fixedhash,
              'head_allocated_parameters':sum(v.numel() for v in model.head.parameters()),
              'head_active_parameters':sum(v.numel() for v in model.head.parameters() if v.requires_grad),
              'decoder_trainable_parameters':sum(v.numel() for v in model.decoder.parameters()),
              'evaluation':'4 independent draws: average individual losses, never predictions',
              'query_delta':'(Qxy-Qxx)/old_reference_scalar_calibration; new train-only coordinate normalization follows',
              'seed_scope':'Head initialization, minibatch and noise streams vary across seeds; decoder initial weights common',
              'conditioning':'C,Y are raw observations determining query inputs; target is phi(Y), not identity with decoder features',
              'scope':'Observed-set posterior reconstruction; fixed learned weights, independent single-sample additional channel bound only.'}
    torch.save({'normalization':cal,'orthonormal_embedding':T},a.output/'calibration.pt')
    protocol['calibration_sha256']=file_hash(a.output/'calibration.pt')
    write(a.output/'protocol.json',protocol);write(a.output/'preflight.json',initial)
    if not a.train:return
    active=[v for v in model.parameters() if v.requires_grad]
    opt=torch.optim.AdamW([{'params':[v for v in active if v.ndim>=2],'weight_decay':.01},
                           {'params':[v for v in active if v.ndim<2],'weight_decay':0.}],lr=3e-4)
    logs=[];start_time=time.monotonic()
    d=data['train'];frozen_head=inherited.state_hash(model.head)
    for step,(idx,eps) in enumerate(zip(batches,noises),1):
        opt.zero_grad(set_to_none=True)
        pred,mu=model(d['current'][idx],d['codes'][idx],head_inputs['train'][idx],eps,baseline=d['baseline'][idx])
        kl_summary(mu.detach())
        loss=(pred-d['future'][idx]).square().mean()
        if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss')
        loss.backward()
        if step in [1,2,10,100,400]:
            def norm(ps):return sum(float(v.grad.square().sum()) for v in ps if v.grad is not None)**.5
            logs.append({'step':step,'mse':float(loss.detach()),'head_gradient_norm':norm(model.head.parameters()),
                         'new_columns_gradient_norm':norm(model.decoder.action_proj.extra.parameters()),
                         'kl':kl_summary(mu.detach())})
            print(json.dumps({k:v for k,v in logs[-1].items() if k!='kl'}),flush=True)
        torch.nn.utils.clip_grad_norm_([v for v in model.parameters() if v.requires_grad],1.,error_if_nonfinite=True)
        opt.step()
    if a.arm=='noise':assert inherited.state_hash(model.head)==frozen_head
    for s,d in data.items():
        for k,h in fixedhash[s].items(): assert inherited.tensor_hash(d[k])==h
    result=evaluate(model,data,head_inputs)
    torch.save({'state_dict':model.state_dict(),'protocol':protocol,'steps':400},a.output/'model.pt')
    write(a.output/'results.json',{'evaluation':result,'logs':logs,'fixed_inputs_unchanged':True,
        'noise_head_frozen':a.arm=='noise','seconds':time.monotonic()-start_time,
        'checkpoint_sha256':file_hash(a.output/'model.pt'),'protocol_sha256':file_hash(a.output/'protocol.json')})

if __name__=='__main__':main()
