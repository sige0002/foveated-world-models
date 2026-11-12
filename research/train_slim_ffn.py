"""Fixed-budget head-FFN comparison; full output dimension and input paths retained."""
import argparse,hashlib,json,sys,time
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.slim_ffn_decoder import make_slim_decoder
from research.train_cached_decoders import file_hash


@torch.inference_mode()
def evaluate(model,data):
    model.eval();rows=[]
    for start in range(0,len(data['rows']),8):
        h,y,z,b=[data[k][start:start+8] for k in ('current','future','codes','baseline')]
        error=(model(h,z,baseline=b)-y).square().flatten(1).mean(1)
        rows += [dict(data['rows'][start+i],mse=float(e)) for i,e in enumerate(error)]
    return dict(mse=sum(r['mse'] for r in rows)/len(rows),rows=rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--expansion',type=int,choices=[2,4],required=True);p.add_argument('--seed',type=int,choices=[42,43,44],required=True)
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    base=a.source/'runs/reference-idm42-nested512-prior10000'
    paths={'train':base/'train.pt','dev64':base/'validation.pt','observed96':a.source/'runs/reference-idm42-observed96-prior10000/validation.pt'}
    data={};hashes={};manifests={}
    for s,path in paths.items():
        m=json.loads((path.parent/'cache_manifest.json').read_text());manifests[s]=m
        hashes[s]=file_hash(path);assert hashes[s]==m['cache_sha256'][path.stem]
        data[s]=torch.load(path,weights_only=True,mmap=True)
    episodes=[{r['episode'] for r in d['rows']} for d in data.values()]
    assert all(episodes[i].isdisjoint(episodes[j]) for i in range(3) for j in range(i+1,3))
    for m in manifests.values():
        for k in ('prior_checkpoint_sha256','prior_weights_sha256'):assert m[k]==manifests['train'][k]
    model=make_slim_decoder(a.expansion,a.seed)
    groups=[dict(params=[v for v in model.parameters() if v.ndim>=2],weight_decay=.01),dict(params=[v for v in model.parameters() if v.ndim<2],weight_decay=0.)]
    optimizer=torch.optim.AdamW(groups,lr=3e-4);rng=torch.Generator().manual_seed(a.seed+1000)
    source_files=[Path(__file__),ROOT/'research/slim_ffn_decoder.py',ROOT/'research/train_cached_decoders.py',ROOT/'research/ffn-source-snapshot.json']
    protocol=dict(expansion=a.expansion,seed=a.seed,steps=400,batch=8,threads=4,precision='highest CPU',lr=.0003,
                  source_sha256=hashes,implementation_sha256={str(f):file_hash(f) for f in source_files},
                  parameters=sum(v.numel() for v in model.parameters()),
                  objective='Full feature MSE; fixed reference42 posterior mean codes; no codec or flow; output768 unchanged',
                  scope='Same current-only prior and features, fresh correction400 updates. Three decoder optimization seeds, one frozen IDM. Reused64/96. Fixed endpoint, no selection.')
    a.output.mkdir(parents=True);(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    batches=hashlib.sha256();history=[];started=time.perf_counter()
    for step in range(1,401):
        model.train();idx=torch.randint(512,(8,),generator=rng);batches.update(idx.numpy().tobytes())
        h,y,z,b=[data['train'][k][idx] for k in ('current','future','codes','baseline')]
        optimizer.zero_grad(set_to_none=True);loss=(model(h,z,baseline=b)-y).square().mean()
        assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if step%100==0:
            history.append(dict(step=step,mse=float(loss.detach())));print(history[-1],flush=True)
    checkpoint=a.output/'decoder.pt';torch.save(dict(state_dict=model.state_dict(),expansion=a.expansion,seed=a.seed,steps=400),checkpoint)
    result=dict(protocol_sha256=file_hash(a.output/'protocol.json'),checkpoint_sha256=file_hash(checkpoint),batch_sha256=batches.hexdigest(),history=history,
                training_seconds=time.perf_counter()-started,evaluation={s:evaluate(model,d) for s,d in data.items()})
    for f,digest in protocol['implementation_sha256'].items():assert file_hash(Path(f))==digest
    (a.output/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print({s:v['mse'] for s,v in result['evaluation'].items()},flush=True)


if __name__=='__main__':main()
