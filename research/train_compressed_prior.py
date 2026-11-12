"""Three fixed400 current-only students; correction stays frozen in every arm."""
import argparse,hashlib,json,time
from pathlib import Path
import torch
from research.compressed_prior import CurrentPrior,load_inputs,objective,evaluate,file_hash


def state_hash(model):
    h=hashlib.sha256()
    for k,v in sorted(model.state_dict().items()):
        h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kind',choices=['prior_teacher','future','downstream_teacher'],required=True);a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(4);torch.set_float32_matmul_precision('highest')
    data,correction,hashes=load_inputs(a.source);frozen_hash=state_hash(correction)
    model=CurrentPrior(42);initial_hash=state_hash(model)
    params=list(model.parameters())
    opt=torch.optim.AdamW([dict(params=[v for v in params if v.ndim>=2],weight_decay=.01),dict(params=[v for v in params if v.ndim<2],weight_decay=0.)],lr=3e-4)
    source_files=[Path(__file__),Path(__file__).with_name('compressed_prior.py'),Path(__file__).with_name('prior-compression-source-snapshot.json')]
    protocol=dict(kind=a.kind,seed=42,steps=400,batch=8,threads=4,precision='highest CPU',lr=.0003,
                  source_sha256=hashes,implementation_sha256={str(f):file_hash(f) for f in source_files},
                  initial_student_sha256=initial_hash,frozen_correction_sha256=frozen_hash,
                  parameters_trainable=sum(v.numel() for v in params),
                  student_inputs='Current features only; zero code created internally. No true future, teacher prior or IDM code at inference.',
                  objectives={'prior_teacher':'MSE(g,b)','future':'MSE(g,Y)','downstream_teacher':'MSE(D(X,Z,g),D(X,Z,b))+.1*MSE(g,b)'},
                  budget='Same updates and batches, NOT equal FLOPs: downstream arm includes extra frozen correction forwards and backprop to input',
                  scope='Frozen reference42 posterior code conditions final correction; final error is posterior reconstruction. Reused64/96. Fixed400, no checkpoint selection. Single-seed pilot.')
    a.output.mkdir(parents=True);(a.output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    generator=torch.Generator().manual_seed(1042);bh=hashlib.sha256();history=[];started=time.perf_counter()
    for step in range(1,401):
        model.train();idx=torch.randint(512,(8,),generator=generator);bh.update(idx.numpy().tobytes())
        h,b,y,z=[data['train'][k][idx] for k in ('current','baseline','future','codes')]
        opt.zero_grad(set_to_none=True);loss=objective(model,correction,a.kind,h,b,y,z)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite loss')
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.,error_if_nonfinite=True);opt.step()
        if step%100==0:
            history.append(dict(step=step,objective=float(loss.detach()),gradient_norm=float(norm)))
            print(history[-1],flush=True)
    duration=time.perf_counter()-started
    assert frozen_hash==state_hash(correction)
    assert all(v.grad is None for v in correction.parameters())
    checkpoint=a.output/'prior.pt';torch.save(dict(state_dict=model.state_dict(),kind=a.kind,seed=42,steps=400),checkpoint)
    report=dict(protocol_sha256=file_hash(a.output/'protocol.json'),checkpoint_sha256=file_hash(checkpoint),
                batch_sha256=bh.hexdigest(),history=history,training_seconds=duration,frozen_correction_unchanged=True,
                evaluation={s:evaluate(model,correction,d) for s,d in data.items()})
    for f,digest in protocol['implementation_sha256'].items():assert file_hash(Path(f))==digest
    (a.output/'results.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print({s:v['metrics'] for s,v in report['evaluation'].items()},flush=True)


if __name__=='__main__':main()
