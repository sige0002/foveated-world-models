"""Same-weight attention arithmetic audit; optional isolated timing follows later."""
import argparse,copy,functools,json,sys,time,resource
from pathlib import Path
import torch
from torch import nn
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from research.train_cached_decoders import make_decoder


def disable_unused_weights(model):
    count=0
    for module in model.modules():
        if isinstance(module,nn.MultiheadAttention):
            original=module.forward
            def forward(*args,_original=original,**kwargs):
                kwargs['need_weights']=False
                return _original(*args,**kwargs)
            module.forward=forward;count+=1
    return count


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--benchmark',action='store_true');a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    checkpoint=a.source/'runs/reference-idm42-nested512-comparison400-20261113/prior_residual-seed42.pt'
    ck=torch.load(checkpoint,weights_only=True)
    baseline=make_decoder('prior_residual',ck['dim'],ck['layers'],768,32,16)
    baseline.load_state_dict(ck['state_dict'],strict=True)
    candidate=copy.deepcopy(baseline);count=disable_unused_weights(candidate)
    assert all(torch.equal(v,candidate.state_dict()[k]) for k,v in baseline.state_dict().items())
    cache=torch.load(a.source/'runs/reference-idm42-nested512-prior10000/train.pt',weights_only=True,mmap=True)
    data={k:cache[k][:2].clone() for k in ('current','future','codes','baseline')}
    def call(m):return m(data['current'],data['codes'],baseline=data['baseline'])
    with torch.inference_mode():
        baseline.eval();candidate.eval();y0=call(baseline);y1=call(candidate)
        inference_error=float((y0-y1).abs().max())
    outputs=[];grads=[]
    for m in (baseline,candidate):
        m.train();m.zero_grad(set_to_none=True);y=call(m);loss=(y-data['future']).square().mean();loss.backward()
        outputs.append(y.detach());grads.append({k:v.grad.detach().clone() for k,v in m.named_parameters() if v.grad is not None})
    assert grads[0].keys()==grads[1].keys()
    delta=sum((grads[0][k].double()-grads[1][k].double()).square().sum() for k in grads[0])
    scale=sum(v.double().square().sum() for v in grads[0].values())
    gradient_relative_l2=float((delta/scale).sqrt())
    training_error=float((outputs[0]-outputs[1]).abs().max())
    torch.testing.assert_close(outputs[0],outputs[1],rtol=1e-4,atol=1e-5)
    assert inference_error<1e-5 and gradient_relative_l2<1e-4
    report=dict(torch_version=torch.__version__,threads=2,batch=2,attention_modules=count,
                inference_max_abs=inference_error,training_max_abs=training_error,
                parameter_gradient_relative_l2=gradient_relative_l2,initial_weights_exact=True,
                benchmark_executed=a.benchmark,
                scope='Research correction only; no vision encoder or frozen prior time. Same weights, no fit. Float32 highest CPU. Numerical equivalence within declared tolerance, not bitwise equivalence.')
    if a.benchmark:
        timings=[]
        for m in (baseline,candidate):
            m.eval()
            with torch.inference_mode():
                for _ in range(3):call(m)
        for round in range(10):
            for name in (['baseline','candidate'] if round%2==0 else ['candidate','baseline']):
                m=baseline if name=='baseline' else candidate
                start=time.perf_counter();cpu=time.process_time()
                with torch.inference_mode():
                    for _ in range(10):call(m)
                timings.append(dict(round=round,variant=name,seconds=(time.perf_counter()-start)/10,cpu_seconds=(time.process_time()-cpu)/10))
        report['inference_timings']=timings
        training_times=[]
        for round in range(6):
            for name in (['baseline','candidate'] if round%2==0 else ['candidate','baseline']):
                m=baseline if name=='baseline' else candidate
                m.train();start=time.perf_counter();cpu=time.process_time()
                for _ in range(3):
                    m.zero_grad(set_to_none=True)
                    (call(m)-data['future']).square().mean().backward()
                training_times.append(dict(round=round,variant=name,seconds=(time.perf_counter()-start)/3,cpu_seconds=(time.process_time()-cpu)/3))
        report['forward_backward_timings']=training_times
        report['memory_note']='Process peak includes both resident models and correctness gradients; not per-variant memory.'
        report['process_peak_rss_kib']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(report)


if __name__=='__main__':main()
