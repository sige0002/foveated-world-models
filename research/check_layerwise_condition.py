"""CPU real-checkpoint smoke test; one disposable optimizer step, no fitted claim."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from research.train_early_condition import load_inputs, file_hash
from research.early_condition_adapter import frozen_hash
from research.layerwise_condition import install_layerwise_adapter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists(): parser.error('Output must be new')
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(42)
    data,model,hashes,_=load_inputs(ROOT)
    x,y,z=[data['train'][k][:1] for k in ('current','future','codes')]
    with torch.no_grad(): prior=model(x,torch.zeros_like(z))
    model=install_layerwise_adapter(model)
    frozen=frozen_hash(model)
    with torch.no_grad(): initial=model(x,z)
    assert torch.equal(prior,initial)
    optimizer=torch.optim.AdamW([model.action_in_proj.U.weight],lr=3e-4,weight_decay=.01)
    start=time.monotonic()
    prediction=model(x,z)
    loss=(prediction-y).square().mean()
    assert torch.isfinite(loss)
    loss.backward()
    gradients=model.action_in_proj.U.weight.grad.reshape(12,1024,32)
    norms=gradients.flatten(1).norm(dim=1)
    assert torch.isfinite(gradients).all() and torch.all(norms>0)
    cancellation=float(gradients.sum(0).norm()/norms.sum())
    assert all(p.grad is None for n,p in model.named_parameters() if n!='action_in_proj.U.weight')
    torch.nn.utils.clip_grad_norm_([model.action_in_proj.U.weight],1.,error_if_nonfinite=True)
    optimizer.step()
    with torch.no_grad():
        zero=model(x,torch.zeros_like(z))
        after=model(x,z)
    assert torch.equal(prior,zero)
    assert not torch.equal(after,initial)
    assert frozen_hash(model)==frozen
    for path in (Path(__file__),ROOT/'research/layerwise_condition.py',ROOT/'research/train_early_condition.py'):
        hashes[str(path)]=file_hash(path)
    result={'complete':True,'scope':'One disposable training-row step; no validation improvement or deployment claim.',
        'threads':2,'source_hashes':hashes,'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
        'initial_prior_exact':True,'zero_prior_after_update_exact':True,'frozen_weights_unchanged':True,
        'layer_gradient_norms':norms.tolist(),'gradient_sum_over_norm_sum':cancellation,
        'initial_train_mse':float(loss.detach()),'updated_train_mse':float((after-y).square().mean()),
        'step_and_checks_seconds':time.monotonic()-start}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'}))


if __name__=='__main__': main()
