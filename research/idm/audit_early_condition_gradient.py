"""Read-only derivative of the frozen original decoder at its zero-code condition."""
import hashlib,json,sys
from pathlib import Path
import torch,yaml
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from research.train_cached_decoders import file_hash


def main():
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest')
    cache=ROOT/'runs/reference-idm42-nested512-prior10000'
    meta=json.loads((cache/'cache_manifest.json').read_text())
    assert file_hash(cache/'train.pt')==meta['cache_sha256']['train']
    data=torch.load(cache/'train.pt',weights_only=True,mmap=True)
    source=ROOT/'runs/frozen-model-step10000'
    assert file_hash(source/'reference_weights.pt')==meta['prior_weights_sha256']
    config=yaml.safe_load((source/'resolved_config.yaml').read_text())['model']
    weights=torch.load(source/'reference_weights.pt',weights_only=True,mmap=True)
    assert weights['global_step']==10000
    model=LAMDecoder_v2(context_dim=config['dim'],input_dim=768,code_dim=config['code_dim'],num_queries=config['num_queries'],
        num_layers=config['dec_layers'],num_heads=config['num_heads'],dropout=0.,grid_hw=(16,16),
        ffn_expansion_factor=config['ffn_expansion_factor'],last_ln=config.get('decoder_last_ln',True)).eval().requires_grad_(False)
    prefix='lam.decoder.'
    model.load_state_dict({k[len(prefix):]:v for k,v in weights['state_dict'].items() if k.startswith(prefix)},strict=True)
    del weights
    ids=torch.randint(512,(8,),generator=torch.Generator().manual_seed(1042))
    h,y,z,b=[data[k][ids] for k in ('current','future','codes','baseline')]
    mean=data['codes'].mean(0,keepdim=True);assert mean.norm()>0
    zero=torch.zeros_like(z)
    with torch.inference_mode():reference=model(h,zero)
    torch.testing.assert_close(reference,b,atol=1e-4,rtol=1e-4)
    conditions=[]
    def hook(module,args,result):
        leaf=result.detach().requires_grad_(True);conditions.append(leaf);return leaf
    handle=model.action_in_proj.register_forward_hook(hook)
    output=model(h,zero);loss=(output-y).square().mean();loss.backward();handle.remove()
    gradient=conditions[0].grad.reshape(8,-1).detach()
    grad_real=gradient.T@z.reshape(8,32)
    grad_constant=gradient.T@mean.expand_as(z).reshape(8,32)
    assert torch.isfinite(grad_real).all() and grad_real.norm()>0
    assert all(p.grad is None for p in model.parameters())
    old_weight=model.action_in_proj.weight.detach().double()
    basis,singular,_=torch.linalg.svd(old_weight,full_matrices=False)
    tol=max(old_weight.shape)*torch.finfo(torch.float64).eps*singular.max()
    rank=int((singular>tol).sum());basis=basis[:,:rank]
    def outside(g):
        g=g.double();return float((g-basis@(basis.T@g)).square().sum()/g.square().sum())
    tensors=ROOT/'research/idm/early-condition-gradient-tensors-20261113.pt'
    if tensors.exists():raise FileExistsError(tensors)
    torch.save(dict(indices=ids,condition_gradient=gradient,real_U_gradient=grad_real,constant_U_gradient=grad_constant),tensors)
    report=dict(no_updates=True,precision='highest CPU2/B8',indices=ids.tolist(),
                rows=[data['rows'][i] for i in ids],mse=float(loss.detach()),
                inference_vs_cached_max_abs=float((reference-b).abs().max()),
                autograd_vs_inference_max_abs=float((output.detach()-reference).abs().max()),
                condition_gradient_norm=float(gradient.norm()),real_U_gradient_norm=float(grad_real.norm()),
                constant_U_gradient_norm=float(grad_constant.norm()),
                train_mean_code_norm=float(mean.norm()),train_code_rms_norm=float(data['codes'].square().sum(-1).mean().sqrt()),
                old_projection_rank=rank,old_projection_rank_tolerance=float(tol),
                real_gradient_outside_old_subspace_fraction=outside(grad_real),
                constant_gradient_outside_old_subspace_fraction=outside(grad_constant),
                tensor_sha256=file_hash(tensors),source_cache_sha256=meta['cache_sha256']['train'],
                prior_weights_sha256=meta['prior_weights_sha256'],script_sha256=file_hash(Path(__file__)),
                scope='Analytic initial U gradient G_condition.T @ reference_z; no adapter update. Outside-subspace fraction concerns the initial gradient, not learned condition distribution. Future-conditioned codes and true future used for this training-path diagnostic.')
    path=ROOT/'research/idm/early-condition-gradient-audit-20261113.json'
    with path.open('x') as f:json.dump(report,f,indent=2);f.write('\n')
    print({k:v for k,v in report.items() if k!='rows'},flush=True)


if __name__=='__main__':main()
