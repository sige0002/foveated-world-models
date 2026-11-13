"""Routing, anchored behavior, optimization and checkpoint contract tests."""
import copy
import io
import pytest
import torch
from research.test_early_condition import LAMDecoder_v2
from research.early_condition_adapter import install_adapter, frozen_hash
from research.layerwise_condition import install_layerwise_adapter
from research.evaluate_prediction_metrics import metrics


def fixture(device='cpu'):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    m = LAMDecoder_v2(context_dim=8, input_dim=4, code_dim=3,
                      num_layers=2, num_heads=2, grid_hw=(2,2), dropout=0).eval()
    for b in m.dec_layers:
        torch.nn.init.normal_(b.adaLN_modulation[-1].weight, std=.1)
        torch.nn.init.normal_(b.adaLN_modulation[-1].bias, std=.1)
    return m.to(device), torch.randn(2,1,4,4,device=device), torch.randn(2,1,3,device=device)


DEVICES = ['cpu'] + (['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device', DEVICES)
def test_equivalence_shared_and_gradient_sum(device):
    base,x,z=fixture(device)
    shared=install_adapter(copy.deepcopy(base))
    rng=torch.get_rng_state().clone()
    layerwise=install_layerwise_adapter(copy.deepcopy(base))
    assert torch.equal(rng,torch.get_rng_state())
    assert sum(p.numel() for p in layerwise.parameters() if p.requires_grad)==48
    with torch.no_grad():
        assert torch.equal(base(x,torch.zeros_like(z)),layerwise(x,z))
        shared.action_in_proj.U.weight.normal_(std=.1)
        layerwise.action_in_proj.U.weight.copy_(shared.action_in_proj.U.weight.repeat(2,1))
    # Different projection matrix sizes may choose different GEMM kernels.
    torch.testing.assert_close(shared(x,z),layerwise(x,z),rtol=1e-5,atol=1e-6)
    shared(x,z).square().mean().backward()
    layerwise(x,z).square().mean().backward()
    torch.testing.assert_close(shared.action_in_proj.U.weight.grad,
        layerwise.action_in_proj.U.weight.grad.reshape(2,8,3).sum(0),rtol=1e-4,atol=1e-7)


@pytest.mark.parametrize('device', DEVICES)
def test_update_zero_anchor_and_roundtrip(device):
    base,x,z=fixture(device);model=install_layerwise_adapter(copy.deepcopy(base))
    frozen=frozen_hash(model)
    prior=model(x,torch.zeros_like(z)).detach()
    opt=torch.optim.AdamW([model.action_in_proj.U.weight],lr=.01)
    (model(x,z)-torch.randn_like(x)).square().mean().backward()
    grads=model.action_in_proj.U.weight.grad.reshape(2,-1)
    assert torch.all(grads.norm(dim=1)>0)
    assert all(p.grad is None for n,p in model.named_parameters() if n!='action_in_proj.U.weight')
    opt.step()
    assert frozen_hash(model)==frozen
    assert torch.equal(prior,model(x,torch.zeros_like(z)))
    assert not torch.equal(prior,model(x,z))
    buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
    restored=install_layerwise_adapter(base)
    restored.load_state_dict(torch.load(buf,weights_only=True),strict=True)
    assert torch.equal(model(x,z),restored(x,z))
    assert torch.equal(model(x,z),model(x.squeeze(1),z.squeeze(1)))
    with pytest.raises(ValueError): model(x,z.expand(2,2,3))
    with pytest.raises(ValueError): model(x[:,:,:3],z)


def test_metrics_known_errors_and_changed_region():
    current=torch.zeros(1,1,10,2);target=current.clone();target[:,:,9]=2
    pred=target.clone();pred[:,:,9]+=1
    result=metrics(pred,target,current)
    assert result['mse'].item()==pytest.approx(.1)
    assert result['mae'].item()==pytest.approx(.1)
    assert result['smooth_l1_beta_0_1'].item()==pytest.approx(.095)
    assert result['high_feature_change_decile_mse'].item()==pytest.approx(1)
    assert result['low_feature_change_half_mse'].item()==0
