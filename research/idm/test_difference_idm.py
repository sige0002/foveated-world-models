import copy
import io
import sys
from pathlib import Path

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT/'vendor/lawam')]
from latent_action_model.core.utils.lam_encoder import LAMEncoder
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from research.idm.difference_idm import DifferenceIDM, DifferenceWorldModel


def fixture(device):
    torch.manual_seed(42)
    torch.set_num_threads(2)
    encoder = LAMEncoder(context_dim=8, input_dim=4, num_layers=2,
                         num_heads=2, num_frames=2, grid_hw=(2,2),
                         code_dim=8, add_state=False, dropout=0.)
    head = nn.Sequential(nn.Linear(8,6),nn.GELU(),nn.Linear(6,3))
    decoder = LAMDecoder_v2(context_dim=8,input_dim=4,code_dim=3,
                           num_layers=2,num_heads=2,grid_hw=(2,2),dropout=0.)
    for block in decoder.dec_layers:
        nn.init.normal_(block.adaLN_modulation[-1].weight,std=.1)
        nn.init.normal_(block.adaLN_modulation[-1].bias,std=.1)
    model=DifferenceWorldModel(DifferenceIDM(encoder,head,.5),decoder).to(device).eval()
    return model,torch.randn(2,1,4,4,device=device),torch.randn(2,1,4,4,device=device)


DEVICES=['cpu']+(['cuda'] if torch.cuda.is_available() else [])


@pytest.mark.parametrize('device',DEVICES)
def test_both_branches_receive_gradient_and_all_components_update(device):
    model,x,y=fixture(device)
    z,q,r=model.idm(x,y,return_queries=True)
    q.retain_grad();r.retain_grad()
    pred=model.predict(x,z)
    loss=(pred-y).square().mean();loss.backward()
    assert q.grad.norm()>0 and r.grad.norm()>0
    torch.testing.assert_close(q.grad,-r.grad,rtol=0,atol=0)
    before={name:next(part.parameters()).detach().clone() for name,part in
            [('encoder',model.idm.encoder),('head',model.idm.head),('decoder',model.decoder)]}
    for part in (model.idm.encoder,model.idm.head,model.decoder):
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in part.parameters())
        assert sum(float(p.grad.square().sum()) for p in part.parameters())>0
    torch.optim.SGD(model.parameters(),lr=.01).step()
    for name,part in [('encoder',model.idm.encoder),('head',model.idm.head),('decoder',model.decoder)]:
        assert not torch.equal(before[name],next(part.parameters()))


@pytest.mark.parametrize('device',DEVICES)
def test_identical_pair_zero_difference_but_head_need_not_be_zero(device):
    model,x,_=fixture(device)
    z,q,r=model.idm(x,x,return_queries=True)
    torch.testing.assert_close(q,r,rtol=0,atol=0)
    torch.testing.assert_close(z,model.idm.head(torch.zeros_like(q)),rtol=0,atol=0)
    assert z.norm()>0
    model.zero_grad(set_to_none=True);z.square().mean().backward()
    for p in model.idm.encoder.parameters():
        torch.testing.assert_close(p.grad,torch.zeros_like(p.grad),rtol=0,atol=1e-6)


def test_fixed_encoder_control_and_checkpoint_roundtrip():
    model,x,y=fixture('cpu')
    model.idm.encoder.requires_grad_(False)
    model(x,y)[0].square().mean().backward()
    assert all(p.grad is None for p in model.idm.encoder.parameters())
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.idm.head.parameters())
    buf=io.BytesIO();torch.save(model.state_dict(),buf);buf.seek(0)
    restored=copy.deepcopy(model)
    restored.load_state_dict(torch.load(buf,weights_only=True),strict=True)
    torch.testing.assert_close(restored(x,y)[0],model(x,y)[0],rtol=0,atol=0)
    with pytest.raises(ValueError):model.idm(x,y[:,:,:3])
