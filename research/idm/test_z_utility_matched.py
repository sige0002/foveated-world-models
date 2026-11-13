import copy
import torch
from torch import nn
from research.idm.test_difference_idm import fixture
from research.z_utility_matched import MatchedModel,compact_state,load_state
from research.evaluate_z_utility_matched import evaluate,code_geometry


def test_learned_constant_is_future_independent_and_encoder_is_frozen():
    base,x,y=fixture('cpu')
    model=MatchedModel(base.idm,base.decoder,constant=True)
    a,z=model(x,y);b,w=model(x,y.flip(0))
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(z,w,rtol=0,atol=0)
    torch.testing.assert_close(z[0],z[1],rtol=0,atol=0)
    a.square().mean().backward()
    assert all(p.grad is None for p in model.idm.encoder.parameters())
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.idm.head.parameters())
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.decoder.parameters())
    other=copy.deepcopy(model)
    with torch.no_grad():
        for p in other.idm.head.parameters():p.add_(1)
    state=compact_state(model)
    assert not any(k.startswith('idm.encoder.') for k in state)
    load_state(other,state)
    torch.testing.assert_close(other(x,y)[0],a,rtol=0,atol=0)


def test_external_prior_stays_frozen_and_zero_correction_is_prior():
    base,x,y=fixture('cpu')
    class Correction(nn.Module):
        def __init__(self):super().__init__();self.scale=nn.Parameter(torch.zeros(()))
        def forward(self,x,z,baseline):return baseline+self.scale*z.mean()*x
    model=MatchedModel(base.idm,Correction(),prior=base.decoder)
    out,z=model(x,y)
    with torch.no_grad():expected=model.prior(x,torch.zeros_like(z))
    torch.testing.assert_close(out,expected,rtol=0,atol=0)
    out.square().mean().backward()
    assert all(p.grad is None and not p.requires_grad for p in model.prior.parameters())
    assert model.decoder.scale.grad is not None
    assert not any(k.startswith('prior.') for k in compact_state(model))


def test_constant_diagnostic_reports_zero_output_change_and_variance():
    base,x,y=fixture('cpu');model=MatchedModel(base.idm,base.decoder,constant=True)
    d={'current':x,'future':y,'rows':[{'episode':str(i),'dt':1.2} for i in range(len(x))]}
    r=evaluate(model,{'train':d,'validation':d},'cpu',{str(i):str(i) for i in range(len(x))})
    for split in r.values():
        assert split['metrics']['null_output_delta_mse']==0
        assert split['metrics']['swapped_output_delta_mse']==0
        assert split['metrics']['null_mse']==split['metrics']['mse']
        assert split['code_geometry']['variance_sum']==0
        assert split['code_geometry']['effective_rank']==0
