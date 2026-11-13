from pathlib import Path
import sys
import pytest
import torch
import yaml
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'vendor/lawam'))
from latent_action_model.core.utils.lam_decoder import StatePredictor
from latent_action_model.data_loader.collate import lam_collate
from latent_action_model.core.utils.utils import eef_reconstruction_loss


def predictor(scale=1):
    return StatePredictor(latent_dim=8,code_dim=4,num_embodiments=2,max_state_dim=32,output_scale=scale)


def test_delta_teacher_and_saturating_gripper_range():
    states=torch.zeros(2,16); states[0,[7,15]]=torch.tensor([-1.,1.]);states[1,[7,15]]=torch.tensor([1.,-1.])
    batch=lam_collate([dict(frames=torch.zeros(2,3,16,16,dtype=torch.uint8),proprio=states,embodiment_id=1)])
    target=batch['delta_proprio'];assert target[0,7]==2 and target[0,15]==-2
    g=predictor(2)
    with torch.no_grad():
        for p in g.parameters():p.zero_()
        g.state_decoder.layer2.b[1,7]=5
        g.state_decoder.layer2.b[1,15]=-5
    y=g(torch.zeros(1,1,4),batch['states'][:,0],batch['embodiment_ids'])
    assert y[0,7]>1.99 and y[0,15]<-1.99
    loss=eef_reconstruction_loss(y,target,batch['state_mask'][:,0],state_loss_type='l2')
    assert loss<1e-7
    loss.backward()
    assert torch.isfinite(g.state_decoder.layer2.b.grad).all()
    assert g.state_decoder.layer2.b.grad[1,7]<0
    assert g.state_decoder.layer2.b.grad[1,15]>0


def test_upstream_scale_and_state_dict_compatibility():
    old=predictor();new=predictor(2);new.load_state_dict(old.state_dict(),strict=True)
    z=torch.randn(2,1,4);s=torch.randn(2,32);ids=torch.ones(2,dtype=torch.long)
    torch.testing.assert_close(new(z,s,ids),2*old(z,s,ids),rtol=0,atol=0)
    assert old(z,s,ids).abs().max()<=1
    assert new(z,s,ids).abs().max()<=2


@pytest.mark.parametrize('scale',[0,-1,float('inf'),float('nan')])
def test_invalid_scale(scale):
    with pytest.raises(ValueError):predictor(scale)


def test_selected_configuration():
    cfg=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/stage1_robotwin_delta2.yaml').read_text())
    assert cfg['model']['state_output_scale']==2
    assert cfg['model']['warmup_steps']==1000


def test_half_delta_with_fourfold_l2_has_identical_gradients():
    # Current-state input is unchanged; only the target units differ.
    raw=torch.tensor([-.8,.2,1.5],dtype=torch.float64,requires_grad=True)
    delta=torch.tensor([-2.,.5,1.8],dtype=torch.float64)
    expanded=((2*raw.tanh()-delta)**2).mean()
    half_teacher=4*((raw.tanh()-delta/2)**2).mean()
    torch.testing.assert_close(expanded,half_teacher,rtol=0,atol=0)
    grad1=torch.autograd.grad(expanded,raw,retain_graph=True)[0]
    grad2=torch.autograd.grad(half_teacher,raw)[0]
    torch.testing.assert_close(grad1,grad2,rtol=0,atol=0)
