import copy
import torch
import pytest

from research.idm.test_difference_idm import fixture,DEVICES
from research.train_idm_ablation import (initialize_ablation,make_batches,optimizer_for,
    save_training,restore_training,evaluate)
from research.summarize_idm_ablation import paired_comparison,verify_runs
import numpy as np


@pytest.mark.parametrize('device',DEVICES)
def test_raw_calls_encoder_once_and_same_image_code_is_not_head_zero(device):
    model,x,y=fixture(device);model.idm.representation='raw'
    calls=[]
    handle=model.idm.encoder.register_forward_hook(lambda *args:calls.append(1))
    z,q,r=model.idm(x,y,return_queries=True)
    assert r is None and len(calls)==1
    torch.testing.assert_close(z,model.idm.head(q/model.idm.scale),rtol=0,atol=0)
    z.square().sum().backward()
    assert any(p.grad is not None and p.grad.norm()>0 for p in model.idm.encoder.parameters())
    same=model.idm(x,x)
    assert not torch.allclose(same,model.idm.head(torch.zeros_like(q)))
    handle.remove()
    ev=evaluate(model,{'toy':{'current':x.cpu(),'future':y.cpu(),'rows':[{},{}]}},device,2)
    with torch.no_grad():expected=float((model.predict(x,same)-y).square().mean())
    assert ev['toy']['metrics']['same_image_code_mse']==pytest.approx(expected,rel=1e-6)


def test_head_initialization_and_calibration_are_paired_and_train_only():
    model,_,_=fixture('cpu');other=copy.deepcopy(model)
    train={'query':torch.randn(8,1,8),'reference':torch.randn(8,1,8)}
    initialize_ablation(model,train,'difference',42)
    initialize_ablation(other,train,'raw',42)
    for a,b in zip(model.idm.head.parameters(),other.idm.head.parameters()):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(model.idm.scale,(train['query']-train['reference']).square().mean().sqrt())
    torch.testing.assert_close(other.idm.scale,train['query'].square().mean().sqrt())
    assert torch.equal(make_batches(42,4,512),make_batches(42,8,512)[:4])
    assert not torch.equal(make_batches(42,4,512),make_batches(43,4,512))


def test_paired_bootstrap_and_incomplete_runs_cannot_unlock_test(tmp_path):
    right=np.ones((3,4));left=right-.1
    result=paired_comparison(left,right,['a','a','a','b'])
    assert result['mse_difference']==pytest.approx(-.1)
    np.testing.assert_allclose(result['episode_paired_ci95'],[-.1,-.1])
    np.testing.assert_allclose(result['video_cluster_paired_ci95'],[-.1,-.1])
    with pytest.raises(FileNotFoundError):verify_runs(tmp_path)


@pytest.mark.parametrize('device',DEVICES)
def test_optimizer_resume_matches_uninterrupted_training(tmp_path,device):
    model,x,y=fixture(device);other=copy.deepcopy(model)
    opt=optimizer_for(model);opt2=optimizer_for(other)
    def step(m,o):
        o.zero_grad(set_to_none=True);(m(x,y)[0]-y).square().mean().backward();o.step()
    step(model,opt)
    protocol={'representation':'difference','steps':2}
    path=tmp_path/'latest.pt'
    save_training(path,model,opt,1,[{'step':1}],protocol,1.)
    step(model,opt)
    restored=restore_training(path,other,opt2,protocol)
    assert restored==(1,[{'step':1}],1.)
    step(other,opt2)
    for a,b in zip(model.parameters(),other.parameters()):torch.testing.assert_close(a,b,rtol=0,atol=0)
    with pytest.raises(ValueError,match='protocol'):
        restore_training(path,other,opt2,{'representation':'raw'})
