import unittest
import torch
from research.train_cached_decoders import make_decoder


class HorizonTests(unittest.TestCase):
    def models(self):
        torch.manual_seed(42)
        base=make_decoder('prior_flow',16,1,16,4,4)
        rng=torch.get_rng_state().clone()
        torch.manual_seed(42)
        model=make_decoder('prior_horizon_flow',16,1,16,4,4)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for key,value in base.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        with torch.no_grad():
            base.output.weight.normal_(std=.01)
            model.output.weight.copy_(base.output.weight)
        return base,model

    def test_zero_head_and_nominal_equivalence(self):
        base,model=self.models()
        h,y,noise=[torch.randn(2,1,16,16) for _ in range(3)]
        z=torch.randn(2,1,4)
        options=dict(baseline=h,noise=noise)
        torch.testing.assert_close(base(h,z,**options),model(h,z,physical_time=torch.tensor([.2,1.2]),**options),rtol=0,atol=0)
        with torch.no_grad():
            model.horizon_proj.weight.normal_()
        dt=torch.full((2,),1.2)
        torch.testing.assert_close(base(h,z,**options),model(h,z,physical_time=dt,**options),rtol=0,atol=0)
        time=torch.tensor([.3,.7])
        a=base.training_output(h,z,y,time=time,**options)
        b=model.training_output(h,z,y,physical_time=dt,time=time,**options)
        torch.testing.assert_close(a.velocity_loss,b.velocity_loss,rtol=0,atol=0)

    def test_physical_condition_gradient_and_validation(self):
        _,model=self.models()
        h=torch.randn(2,1,16,16)
        z=torch.randn(2,1,4)
        # AdaLN starts with zero modulation, so conditioning gradients open only
        # after the existing modulation layers receive an optimization update.
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
        for _ in range(3):
            optimizer.zero_grad()
            model(h,z,baseline=h,noise=torch.zeros_like(h),physical_time=torch.tensor([.2,.8])).square().mean().backward()
            optimizer.step()
        self.assertGreater(model.horizon_proj.weight.grad.norm().item(),0)
        for dt in (torch.tensor([-1.,1.]),torch.tensor([float('nan'),1.]),torch.ones(2,1)):
            with self.assertRaises(ValueError):
                model(h,z,baseline=h,physical_time=dt)


if __name__ == '__main__':
    unittest.main()
