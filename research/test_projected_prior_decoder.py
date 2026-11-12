import unittest
import torch
from research.train_cached_decoders import make_decoder
from research.projected_prior_decoder import ProjectedPriorDecoder


class ProjectionTests(unittest.TestCase):
    def model(self, rank=4, deterministic=False):
        torch.manual_seed(42)
        return ProjectedPriorDecoder(context_dim=16,input_dim=16,code_dim=4,
                                     num_layers=1,num_heads=4,grid_hw=(4,4),
                                     deterministic=deterministic,steps=4,projection_rank=rank)

    def test_full_rank_reproduces_flow(self):
        torch.manual_seed(42)
        base = make_decoder('prior_flow',16,1,16,4,4)
        model = self.model(rank=16)
        # Factory uses .1 noise scale; default is the same.
        for key,value in base.state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[key],rtol=0,atol=0)
        h,target,noise = [torch.randn(2,1,16,16) for _ in range(3)]
        z = torch.randn(2,1,4)
        time = torch.tensor([.2,.7])
        torch.testing.assert_close(base(h,z,baseline=h,noise=noise),model(h,z,baseline=h,noise=noise))
        a = base.training_output(h,z,target,baseline=h,noise=noise,time=time)
        b = model.training_output(h,z,target,baseline=h,noise=noise,time=time)
        torch.testing.assert_close(a.velocity_loss,b.velocity_loss)

    def test_subspace_and_loss_decomposition(self):
        model = self.model()
        q,_ = torch.linalg.qr(torch.randn(16,4))
        model.set_basis(q)
        with torch.no_grad():
            model.output.weight.normal_(std=.01)
        h,target,noise = [torch.randn(2,1,16,16) for _ in range(3)]
        z = torch.randn(2,1,4)
        pred = model(h,z,baseline=h,noise=noise)
        torch.testing.assert_close(pred-h,model.project(pred-h),atol=2e-6,rtol=2e-5)
        delta = target-h
        discarded = delta-model.project(delta)
        projected_error = pred-h-model.project(delta)
        torch.testing.assert_close((pred-target).square().mean(),discarded.square().mean()+projected_error.square().mean())
        (pred-target).square().mean().backward()
        self.assertGreater(model.output.weight.grad.norm().item(),0)
        self.assertIsNone(model.projection_basis.grad)

    def test_invalid_basis(self):
        model = self.model()
        with self.assertRaises(ValueError):
            model.set_basis(torch.zeros(16,4))


if __name__ == '__main__':
    unittest.main()
