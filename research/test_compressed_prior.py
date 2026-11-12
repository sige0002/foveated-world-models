import unittest
import torch
from torch import nn
from research.compressed_prior import fit_linear,CurrentPrior,objective


class Tests(unittest.TestCase):
    def test_linear_recovers_affine_and_is_batch_independent(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        x=torch.randn(40,1,7,4);w=torch.randn(4,4);bias=torch.randn(4)
        y=x+x@w+bias
        m,diagnostics=fit_linear(x,y,ridge=0.)
        torch.testing.assert_close(m(x),y,atol=2e-6,rtol=1e-5)
        self.assertLess(diagnostics['normal_equation_max_abs'],1e-12)
        torch.testing.assert_close(torch.cat([m(v) for v in x.split(3)]),m(x),rtol=0,atol=0)

    def test_current_only_initializes_to_current(self):
        torch.set_num_threads(2);m=CurrentPrior();x=torch.randn(1,1,256,768)
        self.assertTrue(torch.equal(m(x),x))
        with self.assertRaises(TypeError):m(x,future=x)

    def test_downstream_gradient_crosses_frozen_correction(self):
        class Correction(nn.Module):
            def __init__(self):super().__init__();self.scale=nn.Parameter(torch.tensor(2.))
            def forward(self,current,code,*,baseline):return baseline*self.scale+current
        student=nn.Linear(3,3);correction=Correction().requires_grad_(False)
        x=torch.randn(4,3);target=torch.randn(4,3);code=torch.randn(4,1)
        loss=objective(student,correction,'downstream_teacher',x,target,target,code)
        loss.backward()
        self.assertGreater(student.weight.grad.norm().item(),0.)
        self.assertIsNone(correction.scale.grad)


if __name__=='__main__':unittest.main()
