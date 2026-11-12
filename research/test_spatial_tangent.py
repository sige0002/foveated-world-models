import unittest
import torch
from research.diagnose_spatial_tangent import gradients,solve_coefficients


class TangentTest(unittest.TestCase):
    def test_exact_recovery_and_box(self):
        basis=torch.eye(2,dtype=torch.float64)
        free,boxed=solve_coefficients(basis,torch.tensor([.2,2.],dtype=torch.float64))
        torch.testing.assert_close(free,torch.tensor([.2,2.],dtype=torch.float64))
        torch.testing.assert_close(boxed,torch.tensor([.2,.5],dtype=torch.float64))

    def test_correlated_box_solution(self):
        b=torch.tensor([[1.,1.],[0.,.2],[.1,0.]],dtype=torch.float64)
        e=torch.tensor([1.,.5,.3],dtype=torch.float64)
        _,c=solve_coefficients(b,e)
        grid=torch.cartesian_prod(torch.linspace(-.5,.5,101,dtype=torch.float64),torch.linspace(-.5,.5,101,dtype=torch.float64))
        self.assertLessEqual(float((b@c-e).square().sum()),float((grid@b.T-e).square().sum(-1).min())+1e-12)

    def test_gradient_of_linear_field_and_zero_basis(self):
        yy,xx=torch.meshgrid(torch.arange(5,dtype=torch.float64),torch.arange(6,dtype=torch.float64),indexing='ij')
        g=gradients((2*xx+3*yy)[None,:,:,None])
        torch.testing.assert_close(g[...,0],torch.full_like(g[...,0],2))
        torch.testing.assert_close(g[...,1],torch.full_like(g[...,1],3))
        free,boxed=solve_coefficients(torch.zeros(3,2,dtype=torch.float64),torch.ones(3,dtype=torch.float64))
        self.assertTrue(torch.isfinite(free).all() and torch.isfinite(boxed).all())


if __name__=='__main__':unittest.main()
