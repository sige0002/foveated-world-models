import unittest
import torch
from research.idm.probe_interaction512 import combine
from research.idm.probe_spatial_current import combined_kernel,solve_many
class InteractionTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2)
 def test_additive_and_zero_match_original(self):
  gen=torch.Generator().manual_seed(7);x=torch.randn(20,7,generator=gen,dtype=torch.float64);y=torch.randn(4,7,generator=gen,dtype=torch.float64);z=torch.randn(20,32,generator=gen,dtype=torch.float64);zz=torch.randn(4,32,generator=gen,dtype=torch.float64);g=x@x.T/7;q=y@x.T/7;n=y.square().mean(-1)
  for v in ['additive','zero']:
   k,c,_=combine(g,q,n,z,zz,1,v);a,b,_=combined_kernel(g,q,n,z if v=='additive' else torch.zeros_like(z),zz if v=='additive' else torch.zeros_like(zz),1)
   self.assertTrue(torch.equal(k,a));self.assertTrue(torch.equal(c,b))
 def test_product_is_psd_and_cross_training_matches(self):
  gen=torch.Generator().manual_seed(7);x=torch.randn(20,7,generator=gen,dtype=torch.float64);z=torch.randn(20,32,generator=gen,dtype=torch.float64);g=x@x.T/7
  k,c,_=combine(g,g,g.diag(),z,z,1,'interaction')
  self.assertTrue(torch.allclose(k,c,atol=1e-12));self.assertGreater(float(torch.linalg.eigvalsh(k).min()),-1e-12);self.assertAlmostEqual(float(k.diag().mean()),1,places=12)
 def test_reads_state_dependent_sign_that_additive_cannot(self):
  x=torch.tensor([-1.,-1.,1.,1.],dtype=torch.float64)[:,None];z=torch.tensor([-1.,1.,-1.,1.],dtype=torch.float64)[:,None].repeat(1,32);g=x@x.T;target=x*z[:,:1]
  errors={}
  for v in ['additive','interaction']:
   k,c,_=combine(g,g,g.diag(),z,z,None,v);prediction=solve_many(k,c,target,[1e-6])[0];errors[v]=float((prediction-target).square().mean())
  self.assertGreater(errors['additive'],.99);self.assertLess(errors['interaction'],1e-10)
if __name__=='__main__':unittest.main()
