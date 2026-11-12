import unittest
import torch
from research.idm.probe_spatial_current import geometry,combined_kernel,solve_many
class SpatialKernelTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.manual_seed(701)
  self.i=torch.randn(12,24,dtype=torch.float64);self.s=torch.randn(12,20,dtype=torch.float64);self.z=torch.randn(12,32,dtype=torch.float64)
 def test_zero_and_nonzero_code_have_same_total_kernel_scale(self):
  g,q,n,_=geometry(self.i,self.s,self.i,self.s)
  for mult in (None,.25,1.,4.):
   for z in (self.z,torch.zeros_like(self.z)):
    k,c,_=combined_kernel(g,q,n,z,z,mult)
    self.assertAlmostEqual(float(k.diag().mean()),1.,places=12)
    torch.testing.assert_close(k.mean(0),torch.zeros(12,dtype=k.dtype),atol=1e-12,rtol=0)
    torch.testing.assert_close(k,c,atol=1e-12,rtol=1e-12)
 def test_constant_code_exactly_matches_zero(self):
  g,q,n,_=geometry(self.i,self.s,self.i,self.s)
  a,aa,_=combined_kernel(g,q,n,torch.zeros_like(self.z),torch.zeros_like(self.z),1.)
  b,bb,_=combined_kernel(g,q,n,torch.ones_like(self.z)*7,torch.ones_like(self.z)*7,1.)
  self.assertTrue(torch.equal(a,b));self.assertTrue(torch.equal(aa,bb))
 def test_query_values_do_not_fit_training_scale_or_bandwidth(self):
  g,q,n,sc=geometry(self.i,self.s,self.i+1e5,self.s-1e5)
  gg,qq,nn,ssc=geometry(self.i,self.s,self.i,self.s)
  self.assertTrue(torch.equal(g,gg))
  for group in sc:
   for key in sc[group]:self.assertTrue(torch.equal(sc[group][key],ssc[group][key]))
  a,_,meta=combined_kernel(g,q,n,self.z,self.z,1.)
  b,_,other=combined_kernel(gg,qq,nn,self.z,self.z,1.)
  self.assertTrue(torch.equal(a,b));self.assertEqual(meta['median_current_distance'],other['median_current_distance'])
 def test_eigen_solution_matches_direct_ridge(self):
  g,q,n,_=geometry(self.i,self.s,self.i,self.s);k,c,_=combined_kernel(g,q,n,self.z,self.z,1.)
  y=torch.randn(12,8,dtype=torch.float64);y-=y.mean(0)
  pred=solve_many(k,c,y,[.1])[0];direct=c@torch.linalg.solve(k+.1*torch.eye(12,dtype=k.dtype),y)
  torch.testing.assert_close(pred,direct,atol=1e-12,rtol=1e-12)
if __name__=='__main__':unittest.main()
