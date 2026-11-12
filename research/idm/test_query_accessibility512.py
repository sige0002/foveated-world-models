"""Structural tests for frozen-query readout, without dataset fitting."""
import unittest
import torch
from research.idm.probe_query_accessibility512 import pca_fit,pca_apply,combine

class QueryReadoutTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);self.rng=torch.Generator().manual_seed(610)
 def test_pca_orthonormal_and_fullrank_reconstruction(self):
  x=torch.randn(20,6,generator=self.rng,dtype=torch.float64);p=pca_fit(x,6)
  self.assertTrue(torch.allclose(p['components'].T@p['components'],torch.eye(6,dtype=x.dtype),atol=1e-12))
  reconstructed=pca_apply(x,p)@p['components'].T*p['scale']+p['mean']
  self.assertTrue(torch.allclose(x,reconstructed,atol=1e-12))
 def test_heldout_application_does_not_change_calibration(self):
  x=torch.randn(20,6,generator=self.rng,dtype=torch.float64);p=pca_fit(x,3);before={k:v.clone() for k,v in p.items() if isinstance(v,torch.Tensor)}
  pca_apply(torch.randn(5,6,generator=self.rng,dtype=torch.float64)*1000,p)
  self.assertTrue(all(torch.equal(p[k],v) for k,v in before.items()))
 def test_kernel_duplicate_coordinates_keep_strength(self):
  x=torch.randn(20,8,generator=self.rng,dtype=torch.float64);y=torch.randn(5,8,generator=self.rng,dtype=torch.float64);g=x@x.T/8;cross=y@x.T/8;norm=y.square().mean(-1)
  z=torch.randn(20,4,generator=self.rng,dtype=torch.float64);zq=torch.randn(5,4,generator=self.rng,dtype=torch.float64)
  k,q,_=combine(g,cross,norm,z,zq,1);k2,q2,_=combine(g,cross,norm,z.repeat(1,8),zq.repeat(1,8),1)
  self.assertTrue(torch.allclose(k,k2,atol=1e-12));self.assertTrue(torch.allclose(q,q2,atol=1e-12));self.assertAlmostEqual(float(k.diag().mean()),1,places=12)

if __name__=='__main__':unittest.main()
