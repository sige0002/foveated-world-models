import unittest
import torch
from research.idm.postprojection_representations import decompose,fit_calibration,representation,VARIANTS
class RepresentationTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2)
  self.g=torch.Generator().manual_seed(981)
  self.train=torch.randn(17,1,32,generator=self.g,dtype=torch.float64)*7+19
 def test_exact_reconstruction_and_affine_shift(self):
  u,m,logs=decompose(self.train)
  torch.testing.assert_close(m+logs.exp()*u,self.train,atol=1e-13,rtol=1e-13)
  shifted=self.train+123
  uu,mm,ll=decompose(shifted)
  torch.testing.assert_close(uu,u);torch.testing.assert_close(mm,m+123);torch.testing.assert_close(ll,logs)
 def test_constant_coordinate_values(self):
  p=torch.arange(1,18,dtype=torch.float64)[:,None,None].expand(-1,1,32).clone()
  u,m,logs=decompose(p)
  self.assertTrue(torch.equal(u,torch.zeros_like(u)))
  torch.testing.assert_close(m+logs.exp()*u,p)
  cal=fit_calibration(p)
  for variant in VARIANTS:self.assertTrue(torch.isfinite(representation(p,cal,variant)).all())
  self.assertGreater(float(representation(p,cal,'direction_statistics')[...,-2].var()),0)
 def test_train_calibration_fixed_under_validation(self):
  cal=fit_calibration(self.train);snapshot={k:v.clone() for k,v in cal.items() if isinstance(v,torch.Tensor)}
  for offset in (0.,1e8):
   val=torch.randn(5,1,32,generator=self.g,dtype=torch.float64)+offset
   for variant in VARIANTS:representation(val,cal,variant)
  for k,v in snapshot.items():self.assertTrue(torch.equal(v,cal[k]))
  torch.testing.assert_close(cal['raw_mean'],self.train.mean(0,keepdim=True))
  self.assertEqual(cal['fit_count'],17);self.assertEqual(cal['fit_split'],'train')
 def test_equal_input_shape_and_recover_calibrated_raw(self):
  cal=fit_calibration(self.train)
  for variant in VARIANTS:self.assertEqual(representation(self.train,cal,variant).shape,(17,1,34))
  raw=representation(self.train,cal,'calibrated_raw')
  torch.testing.assert_close(raw[...,:32]*cal['raw_scale']+cal['raw_mean'],self.train)
  self.assertTrue(torch.equal(raw[...,-2:],torch.zeros_like(raw[...,-2:])))
  stats=representation(self.train,cal,'direction_statistics')[...,-2:]*cal['statistics_scale']+cal['statistics_mean']
  u,_,_=decompose(self.train)
  torch.testing.assert_close(stats[...,:1]+stats[...,1:].exp()*u,self.train)
 def test_finite_epsilon_does_not_claim_scale_invariance(self):
  tiny=self.train*1e-5
  u,_,_=decompose(tiny);scaled,_,_=decompose(2*tiny)
  self.assertGreater(float((scaled-u).abs().max()),1e-4)
 def test_invalid_shape_or_variant_rejected(self):
  with self.assertRaises(ValueError):fit_calibration(torch.zeros(5,32))
  with self.assertRaises(ValueError):representation(self.train,fit_calibration(self.train),'unknown')

class SharedHeadTests(unittest.TestCase):
 def test_heads_have_identical_capacity_and_initial_values(self):
  from research.idm.postprojection_heads import shared_initial_heads
  heads=shared_initial_heads(seed=42)
  baseline=heads['direction'].state_dict()
  for model in heads.values():
   for key,value in model.state_dict().items():self.assertTrue(torch.equal(value,baseline[key]))
   self.assertEqual(model(torch.zeros(2,1,34)).shape,(2,1,64))
  self.assertIsNot(heads['direction'][0].weight,heads['calibrated_raw'][0].weight)

if __name__=='__main__':unittest.main()
