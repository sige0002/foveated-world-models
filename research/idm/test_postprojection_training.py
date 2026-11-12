"""Synthetic-only harness checks; never loads research feature caches."""
import unittest,hashlib
import torch
from research.idm.train_postprojection_heads import Model,objective,schedule,evaluate,head_gradient_diagnostics,clip_gradients,add_constant_control
from research.idm.train_reference_heads import weight_hash
class HarnessTests(unittest.TestCase):
 def setUp(self):
  torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.manual_seed(991)
  self.model=Model(42,feature_dim=8,grid=2,dim=16,layers=1)
  self.x=torch.randn(3,1,34,requires_grad=True);self.h=torch.randn(3,1,4,8,requires_grad=True);self.y=torch.randn_like(self.h,requires_grad=True);self.b=torch.randn_like(self.h,requires_grad=True);self.eps=torch.randn(3,1,32)
 def test_initial_prediction_equals_frozen_baseline(self):
  mu,_=self.model.posterior(self.x)
  self.assertTrue(torch.equal(self.model.decoder(self.h,mu,baseline=self.b),self.b))
 def test_objective_kl_reduction_and_frozen_inputs(self):
  loss,mse,kl=objective(self.model,self.x,self.h,self.y,self.b,self.eps)
  mu,lv=self.model.posterior(self.x.detach());expected=.5*(mu.square()+lv.exp()-1-lv).sum(-1).mean()
  torch.testing.assert_close(kl,expected);torch.testing.assert_close(loss,mse+5e-5*expected)
  loss.backward()
  for t in (self.x,self.h,self.y,self.b):self.assertIsNone(t.grad)
  self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.head.parameters() if p.grad is not None),0)
 def test_reconstruction_gradient_reaches_head_after_zero_gates_open(self):
  # Zero-initialized output blocks MSE gradients into upstream modules at step0.
  # Opening it is a synthetic path check, not a trained model result.
  with torch.no_grad():
   self.model.decoder.output.weight.normal_(0,.02)
   for block in [*self.model.decoder.blocks,self.model.decoder.head_block]:block.adaLN_modulation[-1].weight.normal_(0,.02)
  loss,_,_=objective(self.model,self.x,self.h,self.y,self.b,self.eps,beta=0)
  loss.backward()
  self.assertGreater(sum(float(p.grad.abs().sum()) for p in self.model.head.parameters() if p.grad is not None),0)
 def test_common_initial_weights_and_schedules(self):
  other=Model(42,feature_dim=8,grid=2,dim=16,layers=1)
  self.assertEqual(weight_hash(self.model),weight_hash(other))
  a=list(schedule(42,3,8,128));b=list(schedule(42,3,8,128))
  for (ia,ea),(ib,eb) in zip(a,b):
   self.assertTrue(torch.equal(ia,ib));self.assertTrue(torch.equal(ea,eb));self.assertEqual(ea.shape,(8,1,32))
 def test_diagnostic_is_rng_grad_and_update_neutral(self):
  other=Model(42,feature_dim=8,grid=2,dim=16,layers=1)
  a=torch.optim.AdamW(self.model.parameters(),lr=3e-4);b=torch.optim.AdamW(other.parameters(),lr=3e-4)
  loss,mse,kl=objective(self.model,self.x,self.h,self.y,self.b,self.eps)
  rng=torch.random.get_rng_state().clone()
  norms=head_gradient_diagnostics(self.model,mse,kl,5e-5)
  self.assertTrue(torch.equal(rng,torch.random.get_rng_state()))
  self.assertTrue(all(p.grad is None for p in self.model.parameters()))
  self.assertEqual(norms['head_mse_gradient_norm'],0.)
  self.assertGreater(norms['head_raw_kl_gradient_norm'],0.)
  loss.backward();a.step()
  loss2,_,_=objective(other,self.x,self.h,self.y,self.b,self.eps);loss2.backward();b.step()
  self.assertEqual(weight_hash(self.model),weight_hash(other))
 def test_nonfinite_gradients_rejected(self):
  next(self.model.parameters()).grad=torch.full_like(next(self.model.parameters()),float("nan"))
  with self.assertRaises(RuntimeError):clip_gradients(self.model)
 def test_constant_control_is_zero34_shared_posterior_and_equal_model(self):
  original={'train':{'direction':torch.randn(7,1,34)},'validation':{'direction':torch.randn(5,1,34)}}
  expanded=add_constant_control(original)
  for split,n in [('train',7),('validation',5)]:
   self.assertNotIn('constant',original[split])
   self.assertIs(expanded[split]['direction'],original[split]['direction'])
   z=expanded[split]['constant'];self.assertEqual(z.shape,(n,1,34))
   self.assertTrue(torch.equal(z,torch.zeros_like(z)))
   mu,lv=self.model.posterior(z)
   torch.testing.assert_close(mu,mu[:1].expand_as(mu),atol=0,rtol=0)
   torch.testing.assert_close(lv,lv[:1].expand_as(lv),atol=0,rtol=0)
  other=Model(42,feature_dim=8,grid=2,dim=16,layers=1)
  self.assertEqual(weight_hash(self.model),weight_hash(other))
  self.assertEqual(sum(p.numel() for p in self.model.parameters()),sum(p.numel() for p in other.parameters()))
 def test_logvar_clamped(self):
  with torch.no_grad():self.model.head[-1].weight.zero_();self.model.head[-1].bias[32:48].fill_(100);self.model.head[-1].bias[48:].fill_(-100)
  _,lv=self.model.posterior(self.x)
  self.assertEqual(float(lv.detach().max()),10);self.assertEqual(float(lv.detach().min()),-10)
 def test_validation_saves_every_row_and_reuses_noise(self):
  d=dict(current=self.h.detach(),future=self.y.detach(),baseline=self.b.detach(),rows=[dict(episode=str(i),index=i,dt=1.) for i in range(3)])
  a=evaluate(self.model,d,self.x.detach(),batch=2);b=evaluate(self.model,d,self.x.detach(),batch=2)
  self.assertEqual(a,b);self.assertEqual(len(a['rows']),3)
  for row in a['rows']:
   self.assertEqual(row['mean_mse'],row['prior_mse']);self.assertEqual(row['sample_mse'],row['prior_mse'])
if __name__=='__main__':unittest.main()
