import unittest
import torch
from research.idm.probe_displacement_ridge import fit,predict,scaler,FLOOR,fold_indices
class RidgeTests(unittest.TestCase):
 def setUp(self):torch.set_num_threads(2);torch.manual_seed(99)
 def test_dual_matches_primal_with_unpenalized_intercept(self):
  x=torch.randn(15,7,dtype=torch.float64);y=torch.randn(15,8,dtype=torch.float64);m=fit(x,y,.1)
  xx=(x-m['x_mean'])/m['x_scale'];yy=(y-m['y_mean'])/m['y_scale']
  primal=torch.linalg.solve(xx.T@xx+.1*torch.eye(7,dtype=x.dtype),xx.T@yy)
  torch.testing.assert_close(m['weight'],primal,atol=1e-12,rtol=1e-12)
  torch.testing.assert_close(predict(m,x).mean(0),y.mean(0),atol=1e-12,rtol=1e-12)
 def test_zero_features_and_target_floors(self):
  x=torch.zeros(12,820,dtype=torch.float64);y=torch.zeros(12,8,dtype=torch.float64);m=fit(x,y,1.)
  self.assertTrue(torch.equal(m['y_scale'],FLOOR));self.assertTrue(torch.isfinite(predict(m,x)).all())
  self.assertTrue(torch.equal(predict(m,x),y))
 def test_group_folds_never_split_episode(self):
  episodes=[str(i//2) for i in range(40)];folds=fold_indices(episodes)
  self.assertEqual(sorted(torch.cat(folds).tolist()),list(range(40)))
  groups=[{episodes[i] for i in f} for f in folds]
  for i in range(5):
   for j in range(i):self.assertTrue(groups[i].isdisjoint(groups[j]))
 def test_validation_not_used_in_scaler(self):
  x=torch.randn(12,5,dtype=torch.float64);y=torch.randn(12,8,dtype=torch.float64);m=fit(x,y,1.)
  original={k:v.clone() for k,v in m.items()};predict(m,torch.full((3,5),1e9,dtype=x.dtype))
  for k,v in original.items():self.assertTrue(torch.equal(v,m[k]))
  torch.testing.assert_close(m['x_mean'],x.mean(0))
if __name__=='__main__':unittest.main()
