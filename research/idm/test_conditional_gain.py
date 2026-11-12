"""Exact finite-support checks for the proposed residual-grounding objective."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from research.idm.conditional_gain_objective import grounded_gain

class ConditionalGainChecks(unittest.TestCase):
    def setUp(self):
        # Full factorial current × physical innovation × independent nuisance.
        self.current=torch.tensor([-1.,-1.,1.,1.]).repeat_interleave(2).unsqueeze(1)
        self.motion=torch.tensor([-1.,1.,-1.,1.]).repeat_interleave(2).unsqueeze(1)
        self.nuisance=torch.tensor([-1.,1.]*4).unsqueeze(1)
        self.target=2*self.current+self.motion
        self.baseline=2*self.current;self.scale=torch.ones(1)
    def gain(self,prediction,baseline=None):
        return grounded_gain(prediction,self.target,self.baseline if baseline is None else baseline,self.scale)['diagnostic_gain'].item()
    def test_null_is_zero(self):self.assertEqual(self.gain(torch.zeros_like(self.motion)),0.)
    def test_physical_innovation_gains(self):self.assertEqual(self.gain(self.motion),1.)
    def test_independent_nuisance_cannot_gain(self):
        self.assertLess(self.gain(self.nuisance),0.)
        # Its conditional-mean optimum is zero on this full factorial support.
        for v in [-1.,1.]:self.assertEqual(self.motion[self.nuisance==v].mean().item(),0.)
    def test_bad_baseline_allows_current_shortcut(self):
        self.assertEqual(self.gain(self.current,baseline=self.current),1.)
    def test_baseline_detached(self):
        prediction=torch.zeros_like(self.motion,requires_grad=True)
        baseline=self.baseline.clone().requires_grad_(True)
        result=grounded_gain(prediction,self.target,baseline,self.scale);result['loss'].backward()
        self.assertIsNotNone(prediction.grad);self.assertIsNone(baseline.grad)
if __name__=='__main__':unittest.main()
