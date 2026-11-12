import copy
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import unittest
import torch
from torch import nn
from research.bounded_innovation import InnovationModel,fit_normalization,normalize,kl_summary

class SmallDecoder(nn.Module):
    def __init__(self):
        super().__init__();self.action_proj=nn.Linear(32,4);self.visual=nn.Linear(4,4)
    def forward(self,x,z,*,baseline):
        return baseline+self.visual(x)+self.action_proj(z).unsqueeze(2)

class Tests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(7)
        self.d=SmallDecoder();self.x=torch.randn(3,1,2,4);self.z=torch.randn(3,1,32)
        self.a=torch.randn(3,1,12);self.eps=torch.randn(3,1,8)
    def model(self,arm='future'):
        return InnovationModel(copy.deepcopy(self.d),arm,42,input_dim=12,width=9)
    def test_initial_and_common(self):
        old=self.d(self.x,self.z,baseline=self.x)
        ms=[self.model(a) for a in ['future','recoding','noise']]
        for m in ms:
            pred,_=m(self.x,self.z,self.a,self.eps,baseline=self.x)
            self.assertTrue(torch.equal(old,pred))
            for p,q in zip(ms[0].head.parameters(),m.head.parameters()):self.assertTrue(torch.equal(p,q))
        self.assertTrue(any(v.abs().sum()>0 for v in ms[0].head[-1].parameters()))
    def test_hard_bound(self):
        m=self.model()
        for a in [self.a,1e6*self.a]:
            mu=m.mean(a);self.assertLessEqual(kl_summary(mu)['max'],1+1e-6)
        self.assertTrue(torch.equal(self.model('noise').mean(self.a),torch.zeros(3,1,8)))
    def test_gradient_opens_without_noise_correlation(self):
        m=self.model();opt=torch.optim.SGD(m.parameters(),lr=.01)
        for step in range(2):
            opt.zero_grad()
            pred,_=m(self.x,self.z,self.a,torch.zeros_like(self.eps),baseline=self.x)
            pred.square().mean().backward()
            if step==0:self.assertGreater(m.decoder.action_proj.extra.weight.grad.norm().item(),0)
            else:self.assertGreater(m.head[-1].weight.grad.norm().item(),0)
            opt.step()
    def test_train_only_normalization(self):
        cal=fit_normalization(self.a);saved={k:v.clone() for k,v in cal.items()}
        normalize(self.a*100,cal)
        for k in cal:self.assertTrue(torch.equal(cal[k],saved[k]))
    def test_inactive_noise_head(self):
        m=self.model('noise')
        self.assertTrue(all(not p.requires_grad for p in m.head.parameters()))
        pred,_=m(self.x,self.z,self.a,self.eps,baseline=self.x);pred.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in m.head.parameters()))
    def test_risk_average_is_not_ensemble(self):
        predictions=torch.tensor([1.,-1.]);target=torch.tensor(0.)
        self.assertEqual(float((predictions-target).square().mean()),1.)
        self.assertEqual(float((predictions.mean()-target).square()),0.)
if __name__=='__main__':unittest.main()
