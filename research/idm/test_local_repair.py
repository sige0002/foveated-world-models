"""Synthetic freeze/gradient/mean-route tests; never fit the real dataset."""
import copy
import unittest
import torch
from torch import nn
from research.idm.train_local_repair import (
    RepairModel, anchors, losses, gradient_diagnostics, state_hash,
    tensor_hash, frozen_audit, mean_head_from_gaussian, row_identity,
)


class TinyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.code = nn.Linear(32, 4, bias=False)
        self.current = nn.Linear(4, 4)
    def forward(self, current, mu, *, baseline):
        return baseline + .25*self.current(current) + self.code(mu).unsqueeze(-2)


class LocalRepairTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2); torch.set_float32_matmul_precision('highest')
        torch.manual_seed(45)
        self.head = nn.Sequential(nn.Linear(5, 7), nn.GELU(), nn.Linear(7, 64))
        self.decoder = TinyDecoder()
        self.x = torch.randn(4, 1, 5)
        self.h, self.b, self.y = [torch.randn(4, 1, 3, 4) for _ in range(3)]
        self.anchor = anchors(self.head, {'train': self.x})['train']
        self.constant = self.anchor['mu'].mean(0, keepdim=True)
    def model(self, arm):
        return RepairModel(self.head, self.decoder, self.constant, arm)
    def loss(self, model):
        return losses(model,self.x,self.h,self.b,self.y,self.anchor['mu'],self.anchor['logvar'])
    def test_common_start_and_zero_initial_drift(self):
        values=[]
        for arm in ('F','H','D','HD'):
            model=self.model(arm);values.append(model(self.x,self.h,self.b)[0])
            self.assertEqual(float(self.loss(model)[2]),0.)
        self.assertTrue(all(torch.equal(v,values[0]) for v in values))
        model=self.model('K')
        self.assertTrue(torch.equal(model(self.x,self.h,self.b)[0],model(self.x*123,self.h,self.b)[0]))
        self.assertEqual(model.head[-1].out_features,32)
    def test_gradient_routes_and_frozen_states_after_update(self):
        reference_hash={k:tensor_hash(v) for k,v in self.anchor.items()}
        for arm in ('H','D','HD','K'):
            model=self.model(arm)
            initial=dict(head=state_hash(model.head),decoder=state_hash(model.decoder),constant_mu=tensor_hash(model.constant_mu))
            parameters=[p for p in model.parameters() if p.requires_grad]
            opt=torch.optim.AdamW(parameters,lr=1e-3,weight_decay=.01)
            loss,*_=self.loss(model);loss.backward()
            for name,expected in [('head',arm in ('H','HD')),('decoder',arm in ('D','HD','K'))]:
                grads=[p.grad for p in getattr(model,name).parameters()]
                if expected:self.assertGreater(sum(float(g.norm()) for g in grads if g is not None),0.)
                else:self.assertTrue(all(g is None for g in grads))
            opt.step();final=frozen_audit(model,initial)
            for name,expected in [('head',arm in ('H','HD')),('decoder',arm in ('D','HD','K'))]:
                self.assertEqual(final[name]!=initial[name],expected)
            self.assertTrue(all(tensor_hash(v)==reference_hash[k] for k,v in self.anchor.items()))
    def test_drift_is_gaussian_kl_with_fixed_variance(self):
        model=self.model('H')
        with torch.no_grad():model.head[-1].bias.add_(.1)
        _,_,drift,_=self.loss(model);mu=model.mean(self.x);scale=(.5*self.anchor['logvar']).exp()
        actual=torch.distributions.kl_divergence(torch.distributions.Normal(mu,scale),torch.distributions.Normal(self.anchor['mu'],scale)).sum(-1).mean()
        torch.testing.assert_close(drift,actual)
    def test_diagnostics_do_not_modify_rng_or_accumulate_gradients(self):
        model=self.model('HD');_,mse,drift,_=self.loss(model);rng=torch.get_rng_state().clone()
        diagnostics=gradient_diagnostics(model,mse,drift)
        self.assertGreater(diagnostics['head_mse_gradient_norm'],0.)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        mse.backward();self.assertGreater(model.head[-1].weight.grad.norm().item(),0.)
    def test_mean_head_conversion_preserves_rng_and_legacy_rows(self):
        rng=torch.get_rng_state().clone();m=mean_head_from_gaussian(self.head)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        torch.testing.assert_close(m(self.x),self.head(self.x)[...,:32])
        self.assertEqual(row_identity([dict(episode='a',index=3,dt=.5)]),[('a',3,.5,None)])


if __name__=='__main__':
    unittest.main()
