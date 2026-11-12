"""Synthetic invariants; no real model fitting or checkpoint loading."""
import sys
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]),str(Path(__file__).resolve().parents[1]/'vendor/lawam')]
import unittest
import torch
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from research.early_condition_adapter import install_adapter, frozen_hash


class TestEarlyCondition(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);torch.manual_seed(42)
        self.m=LAMDecoder_v2(context_dim=8,input_dim=4,code_dim=3,num_layers=2,
                            num_heads=2,grid_hw=(2,2),dropout=0,ffn_expansion_factor=2).eval()
        # Emulate nonzero learned conditioning, not original scratch initialization.
        for b in self.m.dec_layers:
            torch.nn.init.normal_(b.adaLN_modulation[-1].weight,std=.1)
            torch.nn.init.normal_(b.adaLN_modulation[-1].bias,std=.1)
        self.x=torch.randn(2,1,4,4);self.z=torch.randn(2,1,3)

    def test_initial_and_rng(self):
        with torch.no_grad(): expected=self.m(self.x,torch.zeros_like(self.z))
        rng=torch.get_rng_state().clone();install_adapter(self.m)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        with torch.no_grad(): actual=self.m(self.x,self.z)
        self.assertTrue(torch.equal(expected,actual))
        self.assertEqual(sum(p.numel() for p in self.m.parameters() if p.requires_grad),24)

    def test_gradient_and_freeze_zero(self):
        install_adapter(self.m);h=frozen_hash(self.m)
        with torch.no_grad(): prior=self.m(self.x,torch.zeros_like(self.z)).clone()
        opt=torch.optim.AdamW([self.m.action_in_proj.U.weight],lr=.01,weight_decay=.01)
        loss=(self.m(self.x,self.z)-torch.randn_like(self.x)).square().mean();loss.backward()
        self.assertGreater(self.m.action_in_proj.U.weight.grad.norm().item(),0)
        self.assertTrue(all(p.grad is None for n,p in self.m.named_parameters() if n!='action_in_proj.U.weight'))
        opt.step();self.assertEqual(h,frozen_hash(self.m))
        with torch.no_grad(): self.assertTrue(torch.equal(prior,self.m(self.x,torch.zeros_like(self.z))))

    def test_subspace(self):
        install_adapter(self.m);a=self.m.action_in_proj
        with torch.no_grad(): a.U.weight.copy_(a.old_weight)
        self.assertLess(a.diagnostics(self.z)['outside_old_subspace_energy_fraction'],1e-12)
        self.assertEqual(a.diagnostics(torch.zeros_like(self.z))['outside_old_subspace_energy_fraction'],0.)

    def test_constant_condition(self):
        install_adapter(self.m);a=self.m.action_in_proj
        with torch.no_grad(): a.U.weight.normal_()
        z=self.z.mean(0,keepdim=True).expand_as(self.z)
        self.assertTrue(torch.equal(a(z)[0],a(z)[1]))
        # The diagnostic zero bypasses constant substitution: it must give c0.
        self.assertTrue(torch.equal(a(torch.zeros_like(z))[0,0],a.c0))

if __name__=='__main__': unittest.main()
