import unittest
import torch
from research.slim_ffn_decoder import make_slim_decoder


class SlimTests(unittest.TestCase):
    def test_common_weights_rng_and_initial_output(self):
        torch.set_num_threads(2)
        a=make_slim_decoder(4,42);rng=torch.get_rng_state().clone()
        b=make_slim_decoder(2,42)
        self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        for k,v in a.state_dict().items():
            if not k.startswith('head_block.ffn.'):
                self.assertTrue(torch.equal(v,b.state_dict()[k]),k)
        h,baseline=[torch.randn(1,1,256,768) for _ in range(2)];z=torch.randn(1,1,32)
        self.assertTrue(torch.equal(a(h,z,baseline=baseline),baseline))
        self.assertTrue(torch.equal(b(h,z,baseline=baseline),baseline))
        self.assertEqual(a.output.weight.shape,b.output.weight.shape)
        self.assertEqual(sum(p.numel() for p in a.parameters())-sum(p.numel() for p in b.parameters()),2360832)

    def test_reduced_ffn_receives_gradient_after_gates_open(self):
        torch.set_num_threads(2);model=make_slim_decoder(2,42)
        # Test an active trained-state pathway, rather than mistaking zero-init gates for a dead branch.
        with torch.no_grad():
            model.output.weight.normal_(std=.01)
            model.head_block.adaLN_modulation[-1].bias[5*768:].fill_(.1)
        h,b,y=[torch.randn(1,1,256,768) for _ in range(3)];z=torch.randn(1,1,32)
        (model(h,z,baseline=b)-y).square().mean().backward()
        self.assertGreater(model.head_block.ffn[0].weight.grad.norm().item(),0)
        self.assertGreater(model.head_block.ffn[3].weight.grad.norm().item(),0)


if __name__=='__main__':unittest.main()
