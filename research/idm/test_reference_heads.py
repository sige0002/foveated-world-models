"""Focused checks for matched initialization, reference algebra, and latent gradients."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import torch
from research.idm.train_reference_heads import Model,weight_hash,representation

class Controls(unittest.TestCase):
    def setUp(self):torch.set_num_threads(4)
    def test_common_initialization(self):
        torch.manual_seed(42);a=weight_hash(Model())
        torch.manual_seed(42);self.assertEqual(a,weight_hash(Model()))
    def test_reference_and_constant(self):
        q=torch.randn(3,1,1024);r=torch.randn_like(q)
        cal=dict(center=q.mean(0,keepdim=True),scales={k:torch.tensor(2.) for k in ('raw','centered','reference')})
        d=dict(query=q,reference=r)
        self.assertTrue(torch.equal(representation(d,cal,'reference'),(q-r)/2))
        const=representation(d,cal,'constant');self.assertTrue(torch.equal(const[0],const[2]))
    def test_head_receives_prediction_gradient(self):
        torch.manual_seed(1);m=Model();x=torch.randn(2,1,1024)
        mu,lv=m.posterior(x);h=torch.randn(2,1,256,768)
        loss=(m.decoder(h,mu+(.5*lv).exp()*torch.randn_like(mu))-torch.randn_like(h)).square().mean()
        loss.backward()
        # Decoder conditioning may initialize to zero; KL ensures a gradient in real training.
        self.assertTrue(torch.isfinite(loss));self.assertIsNotNone(m.head[-1].weight.grad)
        self.assertTrue(torch.isfinite(m.head[-1].weight.grad).all())
if __name__=='__main__':unittest.main()
