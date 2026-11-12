import unittest
import torch
from research.conditional_residual_codec import ConditionalResidualCodec


class CodecTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.basis = torch.linalg.qr(torch.randn(16,4))[0]
        self.model = ConditionalResidualCodec(self.basis,hidden=16,context_dim=8)
        self.current,self.baseline,self.future = [torch.randn(2,1,9,16) for _ in range(3)]

    def test_initial_pca_and_decoder_interface(self):
        h,b,y = self.current,self.baseline,self.future
        latent = (y-b) @ self.basis
        torch.testing.assert_close(self.model.encode(h,b,y),latent,rtol=0,atol=0)
        torch.testing.assert_close(self.model(h,b,y),b+latent@self.basis.T,rtol=0,atol=0)
        torch.testing.assert_close(self.model.decode(h,b,torch.zeros_like(latent)),b,rtol=0,atol=0)
        with self.assertRaises(TypeError):
            self.model.decode(h,b,latent,future=y)

    def test_trainable_nonlinear_paths(self):
        optimizer = torch.optim.AdamW(self.model.parameters(),lr=1e-3)
        h,b,y = self.current,self.baseline,self.future
        for _ in range(4):
            optimizer.zero_grad()
            loss = (self.model(h,b,y)-y).square().mean()
            loss.backward()
            optimizer.step()
        self.assertGreater(self.model.encoder_adapter[-1].weight.grad.norm().item(),0)
        self.assertGreater(self.model.decoder_adapter[-1].weight.grad.norm().item(),0)
        self.assertIsNone(self.model.basis.grad)

    def test_invalid_basis(self):
        with self.assertRaises(ValueError):
            ConditionalResidualCodec(torch.zeros(16,4))


if __name__ == '__main__':
    unittest.main()
