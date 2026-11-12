import unittest
import torch
from research.train_cached_decoders import make_decoder


class SpatialTangentTests(unittest.TestCase):
    def model(self, kind):
        torch.manual_seed(42)
        return make_decoder(kind, 16, 1, 16, 4, 4)

    def test_common_weights_and_initial_output(self):
        base = self.model('prior_residual')
        tangent = self.model('prior_tangent_residual')
        for key, value in base.state_dict().items():
            torch.testing.assert_close(value, tangent.state_dict()[key], rtol=0, atol=0)
        h = torch.randn(2, 1, 16, 16)
        z = torch.randn(2, 1, 4)
        prior = torch.randn_like(h)
        torch.testing.assert_close(tangent(h, z, baseline=prior), prior, rtol=0, atol=0)
        with torch.no_grad():
            base.output.weight.normal_()
            tangent.output.weight.copy_(base.output.weight)
        torch.testing.assert_close(base(h, z, baseline=prior), tangent(h, z, baseline=prior), rtol=0, atol=0)

    def test_basis_geometry_and_rotated_gram(self):
        model = self.model('prior_tangent_residual')
        rotated = self.model('prior_rotated_residual')
        yy, xx = torch.meshgrid(torch.arange(4.), torch.arange(4.), indexing='ij')
        grid = (2 * xx + 3 * yy)[None, :, :, None].expand(2, 4, 4, 16)
        basis = model.spatial_basis(grid.reshape(2, 1, 16, 16)).reshape(2, 4, 4, 16, 2)
        torch.testing.assert_close(basis[:, 1:-1, 1:-1, :, 0], torch.full((2, 2, 2, 16), 2.))
        torch.testing.assert_close(basis[:, 1:-1, 1:-1, :, 1], torch.full((2, 2, 2, 16), 3.))
        self.assertEqual(basis[:, [0, -1]].count_nonzero().item(), 0)
        self.assertEqual(basis[:, :, [0, -1]].count_nonzero().item(), 0)
        features = torch.randn(2, 1, 16, 16)
        a, b = model.spatial_basis(features), rotated.spatial_basis(features)
        torch.testing.assert_close(a.transpose(-1, -2) @ a, b.transpose(-1, -2) @ b)
        self.assertFalse(torch.equal(a, b))

    def test_coefficient_branch_learns(self):
        model = self.model('prior_tangent_residual')
        features = torch.randn(2, 1, 16, 16)
        z = torch.randn(2, 1, 4)
        target = model.spatial_basis(features)[..., 0].unsqueeze(1)
        loss = (model(features, z, baseline=torch.zeros_like(features)) - target).square().mean()
        loss.backward()
        grad = model.motion_coeff.weight.grad
        self.assertTrue(torch.isfinite(grad).all())
        self.assertGreater(grad.norm().item(), 0)


if __name__ == '__main__':
    unittest.main()
