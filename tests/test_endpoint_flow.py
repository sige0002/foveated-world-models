from research.train_cached_decoders import make_decoder
import torch


def test_analytic_skip_removes_orthogonal_noise_at_every_step_count():
    torch.manual_seed(21)
    model = make_decoder('endpoint_flow', 8, 1, 32, 4, 2).eval()
    h, z = torch.randn(2, 1, 4, 32), torch.randn(2, 1, 4)
    noise = torch.randn_like(h)
    q = torch.linalg.qr(model.output.weight.detach(), mode='reduced').Q
    for steps in (1, 4, 16):
        out = model(h, z, noise=noise, steps=steps)
        residual = out - h - model.output.bias
        orthogonal = residual - (residual @ q) @ q.T
        assert orthogonal.abs().max() < 2e-6


def test_endpoint_loss_is_weighted_velocity_loss_and_has_finite_gradients():
    model = make_decoder('endpoint_flow', 8, 1, 32, 4, 2)
    h, target, z = torch.randn(2, 1, 4, 32), torch.randn(2, 1, 4, 32), torch.randn(2, 1, 4)
    noise, time = torch.randn_like(h), torch.tensor([0.2, 0.9])
    result = model.training_output(h, z, target, noise=noise, time=time)
    t = time.reshape(-1, 1, 1, 1)
    source = model.initial_state(h, noise)
    x = (1-t)*source + t*target
    velocity_error = model.velocity(x, time, h, z) - (target-source)
    torch.testing.assert_close(result.velocity_loss, ((1-t)*velocity_error).square().mean())
    result.velocity_loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())


def test_source_only_control_cannot_use_future_in_prediction():
    model = make_decoder('noisy_residual', 8, 1, 32, 4, 2).eval()
    h, z = torch.randn(2, 1, 4, 32), torch.randn(2, 1, 4)
    noise = torch.randn_like(h)
    first = model.training_output(h, z, torch.randn_like(h), noise=noise)
    second = model.training_output(h, z, torch.randn_like(h), noise=noise)
    torch.testing.assert_close(first.prediction, second.prediction)
    torch.testing.assert_close(first.prediction, model(h, z, noise=noise))
