import torch
from research.train_cached_decoders import make_decoder


def test_wide_controls_have_identical_initial_weights_and_expected_noise_behavior():
    torch.manual_seed(17)
    flow = make_decoder('wide_flow', 8, 1, 32, 4, 2).eval()
    torch.manual_seed(17)
    residual = make_decoder('wide_residual', 8, 1, 32, 4, 2).eval()
    for key, value in flow.state_dict().items():
        torch.testing.assert_close(value, residual.state_dict()[key])
    assert torch.linalg.matrix_rank(flow.output.weight) == 32
    h, z = torch.randn(2, 1, 4, 32), torch.randn(2, 1, 4)
    torch.testing.assert_close(residual(h, z), residual(h, z))
    assert not torch.allclose(flow(h, z), flow(h, z))
    loss = flow.training_output(h, z, h + 0.1).velocity_loss
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters())


def test_raw_input_path_survives_zeroed_narrow_trunk():
    model = make_decoder('wide_flow', 8, 1, 32, 4, 2).eval()
    with torch.no_grad():
        model.trunk_to_head.weight.zero_()
        model.trunk_to_head.bias.zero_()
    h, z = torch.randn(2, 1, 4, 32), torch.randn(2, 1, 4)
    first = model.velocity(h, torch.zeros(2), h, z)
    changed_state = model.velocity(h + 0.1, torch.zeros(2), h, z)
    changed_current = model.velocity(h, torch.zeros(2), h + 0.1, z)
    assert not torch.allclose(first, changed_state)
    assert not torch.allclose(first, changed_current)


def test_wide_refinement_matches_four_generated_euler_steps_with_gradients():
    torch.manual_seed(19)
    model = make_decoder('wide_refinement', 8, 1, 32, 4, 2).eval()
    torch.manual_seed(19)
    flow = make_decoder('wide_flow', 8, 1, 32, 4, 2).eval()
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, flow.state_dict()[key])
    h, z = torch.randn(2, 1, 4, 32), torch.randn(2, 1, 4)
    manual = h
    velocities = []
    for step in range(4):
        v = model.velocity(manual, torch.full((2,), step/4), h, z)
        v.retain_grad()
        velocities.append(v)
        manual = manual + v/4
    torch.testing.assert_close(model(h, z), manual)
    manual.square().mean().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all()
               and v.grad.abs().sum() > 0 for v in velocities)
    assert model.stochastic is False


def test_terminal_objective_control_has_identical_flow_forward_and_weights():
    torch.manual_seed(23)
    terminal = make_decoder('wide_terminal', 8, 1, 32, 4, 2).eval()
    torch.manual_seed(23)
    flow = make_decoder('wide_flow', 8, 1, 32, 4, 2).eval()
    h, z = torch.randn(2, 1, 4, 32), torch.randn(2, 1, 4)
    noise = torch.randn_like(h)
    torch.testing.assert_close(terminal(h, z, noise=noise), flow(h, z, noise=noise))
    assert terminal.stochastic and terminal.steps == flow.steps == 4
    terminal(h, z, noise=noise).square().mean().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in terminal.parameters())
