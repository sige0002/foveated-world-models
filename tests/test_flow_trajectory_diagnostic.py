import torch
from research.diagnose_flow_trajectory import local_metrics, code_swap_metrics


def test_oracle_constant_velocity_has_zero_teacher_and_generated_error():
    x = torch.randn(2, 1, 3, 4)
    y = torch.randn_like(x)
    t = 0.5
    state = (1-t)*x+t*y
    m = local_metrics(x, y, x, state, t, y-x, y-x)
    torch.testing.assert_close(m['teacher_endpoint_mse'], torch.zeros(2), atol=1e-12, rtol=0)
    torch.testing.assert_close(m['generated_endpoint_mse'], torch.zeros(2), atol=1e-12, rtol=0)
    torch.testing.assert_close(m['state_gap'], torch.zeros(2), atol=0, rtol=0)


def test_zero_field_does_not_gain_future_information_on_generated_path():
    x = torch.zeros(2, 1, 3, 4)
    y = torch.ones_like(x)
    m = local_metrics(x, y, x, x, 0.75, x, x)
    torch.testing.assert_close(m['teacher_endpoint_mse'], torch.full((2,), 0.0625))
    torch.testing.assert_close(m['generated_endpoint_mse'], torch.ones(2))
    torch.testing.assert_close(m['state_gap'], torch.full((2,), 0.5625))


def test_endpoint_swap_gap_shrinks_even_when_velocity_code_use_is_constant():
    x = torch.zeros(2, 1, 3, 4)
    y = torch.ones_like(x)
    for t in (0.0, 0.5, 0.75):
        m = code_swap_metrics(y, x, t*y, t, y, y, x, x)
        torch.testing.assert_close(m['teacher_swap_endpoint_gap'], torch.full((2,), (1-t)**2))
        torch.testing.assert_close(m['teacher_swap_velocity_gap'], torch.ones(2))
        torch.testing.assert_close(m['teacher_code_velocity_sensitivity'], torch.ones(2))


def test_code_independent_field_has_zero_swap_metrics_at_any_state():
    x, y, state, vt, vg = [torch.randn(2, 1, 3, 4) for _ in range(5)]
    m = code_swap_metrics(y, x, state, 0.5, vt, vg, vt, vg)
    for value in m.values():
        torch.testing.assert_close(value, torch.zeros(2), atol=0, rtol=0)
