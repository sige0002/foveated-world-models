"""Check the flow objective, target-free integration and conditioning gradients."""
from pathlib import Path
import sys
import types

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'vendor/lawam'))
from latent_action_model.core.utils.latent_flow import LatentFlowDecoder


def decoder(**kwargs):
    return LatentFlowDecoder(context_dim=16, input_dim=8, code_dim=4,
                             num_layers=1, num_heads=2, grid_hw=(2, 2), **kwargs)


@pytest.mark.parametrize('source', ['current', 'noise'])
def test_constant_transport_is_exact(source):
    model = decoder(source=source, noise_scale=0.2)
    current = torch.randn(2, 1, 4, 8)
    noise = torch.randn_like(current)
    actions = torch.randn(2, 1, 4)
    displacement = torch.randn_like(current)
    target = model.initial_state(current, noise) + displacement
    model.velocity = types.MethodType(lambda self, x, time, features, actions: displacement, model)
    output = model.training_output(current, actions, target, noise=noise,
                                   time=torch.tensor([0., 0.7]))
    assert output.velocity_loss.item() < 1e-12
    torch.testing.assert_close(output.prediction, target)
    for steps in (1, 4, 8):
        torch.testing.assert_close(model(current, actions, noise=noise, steps=steps), target)


def test_gradients_and_reproducible_sampling():
    torch.manual_seed(7)
    model = decoder()
    current = torch.randn(2, 1, 4, 8)
    actions = torch.randn(2, 1, 4, requires_grad=True)
    output = model.training_output(current, actions, torch.randn_like(current))
    output.velocity_loss.backward()
    # AdaLN gates begin at zero; take a step before testing action sensitivity.
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.add_(parameter.grad, alpha=-0.01)
    model.zero_grad()
    actions.grad = None
    model.training_output(current, actions, torch.randn_like(current)).velocity_loss.backward()
    assert torch.isfinite(actions.grad).all() and actions.grad.abs().sum() > 0
    for module in (model.current_proj, model.state_proj, model.time_proj, model.output):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters())
    model.eval()
    noise = torch.randn_like(current)
    with torch.no_grad():
        first = model(current, actions, noise=noise)
        torch.testing.assert_close(first, model(current, actions, noise=noise))
        changed = model(current, actions, noise=-noise)
        assert not torch.allclose(first, changed)
    with pytest.raises(TypeError):
        model(current, actions, target=current)


@pytest.mark.parametrize('options', [{'steps': 0}, {'noise_scale': 0},
                                     {'noise_scale': float('nan')}, {'source': 'bad'}])
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        decoder(**options)


def test_refinement_does_not_use_training_target():
    from latent_action_model.core.utils.latent_flow import DeterministicRefinementDecoder
    model = DeterministicRefinementDecoder(context_dim=16, input_dim=8, code_dim=4,
                                           num_layers=1, num_heads=2, grid_hw=(2, 2), steps=2)
    current, actions = torch.randn(2, 1, 4, 8), torch.randn(2, 1, 4)
    first = model.training_output(current, actions, torch.zeros_like(current))
    second = model.training_output(current, actions, torch.ones_like(current))
    torch.testing.assert_close(first, second)
    first.square().mean().backward()
    assert model.output.weight.grad.abs().sum() > 0


def test_variant_keeps_idm_initialization_identical(monkeypatch):
    from latent_action_model.core import lam_model
    monkeypatch.setattr(lam_model, 'build_vision_encoder', lambda *a, **k: (torch.nn.Identity(), 8))
    options = dict(dim=16, num_heads=2, enc_layers=1, dec_layers=1, code_dim=4,
                   image_hw=(32, 32), num_frames=2, num_queries=1,
                   max_state_dim=8, num_embodiments=2, vq_type='vae')
    states = []
    for kind in ('direct', 'flow', 'refinement'):
        torch.manual_seed(42)
        model = lam_model.LatentLAMModel(**options, decoder_type=kind)
        states.append({k: v.clone() for k, v in model.state_dict().items()
                       if not k.startswith('decoder.')})
    for other in states[1:]:
        assert other.keys() == states[0].keys()
        for key in other:
            torch.testing.assert_close(other[key], states[0][key], rtol=0, atol=0)
