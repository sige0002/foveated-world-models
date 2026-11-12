import pytest
import torch
from research.train_cached_decoders import make_decoder, evaluate
from research.cache_current_prior import predict_prior


def inputs():
    torch.manual_seed(7)
    return torch.randn(3, 1, 4, 32), torch.randn(3, 1, 4), torch.randn(3, 1, 4, 32)


def test_initial_direct_correction_preserves_strong_prior_exactly():
    h, z, prior = inputs()
    model = make_decoder('prior_residual', 8, 1, 32, 4, 2).eval()
    torch.testing.assert_close(model(h, z, baseline=prior), prior, atol=0, rtol=0)
    model(h, z, baseline=prior).square().mean().backward()
    assert model.output.weight.grad.abs().sum() > 0
    with pytest.raises(TypeError):
        model(h, z, baseline=prior, target=prior)


def test_refinement_control_uses_four_deterministic_steps_and_matched_weights():
    h, z, prior = inputs()
    torch.manual_seed(19)
    direct = make_decoder('prior_residual', 8, 1, 32, 4, 2)
    torch.manual_seed(19)
    model = make_decoder('prior_refinement', 8, 1, 32, 4, 2)
    assert model.steps == 4 and not model.stochastic
    for key, value in direct.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
    times = []
    def velocity(state, time, current, code):
        times.append(float(time[0]))
        return torch.ones_like(state)
    model.velocity = velocity
    torch.testing.assert_close(model(h, z, baseline=prior), prior+1)
    assert times == [0., .25, .5, .75]


def test_matched_initial_weights_and_flow_source_use_prior_not_current():
    h, z, prior = inputs()
    models = []
    for kind in ('prior_residual', 'prior_flow', 'prior_terminal'):
        torch.manual_seed(19)
        models.append(make_decoder(kind, 8, 1, 32, 4, 2).eval())
    for model in models[1:]:
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], atol=0, rtol=0)
        noise = torch.ones_like(h)
        torch.testing.assert_close(model(h, z, baseline=prior, noise=noise), prior+.1)


def test_flow_training_keeps_original_current_condition():
    h, z, prior = inputs()
    model = make_decoder('prior_flow', 8, 1, 32, 4, 2).eval()
    y, noise = torch.randn_like(h), torch.ones_like(h)
    time = torch.full((3,), .5)
    seen = {}
    def oracle(state, t, current, code):
        seen['state'], seen['current'] = state, current
        return y-(prior+.1*noise)
    model.velocity = oracle
    out = model.training_output(h, z, y, baseline=prior, noise=noise, time=time)
    torch.testing.assert_close(seen['state'], .5*(prior+.1*noise)+.5*y)
    torch.testing.assert_close(seen['current'], h, atol=0, rtol=0)
    torch.testing.assert_close(out.prediction, y)
    assert out.velocity_loss == 0


def test_prior_export_uses_zero_codes_and_evaluator_keeps_each_examples_prior():
    h, z, prior = inputs()
    def decoder(current, code):
        assert torch.count_nonzero(code) == 0
        return 2*current
    torch.testing.assert_close(predict_prior(decoder, h, 4, batch_size=2), 2*h)
    model = make_decoder('prior_residual', 8, 1, 32, 4, 2).eval()
    data = dict(current=h, future=prior, codes=z, baseline=prior,
                rows=[dict(index=i, episode=str(i), dt=1.2) for i in range(3)])
    metrics = evaluate(model, data, batch_size=2)['metrics']
    assert metrics['prior_mse'] == metrics['mse'] == metrics['shuffled_mse'] == 0
    assert metrics['copy_mse'] > 0
