from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'vendor/lawam'))
import torch
from research.centered_prior_correction import CenteredPriorCorrectionDecoder
from research.prior_correction import PriorCorrectionDecoder
from research.train_cached_decoders import make_decoder


def make(cls, deterministic=False):
    return cls(context_dim=8, input_dim=32, code_dim=4, num_layers=1,
               num_heads=4, grid_hw=(2,2), dropout=0., ffn_expansion_factor=4,
               source='current', noise_scale=.1, steps=4, deterministic=deterministic)


def test_same_parameters_and_initial_prior():
    torch.manual_seed(4)
    old = make(PriorCorrectionDecoder, True)
    torch.manual_seed(4)
    new = make(CenteredPriorCorrectionDecoder, True)
    for key, value in old.state_dict().items():
        torch.testing.assert_close(value, new.state_dict()[key], atol=0, rtol=0)
    h, z, b = torch.randn(2,1,4,32), torch.randn(2,1,4), torch.randn(2,1,4,32)
    torch.testing.assert_close(new(h,z,baseline=b), b, atol=0, rtol=0)


def test_teacher_coordinate_and_physical_endpoint_agree():
    model = make(CenteredPriorCorrectionDecoder)
    h, b, y, noise = [torch.randn(2,1,4,32) for _ in range(4)]
    z = torch.randn(2,1,4)
    def oracle(state, t, current, code):
        expected = h+.75*.1*noise+.25*(y-b)
        torch.testing.assert_close(state, expected)
        torch.testing.assert_close(current, h)
        return (y-b)-.1*noise
    model.velocity = oracle
    out = model.training_output(h,z,y,baseline=b,noise=noise,time=torch.full((2,),.25))
    torch.testing.assert_close(out.prediction,y)
    assert out.velocity_loss < 1e-12


def test_one_step_source_noise_can_be_removed_using_supplied_current():
    model = make(CenteredPriorCorrectionDecoder)
    h, b, noise = [torch.randn(2,1,4,32) for _ in range(3)]
    z = torch.randn(2,1,4)
    delta = torch.full_like(h,.3)
    model.velocity = lambda state,t,current,code: delta-(state-current)
    torch.testing.assert_close(model(h,z,baseline=b,noise=noise,steps=1),b+delta)


def test_factory_keeps_capacity_and_sampling_controls():
    models = []
    for kind in ('prior_flow', 'prior_centered_flow', 'prior_centered_residual'):
        torch.manual_seed(29)
        models.append(make_decoder(kind,8,1,32,4,2))
    assert models[1].steps == 4 and models[1].stochastic
    assert models[2].steps == 1 and not models[2].stochastic
    for model in models[1:]:
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value,model.state_dict()[key],atol=0,rtol=0)
