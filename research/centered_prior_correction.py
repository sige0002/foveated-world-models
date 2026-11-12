"""Residual coordinates around a frozen prior, with an explicit clean source.

For physical state x, the network sees u=h+(x-b(h)). Its clean source is h,
already provided as a conditioning input. It need not reproduce the frozen
large predictor b(h) merely to identify the injected source noise. This is a
residual-diffusion parameterization control, not a novel method claim.
"""
from research.prior_correction import PriorCorrectionDecoder
from latent_action_model.core.utils.latent_flow import FlowTrainingOutput


class CenteredPriorCorrectionDecoder(PriorCorrectionDecoder):
    def forward(self, features, actions, *, baseline, noise=None, steps=None):
        if baseline.shape != features.shape:
            raise ValueError('Baseline and current shapes must match')
        coordinate_prediction = super().forward(features, actions, baseline=features,
                                                 noise=noise, steps=steps)
        return baseline + (coordinate_prediction-features)

    def training_output(self, features, actions, target, *, baseline, noise=None, time=None):
        if baseline.shape != features.shape or target.shape != features.shape:
            raise ValueError('Baseline, target and current shapes must match')
        coordinate_target = features + (target-baseline)
        out = super().training_output(features, actions, coordinate_target,
                                      baseline=features, noise=noise, time=time)
        return FlowTrainingOutput(baseline+(out.prediction-features), out.velocity_loss)
