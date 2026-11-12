"""Research control: clean residual endpoint with an analytic raw-state skip.

This is a known denoiser parameterization (JiT/JLT), not a novel method.
Endpoint MSE is (1-t)^2-weighted velocity MSE for the linear bridge;
therefore this control changes both readout and time weighting.
"""
import torch
from torch.nn import functional as F

from latent_action_model.core.utils.latent_flow import LatentFlowDecoder, FlowTrainingOutput


class EndpointFlowDecoder(LatentFlowDecoder):
    def __init__(self, *args, source_time_probability=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        if not 0 <= source_time_probability <= 1:
            raise ValueError('source_time_probability must be in [0, 1]')
        self.source_time_probability = source_time_probability

    def endpoint(self, x, time, features, actions):
        return features + super().velocity(x, time, features, actions)

    def velocity(self, x, time, features, actions):
        remaining = 1 - time.to(x).reshape(-1, 1, 1, 1)
        if torch.any(remaining <= 0):
            raise ValueError('Endpoint velocity is defined only for time < 1')
        return (self.endpoint(x, time, features, actions) - x) / remaining

    def training_output(self, features, actions, target, *, noise=None, time=None):
        if target.shape != features.shape:
            raise ValueError('target and current features must have identical shape')
        noise = torch.randn_like(features) if noise is None else noise
        if time is None:
            time = torch.rand(features.shape[0], device=features.device)
            if self.source_time_probability:
                at_source = torch.rand_like(time) < self.source_time_probability
                time = torch.where(at_source, 0., time)
        t = time.to(features).reshape(-1, 1, 1, 1)
        start = self.initial_state(features, noise)
        x = (1-t)*start + t*target
        prediction = self.endpoint(x, time, features, actions)
        # Existing result container's field denotes the training objective here.
        return FlowTrainingOutput(prediction, F.mse_loss(prediction.float(), target.float()))
