"""Matched-capacity prior correction confined to a train-only residual subspace."""
import torch
from torch.nn import functional as F
from research.prior_correction import PriorCorrectionDecoder
from latent_action_model.core.utils.latent_flow import FlowTrainingOutput


class ProjectedPriorDecoder(PriorCorrectionDecoder):
    def __init__(self, *args, projection_rank=128, **kwargs):
        super().__init__(*args, **kwargs)
        channels = self.output.out_features
        rank = min(projection_rank, channels)
        self.register_buffer('projection_basis', torch.eye(channels, rank))

    def set_basis(self, basis):
        if basis.shape != self.projection_basis.shape:
            raise ValueError('Projection basis shape mismatch')
        eye = torch.eye(basis.shape[1], device=basis.device, dtype=basis.dtype)
        if not torch.allclose(basis.T @ basis, eye, atol=2e-5, rtol=2e-5):
            raise ValueError('Projection columns must be orthonormal')
        with torch.no_grad():
            self.projection_basis.copy_(basis)

    def project(self, value):
        return (value @ self.projection_basis) @ self.projection_basis.T

    def start(self, features, baseline, noise):
        if baseline.shape != features.shape or noise.shape != features.shape:
            raise ValueError('Current, baseline and noise shapes differ')
        return baseline + self.noise_scale * self.project(noise)

    def velocity(self, x, time, features, actions):
        return self.project(super().velocity(x, time, features, actions))

    def training_output(self, features, actions, target, *, baseline, noise=None, time=None):
        if target.shape != features.shape:
            raise ValueError('Target/current shapes differ')
        noise = torch.randn_like(features) if noise is None else noise
        time = torch.rand(len(features), device=features.device) if time is None else time
        t = time.to(features).reshape(-1, 1, 1, 1)
        start = self.start(features, baseline, noise)
        endpoint = baseline + self.project(target-baseline)
        state = (1-t)*start + t*endpoint
        velocity = self.velocity(state, time, features, actions)
        loss = F.mse_loss(velocity.float(), (endpoint-start).float())
        return FlowTrainingOutput(state + (1-t)*velocity, loss)
