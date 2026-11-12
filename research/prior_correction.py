"""Strong frozen current-only prior plus learned full-width correction; research control."""
import torch
from torch import nn
from torch.nn import functional as F
from research.wide_head_flow import WideHeadFlowDecoder
from latent_action_model.core.utils.latent_flow import FlowTrainingOutput


class PriorCorrectionDecoder(WideHeadFlowDecoder):
    requires_baseline = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Deterministic output equals the strong prior before the first update.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def start(self, features, baseline, noise):
        if baseline.shape != features.shape or noise.shape != features.shape:
            raise ValueError('Current, baseline and noise must have identical shapes')
        return baseline + self.noise_scale * noise

    def forward(self, features, actions, *, baseline, noise=None, steps=None):
        steps = self.steps if steps is None else steps
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError('Positive integer steps required')
        if not self.stochastic:
            if noise is not None:
                raise ValueError('Deterministic prior correction does not accept noise')
            noise = torch.zeros_like(features)
        elif noise is None:
            noise = torch.randn_like(features)
        state = self.start(features, baseline, noise)
        for step in range(steps):
            time = features.new_full((len(features),), step / steps)
            state = state + self.velocity(state, time, features, actions) / steps
        return state

    def training_output(self, features, actions, target, *, baseline, noise=None, time=None):
        if target.shape != features.shape:
            raise ValueError('Target/current shapes differ')
        noise = torch.randn_like(features) if noise is None else noise
        time = torch.rand(len(features), device=features.device) if time is None else time
        t = time.to(features).reshape(-1, 1, 1, 1)
        start = self.start(features, baseline, noise)
        state = (1-t)*start + t*target
        velocity = self.velocity(state, time, features, actions)
        loss = F.mse_loss(velocity.float(), (target-start).float())
        return FlowTrainingOutput(state + (1-t)*velocity, loss)
