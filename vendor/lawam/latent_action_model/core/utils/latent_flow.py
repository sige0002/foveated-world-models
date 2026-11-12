"""Conditional rectified flow in frozen visual-feature space (not video diffusion)."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from .lam_decoder import AdaLNBlock
from .pos_embs import Fixed2DPositionalEncoding


@dataclass
class FlowTrainingOutput:
    # One-step endpoint estimate uses a target-dependent interpolation. Diagnostic only!
    prediction: torch.Tensor
    velocity_loss: torch.Tensor


class LatentFlowDecoder(nn.Module):
    def __init__(self, context_dim, input_dim, code_dim, num_layers, num_heads,
                 grid_hw, dropout=0.0, ffn_expansion_factor=4,
                 source='current', noise_scale=0.1, steps=8, **kwargs):
        super().__init__()
        if source not in ('current', 'noise'):
            raise ValueError('flow source must be current or noise')
        if not math.isfinite(noise_scale) or noise_scale <= 0:
            raise ValueError('flow noise_scale must be finite and positive')
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError('flow steps must be a positive integer')
        if not kwargs.get('train_in_latent', True):
            raise ValueError('Latent flow requires a frozen feature encoder')
        self.source, self.noise_scale, self.steps = source, noise_scale, steps
        self.tokens = grid_hw[0] * grid_hw[1]
        self.state_proj = nn.Linear(input_dim, context_dim)
        self.current_proj = nn.Linear(input_dim, context_dim)
        self.action_proj = nn.Linear(code_dim, context_dim)
        self.time_proj = nn.Sequential(nn.Linear(3, context_dim), nn.SiLU(),
                                       nn.Linear(context_dim, context_dim))
        self.pos_embed = Fixed2DPositionalEncoding(context_dim, *grid_hw)
        self.blocks = nn.ModuleList([AdaLNBlock(context_dim, num_heads,
                                   ffn_expansion_factor, dropout) for _ in range(num_layers)])
        # Velocities must not be layer-normalized: magnitude is part of the target.
        self.output = nn.Linear(context_dim, input_dim)

    def velocity(self, x, time, features, actions):
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != self.tokens:
            raise ValueError('flow expects [B, 1, grid_height*grid_width, feature_dim]')
        t = time.to(x).reshape(x.shape[0], 1)
        t_embed = torch.cat((t, torch.sin(torch.pi*t), torch.cos(torch.pi*t)), dim=-1)
        condition = self.action_proj(actions.squeeze(1)) + self.time_proj(t_embed)
        hidden = self.pos_embed(self.state_proj(x[:, 0]) + self.current_proj(features[:, 0]))
        for block in self.blocks:
            hidden = block(hidden, condition)
        return self.output(hidden).unsqueeze(1)

    def initial_state(self, features, noise):
        if noise.shape != features.shape:
            raise ValueError('noise and features must have identical shape')
        center = features if self.source == 'current' else torch.zeros_like(features)
        return center + self.noise_scale * noise

    def training_output(self, features, actions, target, *, noise=None, time=None):
        if target.shape != features.shape:
            raise ValueError('target and current features must have identical shape')
        noise = torch.randn_like(features) if noise is None else noise
        time = torch.rand(features.shape[0], device=features.device) if time is None else time
        t = time.to(features).reshape(-1, 1, 1, 1)
        start = self.initial_state(features, noise)
        x = (1-t) * start + t * target
        velocity = self.velocity(x, time, features, actions)
        loss = F.mse_loss(velocity.float(), (target-start).float())
        return FlowTrainingOutput(x + (1-t)*velocity, loss)

    def forward(self, features, actions, *, noise=None, steps=None):
        """Euler sampling; no future target is accepted. Caller controls eval()/grad mode."""
        steps = self.steps if steps is None else steps
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError('sampling steps must be a positive integer')
        noise = torch.randn_like(features) if noise is None else noise
        x = self.initial_state(features, noise)
        for i in range(steps):
            time = torch.full((features.shape[0],), i/steps, device=features.device)
            x = x + self.velocity(x, time, features, actions) / steps
        return x


class DeterministicRefinementDecoder(LatentFlowDecoder):
    """Compute control: same update network, endpoint regression, no stochastic source."""
    def __init__(self, *args, **kwargs):
        if kwargs.get("source", "current") != "current":
            raise ValueError("deterministic refinement must start from current features")
        super().__init__(*args, **kwargs)

    def forward(self, features, actions, *, steps=None):
        return super().forward(features, actions, noise=torch.zeros_like(features), steps=steps)

    def training_output(self, features, actions, target):
        # Target is used only by the outer endpoint loss, never by these updates.
        return self(features, actions)
