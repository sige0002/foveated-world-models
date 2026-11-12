"""Raw-conditioned wide readout control, inspired by RAE/FlowWM DDT heads.

This is a small controlled variant, not an exact reproduction of either paper.
The narrow trunk is shared with the pilot; raw state/current bypass that trunk.
"""
import torch
from torch import nn
from latent_action_model.core.utils.latent_flow import LatentFlowDecoder
from latent_action_model.core.utils.lam_decoder import AdaLNBlock


class WideHeadFlowDecoder(LatentFlowDecoder):
    def __init__(self, *args, deterministic=False, **kwargs):
        super().__init__(*args, **kwargs)
        narrow, feature_dim = self.output.in_features, self.output.out_features
        self.stochastic = not deterministic
        self.trunk_to_head = nn.Linear(narrow, feature_dim)
        self.raw_state = nn.Linear(feature_dim, feature_dim)
        self.raw_current = nn.Linear(feature_dim, feature_dim)
        self.head_condition = nn.Linear(narrow, feature_dim)
        self.head_block = AdaLNBlock(feature_dim, 4, 4, 0.0)
        self.output = nn.Linear(feature_dim, feature_dim)

    def velocity(self, x, time, features, actions):
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != self.tokens:
            raise ValueError('wide flow expects [B, 1, tokens, features]')
        t = time.to(x).reshape(x.shape[0], 1)
        t_embed = torch.cat((t, torch.sin(torch.pi*t), torch.cos(torch.pi*t)), dim=-1)
        condition = self.action_proj(actions.squeeze(1)) + self.time_proj(t_embed)
        hidden = self.pos_embed(self.state_proj(x[:, 0]) + self.current_proj(features[:, 0]))
        for block in self.blocks:
            hidden = block(hidden, condition)
        head = self.trunk_to_head(hidden) + self.raw_state(x[:, 0]) + self.raw_current(features[:, 0])
        head = self.head_block(head, self.head_condition(condition))
        return self.output(head).unsqueeze(1)

    def forward(self, features, actions, *, noise=None, steps=None):
        if not self.stochastic:
            if noise is not None:
                raise ValueError('deterministic wide control does not accept sampling noise')
            noise = torch.zeros_like(features)
        return super().forward(features, actions, noise=noise, steps=steps)
