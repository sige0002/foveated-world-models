"""Current-feature spatial tangent branch with a Gram-matched rotated control."""
import torch
from torch import nn
from research.prior_correction import PriorCorrectionDecoder


class SpatialTangentDecoder(PriorCorrectionDecoder):
    def __init__(self, *args, rotated=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.grid_hw = tuple(kwargs['grid_hw'])
        self.rotated = rotated
        channels = self.output.out_features
        self.motion_coeff = nn.Linear(channels, 2)
        nn.init.zeros_(self.motion_coeff.weight)
        nn.init.zeros_(self.motion_coeff.bias)
        generator = torch.Generator().manual_seed(20261113)
        self.register_buffer('channel_permutation', torch.randperm(channels, generator=generator))
        self.register_buffer('channel_sign', 2 * torch.randint(2, (channels,), generator=generator) - 1)

    def spatial_basis(self, features):
        b, one, tokens, channels = features.shape
        if one != 1 or tokens != self.tokens:
            raise ValueError('Expected one current feature grid')
        h, w = self.grid_hw
        grid = features[:, 0].reshape(b, h, w, channels)
        basis = grid.new_zeros(b, h, w, channels, 2)
        basis[:, 1:-1, 1:-1, :, 0] = (grid[:, 1:-1, 2:] - grid[:, 1:-1, :-2]) / 2
        basis[:, 1:-1, 1:-1, :, 1] = (grid[:, 2:, 1:-1] - grid[:, :-2, 1:-1]) / 2
        if self.rotated:
            basis = basis[..., self.channel_permutation, :] * self.channel_sign.to(basis)[None, None, None, :, None]
        return basis.reshape(b, tokens, channels, 2)

    def velocity(self, x, time, features, actions):
        if x.ndim != 4 or x.shape[1] != 1 or x.shape[2] != self.tokens:
            raise ValueError('Expected [batch, 1, tokens, features]')
        t = time.to(x).reshape(x.shape[0], 1)
        t_embed = torch.cat((t, torch.sin(torch.pi*t), torch.cos(torch.pi*t)), dim=-1)
        condition = self.action_proj(actions.squeeze(1)) + self.time_proj(t_embed)
        hidden = self.pos_embed(self.state_proj(x[:, 0]) + self.current_proj(features[:, 0]))
        for block in self.blocks:
            hidden = block(hidden, condition)
        head = self.trunk_to_head(hidden) + self.raw_state(x[:, 0]) + self.raw_current(features[:, 0])
        head = self.head_block(head, self.head_condition(condition))
        coeff = 0.5 * torch.tanh(self.motion_coeff(head))
        correction = torch.einsum('bpdi,bpi->bpd', self.spatial_basis(features), coeff)
        return (self.output(head) + correction).unsqueeze(1)
