"""Single-backbone decoder with independent zero-anchored layer conditions.

Install after loading the original checkpoint. Full state_dict serialization
includes the frozen backbone; no future features or separate predictor are used.
"""
import torch
from torch import nn


class LayerwiseCondition(nn.Module):
    def __init__(self, projection, layers):
        super().__init__()
        if not isinstance(projection, nn.Linear) or projection.bias is None or layers < 1:
            raise ValueError('Requires an affine action projection and nonempty backbone')
        self.layers = layers
        self.width = projection.out_features
        self.register_buffer('c0', projection.bias.detach().clone())
        with torch.random.fork_rng(devices=[]):
            self.U = nn.Linear(projection.in_features, layers*self.width, bias=False)
        self.U.to(device=projection.weight.device, dtype=projection.weight.dtype)
        nn.init.zeros_(self.U.weight)

    def forward(self, z):
        if z.ndim == 3 and z.shape[1] == 1:
            z = z.squeeze(1)
        if z.ndim != 2:
            raise ValueError('Expected actions [B,D] or [B,1,D]')
        return self.U(z).reshape(z.shape[0], self.layers, self.width) + self.c0

    @torch.no_grad()
    def diagnostics(self, z):
        displacement = self.U(z).double().reshape(-1, self.layers, self.width)
        return {'displacement_rms_by_layer': displacement.square().mean((0, 2)).sqrt().tolist(),
                'stacked_map_rank': int(torch.linalg.matrix_rank(self.U.weight.double()))}


class LayerwiseDecoder(nn.Module):
    """Reuse original modules once; replacement consumes the supplied decoder."""
    def __init__(self, original):
        super().__init__()
        condition = LayerwiseCondition(original.action_in_proj, len(original.dec_layers))
        original.requires_grad_(False)
        self.action_in_proj = condition
        for name in ('dec_layers', 'pos_embed', 'last_ln', 'project_input', 'project_output'):
            setattr(self, name, getattr(original, name))
        self.grid_height = original.grid_height
        self.grid_width = original.grid_width
        self.train_in_latent = original.train_in_latent
        if not self.train_in_latent:
            self.to_pixel = original.to_pixel
        self.eval()

    def forward(self, features, actions):
        conditions = self.action_in_proj(actions)
        x = self.project_input(features)
        if x.ndim == 4 and x.shape[1] == 1:
            x = x.squeeze(1)
        if x.ndim != 3 or x.shape[1] != self.grid_height*self.grid_width:
            raise ValueError('Decoder feature shape does not match its patch grid')
        if x.shape[0] != conditions.shape[0]:
            raise ValueError('Feature/action batch mismatch')
        x = self.pos_embed(x)
        for i, layer in enumerate(self.dec_layers):
            x = layer(x, conditions[:, i])
        x = self.last_ln(self.project_output(x))
        if not self.train_in_latent:
            b, _, d = x.shape
            x = self.to_pixel(x.transpose(1, 2).reshape(b, d, self.grid_height, self.grid_width))
        return x.unsqueeze(1)


def install_layerwise_adapter(decoder):
    return LayerwiseDecoder(decoder)
