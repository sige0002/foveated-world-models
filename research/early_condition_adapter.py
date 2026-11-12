"""Zero-anchored condition map for a frozen, pretrained LAMDecoder_v2."""
import hashlib
import torch
from torch import nn


def state_hash(state):
    h = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        t = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(t.dtype).encode())
        h.update(str(tuple(t.shape)).encode()); h.update(t.numpy().tobytes())
    return h.hexdigest()


class ConditionAdapter(nn.Module):
    def __init__(self, original):
        super().__init__()
        if not isinstance(original, nn.Linear) or original.bias is None:
            raise ValueError('Expected the trained affine action projection')
        self.register_buffer('c0', original.bias.detach().clone())
        self.register_buffer('old_weight', original.weight.detach().clone())
        # Constructing zero U must not consume the caller's random stream.
        with torch.random.fork_rng(devices=[]):
            self.U = nn.Linear(original.in_features, original.out_features, bias=False)
        self.U.to(device=original.weight.device, dtype=original.weight.dtype)
        nn.init.zeros_(self.U.weight)
        left, values, _ = torch.linalg.svd(self.old_weight.double(), full_matrices=False)
        tolerance = max(self.old_weight.shape) * torch.finfo(torch.float64).eps * values.max()
        self.register_buffer('old_basis', left[:, values > tolerance])
        self.register_buffer('rank_tolerance', tolerance)

    def forward(self, z):
        return self.c0 + self.U(z)

    @torch.no_grad()
    def diagnostics(self, z):
        displacement = self.U(z).double().reshape(-1, self.U.out_features)
        projected = (displacement @ self.old_basis) @ self.old_basis.T
        energy = displacement.square().sum()
        outside = (displacement-projected).square().sum()
        return {'displacement_rms': displacement.square().mean().sqrt().item(),
                'outside_old_subspace_energy_fraction': (outside/energy).item() if energy > 0 else 0.,
                'old_rank': self.old_basis.shape[1], 'rank_tolerance': self.rank_tolerance.item(),
                'U_singular_values': torch.linalg.svdvals(self.U.weight.double()).tolist()}


def install_adapter(decoder):
    decoder.requires_grad_(False)
    decoder.action_in_proj = ConditionAdapter(decoder.action_in_proj)
    decoder.eval()
    return decoder


def frozen_hash(decoder):
    return state_hash({k:v for k,v in decoder.state_dict().items()
                       if k != 'action_in_proj.U.weight'})
