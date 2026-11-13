"""Trainable reference-difference IDM and a directly conditioned decoder.

Q(current,future) and Q(current,current) share all encoder weights. Neither
branch is detached. Vision features may be cached while Q remains trainable.
The deterministic pilot predicts a 32D posterior mean; no sampling/KL loss.
"""
import torch
from torch import nn


class DifferenceIDM(nn.Module):
    def __init__(self, encoder, head, scale):
        super().__init__()
        scale = torch.as_tensor(scale).detach().clone().float()
        if scale.numel() != 1 or not torch.isfinite(scale).all() or scale.item() <= 0:
            raise ValueError('Expected a positive finite train-only scalar scale')
        if encoder.add_state:
            raise ValueError('This pilot supports visual features only')
        self.encoder = encoder
        self.head = head
        self.register_buffer('scale', scale.reshape(()))

    def forward(self, current, future, *, return_queries=False):
        if current.shape != future.shape or current.ndim != 4 or current.shape[1] != 1:
            raise ValueError('Expected matching current/future [B,1,tokens,features]')
        transition = self.encoder(torch.cat((current, future), dim=1))
        reference = self.encoder(torch.cat((current, current), dim=1))
        z = self.head((transition-reference)/self.scale)
        if return_queries:
            return z, transition, reference
        return z


class DifferenceWorldModel(nn.Module):
    def __init__(self, idm, decoder):
        super().__init__()
        self.idm = idm
        self.decoder = decoder

    def predict(self, current, z):
        """Prediction path consumes current features and an explicitly given z."""
        return self.decoder(current, z)

    def forward(self, current, future):
        z = self.idm(current, future)
        return self.predict(current, z), z
