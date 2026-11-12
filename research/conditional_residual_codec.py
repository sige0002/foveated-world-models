"""PCA-initialized nonlinear codec for prediction residuals, separate from IDM z."""
import torch
from torch import nn


class ConditionalResidualCodec(nn.Module):
    def __init__(self, basis, hidden=256, context_dim=128):
        super().__init__()
        channels, rank = basis.shape
        if not torch.allclose(basis.T @ basis, torch.eye(rank, dtype=basis.dtype), atol=2e-5, rtol=2e-5):
            raise ValueError('Orthonormal PCA initialization required')
        self.register_buffer('basis', basis.clone())
        self.context = nn.Sequential(nn.Linear(2*channels, context_dim), nn.SiLU())
        self.encoder_adapter = nn.Sequential(nn.Linear(channels+context_dim, hidden), nn.SiLU(),
                                             nn.Linear(hidden, rank))
        self.decoder_adapter = nn.Sequential(nn.Linear(rank+context_dim, hidden), nn.SiLU(),
                                             nn.Linear(hidden, channels))
        for layer in (self.encoder_adapter[-1], self.decoder_adapter[-1]):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def condition(self, current, baseline):
        return self.context(torch.cat((current, baseline-current), dim=-1))

    def encode(self, current, baseline, future):
        delta = future-baseline
        return delta @ self.basis + self.encoder_adapter(torch.cat((delta, self.condition(current,baseline)), dim=-1))

    def decode(self, current, baseline, latent):
        # Future features must never be accepted by this inference interface.
        return baseline + latent @ self.basis.T + self.decoder_adapter(torch.cat((latent, self.condition(current,baseline)), dim=-1))

    def forward(self, current, baseline, future):
        return self.decode(current, baseline, self.encode(current,baseline,future))
