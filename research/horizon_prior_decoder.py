"""Separate physical prediction horizon from flow integration time."""
import torch
from torch import nn
from research.prior_correction import PriorCorrectionDecoder


class HorizonPriorDecoder(PriorCorrectionDecoder):
    requires_physical_time = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Preserve the original flow's subsequent batch/noise RNG sequence.
        with torch.random.fork_rng(devices=[]):
            self.horizon_proj = nn.Linear(2, self.action_proj.out_features, bias=False)
        nn.init.zeros_(self.horizon_proj.weight)

    def pack(self, actions, physical_time):
        dt = torch.as_tensor(physical_time, device=actions.device, dtype=actions.dtype)
        if dt.shape != (len(actions),) or not torch.isfinite(dt).all() or (dt < 0).any():
            raise ValueError('Finite nonnegative physical horizon per example required')
        return torch.cat((actions, dt[:, None, None]), dim=-1)

    def forward(self, features, actions, *, physical_time, **kwargs):
        return super().forward(features, self.pack(actions, physical_time), **kwargs)

    def training_output(self, features, actions, target, *, physical_time, **kwargs):
        return super().training_output(features, self.pack(actions, physical_time), target, **kwargs)

    def velocity(self, x, time, features, actions):
        code, dt = actions[..., :-1], actions[:, 0, -1]
        ratio = dt / 1.2
        ratio = torch.where((ratio-1).abs() < 1e-4, torch.ones_like(ratio), ratio)
        physical = torch.stack((ratio-1, ratio.square()-1), dim=-1)
        t = time.to(x).reshape(x.shape[0], 1)
        t_embed = torch.cat((t, torch.sin(torch.pi*t), torch.cos(torch.pi*t)), dim=-1)
        condition = self.action_proj(code.squeeze(1)) + self.time_proj(t_embed) + self.horizon_proj(physical)
        hidden = self.pos_embed(self.state_proj(x[:, 0]) + self.current_proj(features[:, 0]))
        for block in self.blocks:
            hidden = block(hidden, condition)
        head = self.trunk_to_head(hidden) + self.raw_state(x[:, 0]) + self.raw_current(features[:, 0])
        head = self.head_block(head, self.head_condition(condition))
        return self.output(head).unsqueeze(1)
