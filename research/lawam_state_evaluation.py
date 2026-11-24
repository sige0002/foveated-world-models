"""Variable-dimensional state evaluation; padding never contributes to error."""
import torch


def pad_states(states, mask=None, *, max_state_dim=32):
    if states.ndim != 3 or states.shape[1] != 2:
        raise ValueError('states must have shape [B,2,D]')
    dim = states.shape[-1]
    if not 1 <= dim <= max_state_dim:
        raise ValueError(f'state dimension {dim} exceeds supported range 1..{max_state_dim}')
    if mask is None:
        mask = torch.ones_like(states, dtype=torch.bool)
    elif mask.ndim == 2:
        mask = mask[:, None, :].expand_as(states)
    if mask.shape != states.shape:
        raise ValueError('mask must have shape [B,D] or [B,2,D]')
    mask = mask.to(device=states.device, dtype=torch.bool)
    valid = mask[:, 0] & mask[:, 1]
    if not valid.any(-1).all():
        raise ValueError('Each sample must have at least one valid state component')
    padded = torch.nn.functional.pad(states.masked_fill(~mask, 0), (0, max_state_dim-dim))
    valid = torch.nn.functional.pad(valid, (0, max_state_dim-dim))
    delta = (padded[:, 1] - padded[:, 0]).masked_fill(~valid, 0)
    return padded, delta, valid


def masked_state_mse(prediction, target, mask):
    if prediction.ndim == 3 and prediction.shape[1] == 1:
        prediction = prediction[:, 0]
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError('Prediction, target, mask must match [B,D]')
    valid = mask.bool()
    count = valid.sum(-1)
    if not (count > 0).all():
        raise ValueError('Each sample must have at least one valid state component')
    error = (prediction.float()-target.float()).masked_fill(~valid, 0).square()
    return error.sum(-1)/count
