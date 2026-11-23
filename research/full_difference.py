"""RoboTwin Stage 1 with the same shared visual IDM evaluated on (C,F) and (C,C)."""
import torch
from torch.utils.checkpoint import checkpoint
from latent_action_model.core.lam_lightinng import VJEPA_LAM


def subtract_current_reference(encoder, args, output):
    features = args[0]
    if features.shape[1] != 2 or encoder.add_state:
        raise ValueError("Difference IDM expects two image frames and a visual-only encoder")
    reference = features[:, :1].expand_as(features)
    # Call the underlying forward directly to avoid recursively applying this hook.
    if torch.is_grad_enabled():
        baseline = checkpoint(encoder.forward, reference, use_reentrant=False)
    else:
        baseline = encoder.forward(reference)
    return output - baseline


class DifferenceLAM(VJEPA_LAM):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.lam.encoder.register_forward_hook(subtract_current_reference)
