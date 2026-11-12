"""Change only the wide output block's FFN expansion, retaining 768D mixing."""
import torch
from torch import nn
from research.train_cached_decoders import make_decoder


def make_slim_decoder(expansion, seed):
    if expansion not in (2,4):
        raise ValueError('Prespecified expansions: 2 or 4')
    torch.manual_seed(seed)
    model=make_decoder('prior_residual',128,2,768,32,16)
    if expansion==2:
        # Keep every shared tensor and the global RNG identical to expansion4.
        # Fresh native initialization for the differently shaped FFN is intentional.
        with torch.random.fork_rng(devices=[]):
            model.head_block.ffn=nn.Sequential(nn.Linear(768,1536),nn.GELU(),nn.Dropout(0.),
                                               nn.Linear(1536,768),nn.Dropout(0.))
    return model
