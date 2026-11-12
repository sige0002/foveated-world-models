"""Identical fresh Gaussian heads for all34D postprojection controls."""
from copy import deepcopy
from torch import nn
import torch
from research.idm.postprojection_representations import VARIANTS
def shared_initial_heads(seed=42,hidden_dim=256,code_dim=32):
 # Preserve caller RNG state; representation choice cannot alter initialization.
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed)
  initial=nn.Sequential(nn.Linear(34,hidden_dim),nn.GELU(),nn.Linear(hidden_dim,2*code_dim))
 return {variant:deepcopy(initial) for variant in VARIANTS}
