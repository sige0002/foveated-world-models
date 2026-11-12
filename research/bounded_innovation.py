"""A fixed-rate sampled innovation channel; no deterministic-mean information claim."""
import math
import torch
from torch import nn


class ExpandedActionProjection(nn.Module):
    """40D affine map evaluated as preserved32 + zero-initialized8 columns.

    Split evaluation preserves the original GEMM at initialization; a fused40D
    GEMM can change floating-point rounding even with eight zero columns.
    """
    def __init__(self, original, extra_dim=8):
        super().__init__()
        self.original=original
        with torch.random.fork_rng(devices=[]):
            self.extra=nn.Linear(extra_dim,original.out_features,bias=False)
        self.extra.to(original.weight)
        nn.init.zeros_(self.extra.weight)
        self.old_dim=original.in_features

    def forward(self,z):
        return self.original(z[...,:self.old_dim].contiguous())+self.extra(z[...,self.old_dim:].contiguous())


class InnovationModel(nn.Module):
    def __init__(self,decoder,arm,seed=42,input_dim=2080,width=128,extra_dim=8,delta=1.):
        super().__init__()
        if arm not in ('future','recoding','noise'): raise ValueError(arm)
        self.decoder=decoder
        self.decoder.action_proj=ExpandedActionProjection(decoder.action_proj,extra_dim)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.head=nn.Sequential(nn.Linear(input_dim,width),nn.GELU(),nn.Linear(width,extra_dim))
        # Deliberately NONZERO standard head initialization: zero head + zero new
        # projection would make initial new-column gradients noise-only in expectation.
        self.arm=arm;self.delta=delta;self.extra_dim=extra_dim
        if arm=='noise': self.head.requires_grad_(False)

    def mean(self,head_input):
        raw=self.head(head_input)
        return torch.zeros_like(raw) if self.arm=='noise' else math.sqrt(2*self.delta/self.extra_dim)*raw.tanh()

    def forward(self,current,z0,head_input,epsilon,*,baseline):
        mean=self.mean(head_input)
        if epsilon.shape!=mean.shape: raise ValueError('One independent draw per example required')
        code=torch.cat((z0,mean+epsilon),dim=-1)
        return self.decoder(current,code,baseline=baseline),mean


def fit_normalization(train):
    return {'mean':train.mean(0,keepdim=True),'scale':train.std(0,unbiased=False,keepdim=True).clamp_min(1e-6)}


def normalize(x,cal):
    return (x-cal['mean'])/cal['scale']


def kl_summary(mean):
    kl=mean.double().square().sum(-1).flatten()/2
    if not torch.isfinite(kl).all() or float(kl.max())>1+1e-6:
        raise RuntimeError('Hard1nat bound or finite invariant failed')
    return {'mean':float(kl.mean()),'max':float(kl.max()),'per_example':kl.tolist()}
