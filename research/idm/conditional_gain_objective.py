"""Proper auxiliary grounding loss; baseline and scale must be train-only/cross-fitted.

No claim of causal identification. Positive gain can reflect baseline error or
conditionally correlated nuisance. This module does not fit the baseline.
"""
import torch

def grounded_gain(prediction, observed_increment, baseline_increment, scale):
    if prediction.shape!=observed_increment.shape or baseline_increment.shape!=observed_increment.shape:
        raise ValueError('Prediction/target/baseline shapes must match')
    if not torch.isfinite(scale).all() or (scale<=0).any():raise ValueError('Positive finite train scale required')
    target=(observed_increment.detach()-baseline_increment.detach())/scale.detach()
    risk=(prediction-target).square().mean()
    null_risk=target.square().mean()
    return dict(loss=risk,diagnostic_gain=(null_risk-risk).detach(),target=target)
