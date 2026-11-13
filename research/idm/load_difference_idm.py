"""Load a self-contained trained difference-IDM/direct-decoder checkpoint.

model(current_features, future_features) performs posterior reconstruction.
model.predict(current_features, z) consumes an explicitly supplied latent code;
this does not supply a policy/action-to-z model for executable forecasting.
"""
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'vendor/lawam')]
import torch
from torch import nn
from latent_action_model.core.utils.lam_encoder import LAMEncoder
from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
from research.idm.difference_idm import DifferenceIDM,DifferenceWorldModel


def load_trained_model(path,device='cpu'):
    saved=torch.load(path,weights_only=True,map_location='cpu',mmap=True)
    protocol=saved['protocol']
    cfg=protocol['options']['encoder']
    if cfg['enc_add_state'] or cfg['code_dim']!=32:
        raise ValueError('Unsupported checkpoint configuration')
    encoder=LAMEncoder(context_dim=cfg['dim'],input_dim=768,num_heads=cfg['num_heads'],
        num_queries=cfg['num_queries'],grid_hw=tuple(v//cfg['patch_size'] for v in cfg['image_hw']),
        ffn_expansion_factor=cfg['ffn_expansion_factor'],dropout=cfg['dropout'],code_dim=cfg['dim'],
        num_layers=cfg['enc_layers'],num_frames=cfg['num_frames'],modal_mask=cfg['enc_modal_mask'],
        add_state=False,max_state_dim=cfg['max_state_dim'],num_embodiments=cfg['num_embodiments'])
    head=nn.Sequential(nn.Linear(cfg['dim'],256),nn.GELU(),nn.Linear(256,32))
    decoder=LAMDecoder_v2(**protocol['options']['decoder'])
    model=DifferenceWorldModel(DifferenceIDM(encoder,head,protocol['scale'],
        representation=protocol.get('representation','difference')),decoder)
    model.load_state_dict(saved['model'],strict=True)
    return model.to(device).eval().requires_grad_(False),protocol
