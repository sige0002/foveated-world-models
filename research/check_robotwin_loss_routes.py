"""Measure each Stage-1 loss's gradient routes on four real validation pairs.

Uses the training VAE forward path, with deterministic image preprocessing.
No optimizer is constructed and no parameter/checkpoint updates are performed.
"""
from pathlib import Path
from functools import partial
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.train_robotwin_stage1 import install_episode_split, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    a = parser.parse_args()
    import yaml
    cfg = yaml.safe_load((a.run / 'resolved_config.yaml').read_text())
    requested = yaml.safe_load((a.run / 'requested_config.yaml').read_text())
    contract = json.loads((a.run / 'data_contract.json').read_text())
    os.environ.update(cfg['environment'])
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    source = Path(requested['official_source'])
    sys.path[:0] = [str(source), *requested['dependency_roots']]
    patch = subprocess.check_output(['git', '-C', str(source), 'diff', '--binary'])
    assert hashlib.sha256(patch).hexdigest() == contract['source_patch_sha256']
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'],
                                   text=True).strip() == contract['source_commit']
    split = json.loads((a.run / 'episode_split.json').read_text())['ids']
    install_episode_split(split)

    import torch
    from torch.nn import functional as F
    from torch.utils.data import DataLoader, Subset
    from latent_action_model.core.lam_lightinng import VJEPA_LAM
    from latent_action_model.core.utils.utils import eef_reconstruction_loss
    from latent_action_model.data_loader.lerobot_dataset import LeRobotLAMDataset
    from latent_action_model.data_loader.collate import lam_collate
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    mc = dict(cfg['model'])
    mc.pop('optimizer')
    model = VJEPA_LAM(**mc).eval()
    checkpoint = a.run / 'checkpoints/last.ckpt'
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    assert saved['global_step'] == 10000 and saved['data_contract'] == contract
    model.load_state_dict(saved['state_dict'], strict=True)
    del saved
    model.cuda()
    model.lam.vision_encoder.requires_grad_(False)
    dc = cfg['data']
    keys = ('data_root_dir', 'data_mix', 'num_frames', 'val_tail_ratio',
            'video_backend', 'image_hw', 'frame_dt_sec', 'human_frame_dt_sec')
    ds = LeRobotLAMDataset(**{k: dc[k] for k in keys if k in dc}, mode='val')
    for single in ds.mixture.datasets:
        for transform in single._lam_transforms_by_video_key.values():
            transform.eval()
    batch = next(iter(DataLoader(Subset(ds, range(4)), batch_size=4, num_workers=0,
                                collate_fn=partial(lam_collate, max_state_dim=32))))
    pairs = list(zip(map(int, batch['trajectory_ids']), map(int, batch['base_indices'])))
    batch = model.transfer_batch_to_device(batch, torch.device('cuda'), 0)
    assert (batch['embodiment_ids'] == 1).all()
    assert torch.equal(batch['delta_proprio'], batch['states'][:, 1] - batch['states'][:, 0])
    captured = {}

    def save_output(name):
        def hook(module, args, output):
            captured[name] = output
        return hook

    hooks = [model.lam.register_forward_hook(save_output('lam')),
             model.lam.encoder.register_forward_hook(save_output('nodes'))]
    with torch.autocast('cuda', dtype=torch.bfloat16):
        total, _ = model._compute_step(batch, vq_training=True)
        out = captured['lam']
        image_loss = F.smooth_l1_loss(out[0], out[2], beta=.1)
        state_loss = model.lambda_aux * eef_reconstruction_loss(
            out[5], batch['delta_proprio'], batch['state_mask'].any(1),
            state_loss_type=model.state_loss_type)
        weighted_kl = out[9]
        reconstructed_total = image_loss + model.lambda_diversity * out[8] + weighted_kl + state_loss
    torch.testing.assert_close(total, reconstructed_total, rtol=0, atol=1e-6)
    for hook in hooks:
        hook.remove()
    groups = {'IDM_encoder': model.lam.encoder, 'VAE': model.lam.vq,
              'visual_decoder': model.lam.decoder, 'g_state_decoder': model.lam.state_decoder}
    named = [(group, name, param) for group, module in groups.items()
             for name, param in module.named_parameters() if param.requires_grad]
    targets = [param for _, _, param in named] + [captured['nodes'], out[7]]
    routes = {}
    losses = {'image_prediction': image_loss, 'g_state_prediction': state_loss,
              'weighted_KL': weighted_kl}
    expected = {'image_prediction': {'IDM_encoder', 'VAE', 'visual_decoder'},
                'g_state_prediction': {'IDM_encoder', 'VAE', 'g_state_decoder'},
                'weighted_KL': {'IDM_encoder', 'VAE'}}
    for i, (name, loss) in enumerate(losses.items()):
        gradients = torch.autograd.grad(loss, targets, allow_unused=True,
                                        retain_graph=i < len(losses) - 1)
        entries = {g: {'gradient_norm_squared': 0., 'connected_parameters': 0,
                       'disconnected_parameters': 0} for g in groups}
        for (group, _, _), grad in zip(named, gradients[:-2]):
            if grad is None:
                entries[group]['disconnected_parameters'] += 1
            else:
                assert torch.isfinite(grad).all()
                entries[group]['connected_parameters'] += 1
                entries[group]['gradient_norm_squared'] += float(grad.float().norm()) ** 2
        for group, entry in entries.items():
            entry['gradient_norm'] = math.sqrt(entry.pop('gradient_norm_squared'))
            assert (entry['gradient_norm'] > 0) == (group in expected[name]), (name, group)
        routes[name] = {'loss': float(loss.detach()), 'modules': entries,
                        'encoder_output_gradient_norm': None if gradients[-2] is None else float(gradients[-2].float().norm()),
                        'sampled_z_gradient_norm': None if gradients[-1] is None else float(gradients[-1].float().norm())}
        print(json.dumps({'loss_route': name, **routes[name]}), flush=True)
        del gradients
    report = {'complete': True, 'checkpoint': str(checkpoint), 'step': 10000,
              'pairs': pairs, 'source': str(source), 'source_commit': contract['source_commit'],
              'source_patch_sha256': contract['source_patch_sha256'],
              'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'mode': 'training VAE path, BF16 mixed, deterministic validation images, no optimizer updates',
              'total_loss': float(total.detach()), 'sum_of_components': float(reconstructed_total.detach()),
              'lambda_aux': model.lambda_aux, 'kl_beta': model.lam.vq.beta,
              'encoder_class': type(model.lam.encoder).__name__,
              'state_predictor_class': type(model.lam.state_decoder).__name__,
              'routes': routes}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(a.output, report)


if __name__ == '__main__':
    main()
