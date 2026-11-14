"""Check highest-FP32 encoder and FP64 posterior head on cached RoboTwin features."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.diagnose_robotwin_z import geometry
from research.train_robotwin_stage1 import write_json


def main():
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--diagnostics', type=Path, required=True)
    a = p.parse_args()
    req = yaml.safe_load((a.run / 'requested_config.yaml').read_text())
    cfg = yaml.safe_load((a.run / 'resolved_config.yaml').read_text())
    os.environ.update(cfg['environment'])
    sys.path[:0] = [str(Path(req['official_source']).parent / 'LaWAM'), *req['dependency_roots']]
    import torch
    from latent_action_model.core.utils.lam_encoder import LAMEncoder
    from latent_action_model.core.vq import VAEQuantizer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    cache = torch.load(a.diagnostics / 'features.pt', weights_only=True)
    reports = {}
    for kind in ['local', 'official']:
        started = time.monotonic()
        old = json.loads((a.diagnostics / kind / 'results.json').read_text())
        previous = torch.load(a.diagnostics / kind / 'tensors.pt', weights_only=True)['encoding']
        mc = cfg['model']
        encoder = LAMEncoder(context_dim=mc['dim'], input_dim=cache['x'].shape[-1],
            num_layers=mc['enc_layers'], num_heads=mc['num_heads'], ffn_expansion_factor=mc['ffn_expansion_factor'],
            dropout=mc['dropout'], num_frames=mc['num_frames'], num_queries=mc['num_queries'],
            grid_hw=tuple(i // mc['patch_size'] for i in mc['image_hw']), patch_size=mc['patch_size'],
            add_state=False, modal_mask=mc['enc_modal_mask'], code_dim=mc['code_dim'])
        vae = VAEQuantizer(code_dim=mc['code_dim'], **mc['vq_kwargs'])
        saved = torch.load(old['checkpoint'], weights_only=True, map_location='cpu', mmap=True)['state_dict']
        for name, module in [('encoder', encoder), ('vq', vae)]:
            prefix = f'lam.{name}.'
            module.load_state_dict({k.removeprefix(prefix): v for k, v in saved.items() if k.startswith(prefix)}, strict=True)
            module.eval().requires_grad_(False).cuda()
        del saved
        vae64 = copy.deepcopy(vae).double()
        values = {}
        with torch.no_grad():
            for mode in ['real', 'current']:
                mu32, mu64, logvars = [], [], []
                future = cache['y'] if mode == 'real' else cache['x']
                for start in range(0, len(future), 4):
                    sl = slice(start, start + 4)
                    nodes = encoder(torch.cat([cache['x'][sl], future[sl]], 1).float().cuda())
                    m32, _ = vae._encode(nodes)
                    m64, l64 = vae64._encode(nodes.double())
                    mu32.append(m32.cpu())
                    mu64.append(m64.cpu())
                    logvars.append(l64.cpu())
                values[mode] = dict(mu32=torch.cat(mu32), mu64=torch.cat(mu64), logvar=torch.cat(logvars))
        m = values['real']['mu64'].flatten(1)
        l = values['real']['logvar'].flatten(1)
        var = l.exp()
        centered = m - m.mean(0)
        covariance = centered.T @ centered / len(m) + torch.diag(var.mean(0))
        sign, logdet = torch.linalg.slogdet(covariance)
        assert sign > 0
        signal = float(centered.square().sum() / len(m))
        noise = float(var.sum(1).mean())
        report = dict(samples=len(m), precision=torch.get_float32_matmul_precision(),
            mu_geometry=geometry(values['real']['mu64']),
            current_replacement_distance=float((values['real']['mu64'] - values['current']['mu64']).flatten(1).norm(dim=1).mean()),
            fp32_vs_fp64_head_max_error=float((values['real']['mu32'].double() - values['real']['mu64']).abs().max()),
            medium_vs_highest_mu_max_error=float((previous['fp32']['mu'].double() - values['real']['mu64']).abs().max()),
            signal_variance_sum=signal, noise_variance_sum=noise,
            noise_fraction=noise / (signal + noise),
            empirical_mixture_information_upper_bound_nats=float(.5 * (logdet - l.sum(1).mean())),
            elapsed_seconds=time.monotonic() - started,
            scope='Same cached DINO features; encoder highest FP32 without autocast, posterior head FP64. Cached input precision is unchanged.')
        reports[kind] = report
        torch.save(values, a.diagnostics / f'{kind}-highest-precision.pt')
        print(json.dumps({kind: report}), flush=True)
        del encoder, vae, vae64
        torch.cuda.empty_cache()
    write_json(a.diagnostics / 'precision-check.json', dict(complete=True, reports=reports,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
