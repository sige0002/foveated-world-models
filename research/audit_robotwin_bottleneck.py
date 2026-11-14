"""Frozen Stage-1 head transmission and loss-gradient directions; no training.

Future-to-current replacement is a dependence probe, not a physical no-op label.
Geometry attributes the current computation, not the historical training cause.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.train_robotwin_stage1 import write_json


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    assert not args.output.exists()
    cfg = yaml.safe_load((args.run / 'resolved_config.yaml').read_text())
    req = yaml.safe_load((args.run / 'requested_config.yaml').read_text())
    os.environ.update(cfg['environment'])
    sys.path[:0] = [req['official_source'], *req['dependency_roots']]
    import torch
    from torch.nn import functional as F
    from latent_action_model.core.lam_lightinng import VJEPA_LAM
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(142)
    diag = args.run / 'evaluations/z-mechanism-20261114'
    cache = torch.load(diag / 'features.pt', weights_only=True)
    states = torch.load(args.run / 'evaluations/idm-state-probe-20261114/codes.pt', weights_only=True)['val']
    for a, b in zip(cache['rows'], states['rows']):
        assert a['episode'] == b['episode'] and a['base'] == b['base']
    x, y = cache['x'], cache['y']
    reports = {}
    for kind in ['local', 'official']:
        old = json.loads((diag / kind / 'results.json').read_text())
        checkpoint = Path(old['checkpoint'])
        mc = dict(cfg['model']) if kind == 'local' else dict(yaml.safe_load((checkpoint.parents[1] / 'dino_large_vae.yaml').read_text())['model'])
        mc.pop('optimizer')
        mc['vision_model_id'] = cfg['model']['vision_model_id']
        model = VJEPA_LAM(**mc).eval().requires_grad_(False)
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)['state_dict']
        expected = model.state_dict()
        for key in list(saved):
            if key not in expected and key.startswith('lam.vision_encoder.model.layer.'):
                new = key.replace('lam.vision_encoder.model.layer.', 'lam.vision_encoder.model.model.layer.', 1)
                assert new in expected
                saved[new] = saved.pop(key)
        model.load_state_dict(saved, strict=True)
        del saved, expected
        model.cuda()
        torch.set_float32_matmul_precision('highest')
        lam = model.lam
        assert not lam.disable_vq and lam.encoder.QFormer.num_queries == 1
        queries = {}
        capture = []
        h = lam.encoder.QFormer.register_forward_hook(lambda m, a, out: capture.append(out.detach().double().cpu()))
        with torch.no_grad():
            for name, future in [('real', y), ('current', x)]:
                capture.clear()
                for start in range(0, len(x), 4):
                    sl = slice(start, start + 4)
                    lam.encoder(torch.cat([x[sl], future[sl]], 1).float().cuda())
                queries[name] = torch.cat(capture).squeeze(1)
        h.remove()
        # Head calculations on the same highest-FP32 queries, in float64.
        w, bias = [p.detach().double().cpu() for p in [lam.encoder.out_proj.weight, lam.encoder.out_proj.bias]]
        gamma, beta = [p.detach().double().cpu() for p in [lam.vq.pre_norm.weight, lam.vq.pre_norm.bias]]
        mw, mb = [p.detach().double().cpu() for p in [lam.vq.mu.weight, lam.vq.mu.bias]]
        lw, lb = [p.detach().double().cpu() for p in [lam.vq.logvar.weight, lam.vq.logvar.bias]]
        eps = lam.vq.pre_norm.eps
        q, qq = queries['real'], queries['current']
        p, pp = F.linear(q, w, bias), F.linear(qq, w, bias)
        dq, dp = qq - q, pp - p
        pc = p - p.mean(-1, keepdim=True)
        dpc = dp - dp.mean(-1, keepdim=True)
        radial = pc * (dpc * pc).sum(-1, keepdim=True) / pc.square().sum(-1, keepdim=True)
        tangent = dpc - radial
        scale = (pc.square().mean(-1, keepdim=True) + eps).sqrt()
        jacobian_delta = gamma * (dpc / scale - pc * (pc * dpc).mean(-1, keepdim=True) / scale.pow(3))
        norm = lambda v: F.layer_norm(v, (32,), gamma, beta, eps)
        hn, hnn = norm(p), norm(pp)
        mu, mm = F.linear(hn, mw, mb), F.linear(hnn, mw, mb)
        lv_raw, ll_raw = F.linear(hn, lw, lb), F.linear(hnn, lw, lb)
        lv, ll = lv_raw.clamp(-10, 10), ll_raw.clamp(-10, 10)
        var, vv = lv.exp(), ll.exp()
        klpq = .5 * ((ll - lv) + (var + (mu-mm).square()) / vv - 1).sum(-1)
        klqp = .5 * ((lv - ll) + (vv + (mu-mm).square()) / var - 1).sum(-1)
        energy = lambda v: float(v.square().sum(-1).mean())
        record = {
            'samples': len(q),
            'query_mean_norm': float(q.norm(dim=-1).mean()),
            'query_future_replacement_energy': energy(dq),
            'projected_mean_norm': float(p.norm(dim=-1).mean()),
            'projected_coordinate_std_mean': float(scale.mean()),
            'projected_future_replacement_energy': energy(dp),
            'projection_transmission_vs_equal_norm_isotropic_expectation': energy(dp) / (energy(dq) * float(w.square().sum()) / w.shape[1]),
            'projected_displacement_coordinate_mean_energy': energy(dp-dpc),
            'projected_displacement_radial_energy': energy(radial),
            'projected_displacement_tangent_energy': energy(tangent),
            'radial_fraction_of_coordinate_centered_displacement': energy(radial) / energy(dpc),
            'layer_norm_affine_gamma_abs_mean': float(gamma.abs().mean()),
            'layer_norm_affine_gamma_abs_min': float(gamma.abs().min()),
            'layer_norm_displacement_energy': energy(hnn-hn),
            'layer_norm_jacobian_relative_squared_error': energy(jacobian_delta-(hnn-hn))/energy(hnn-hn),
            'mu_displacement_energy': energy(mm-mu),
            'mu_displacement_mahalanobis_squared_mean': float(((mm-mu).square()/var).sum(-1).mean()),
            'posterior_symmetric_kl_mean': float(((klpq+klqp)*.5).mean()),
            'unclamped_logvar_min': float(lv_raw.min()),
            'unclamped_logvar_max': float(lv_raw.max()),
            'logvar_clamped_fraction': float((lv_raw.abs()>10).double().mean()),
            'mu_weight_frobenius_norm': float(mw.norm()),
            'out_proj_weight_frobenius_norm': float(w.norm()),
        }
        previous = torch.load(diag / f'{kind}-highest-precision.pt', weights_only=True)
        torch.testing.assert_close(mu.unsqueeze(1), previous['real']['mu64'].double(), rtol=1e-4, atol=2e-6)
        # Change only the perturbation before LN, holding its baseline fixed.
        interventions = {'coordinate_mean_only': dp-dpc, 'radial_only': radial,
                         'tangent_only': tangent, 'full_displacement': dp}
        record['normalization_component_interventions'] = {
            name: {'input_displacement_energy': energy(d),
                   'mu_displacement_energy': energy(F.linear(norm(p+d), mw, mb)-mu)}
            for name, d in interventions.items()}
        if kind == 'local':
            # Differentiate the official loss algebra with the model frozen.
            # This computes gradients only with respect to the leaf activation;
            # no optimizer is constructed and all parameters remain unchanged.
            gradient_rows = []
            for start in range(0, 16, 4):
                sl = slice(start, start+4)
                s = F.pad(states['state'][sl], (0,16)).cuda().unsqueeze(1)
                emb = torch.ones(4, dtype=torch.long, device='cuda')
                for draw in range(4):
                    nodes = p[sl].float().cuda().unsqueeze(1).detach().requires_grad_(True)
                    m, l = lam.vq._encode(nodes)
                    noise = torch.randn_like(m)
                    z = m + (.5*l).exp()*noise
                    pred = lam.decoder(x[sl].float().cuda(), z)
                    g = lam.state_decoder(z, s, emb)[:, :16]
                    losses = {'image': F.smooth_l1_loss(pred, y[sl].float().cuda(), beta=.1),
                              'state': (g-states['delta'][sl].cuda()).square().mean(),
                              'kl': .5*(m.square()+l.exp()-1-l).sum(-1).mean()*lam.vq.beta}
                    grads = {name: torch.autograd.grad(loss, nodes, retain_graph=True)[0].detach().double().cpu() for name,loss in losses.items()}
                    values = {name: float(loss.detach()) for name,loss in losses.items()}
                    norms = {name: float(v.norm()) for name,v in grads.items()}
                    cos = {}
                    for aa,bb in [('image','state'),('image','kl'),('state','kl')]:
                        cos[aa+'__'+bb] = float((grads[aa]*grads[bb]).sum() / (grads[aa].norm()*grads[bb].norm()).clamp_min(1e-30))
                    recon = grads['image']+grads['state']
                    gradient_rows.append({'start':start,'draw':draw,'losses':values,'node_gradient_norms':norms,'pairwise_cosines':cos,
                        'kl_norm_over_image_plus_state': float(grads['kl'].norm()/recon.norm()),
                        'kl_cosine_with_image_plus_state': float((grads['kl']*recon).sum()/(grads['kl'].norm()*recon.norm()).clamp_min(1e-30))})
            record['loss_gradient_directions'] = gradient_rows
            assert all(v.grad is None for v in model.parameters())
        reports[kind] = record
        print(json.dumps({'kind':kind, **{k:v for k,v in record.items() if k!='loss_gradient_directions'}}),flush=True)
        del lam, model
        torch.cuda.empty_cache()
    write_json(args.output, {'complete': True, 'reports':reports,
        'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'Frozen checkpoint, no optimizer or parameter updates. Same-current future-to-current feature replacement; not action-labelled or physical counterfactual. Geometry and loss-gradient directions at current weights do not identify historical training causes.'})


if __name__ == '__main__':
    main()
