"""Gradients at the released and local latent distributions, without updates.

Uses real cached image features, shared Gaussian draws, and frozen weights.
This is a derivative probe at fine-tuning entry, not fine-tuning itself.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

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
    req = yaml.safe_load((args.run / 'requested_config.yaml').read_text())
    cfg = yaml.safe_load((args.run / 'resolved_config.yaml').read_text())
    source = Path(req['official_source']).parent / 'LaWAM'
    os.environ.update(cfg['environment'])
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    sys.path[:0] = [str(source), *req['dependency_roots']]
    commit = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    assert commit == req['official_commit']
    assert not subprocess.check_output(['git', '-C', str(source), 'diff', '--binary'])
    import torch
    from torch.nn import functional as F
    from latent_action_model.core.utils.lam_encoder import LAMEncoder
    from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
    from latent_action_model.core.vq import VAEQuantizer
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(142)
    cache_path = args.run / 'evaluations/z-mechanism-20261114/features.pt'
    cache = torch.load(cache_path, weights_only=True, map_location='cpu')
    x, y = cache['x'][:32], cache['y'][:32]
    gaussian = torch.randn(4, 32, 1, 32, generator=torch.Generator().manual_seed(81073))
    release = args.run.parents[1] / 'models/lawam-lam-bd993da'
    model_reports = {}
    start_time = time.monotonic()
    source_hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
        source / 'latent_action_model/core/utils/lam_encoder.py',
        source / 'latent_action_model/core/utils/lam_decoder.py',
        source / 'latent_action_model/core/utils/modules.py',
        source / 'latent_action_model/core/vq.py']}

    def fingerprint(modules):
        h = hashlib.sha256()
        count = 0
        for name, module in modules.items():
            for key, value in module.state_dict().items():
                h.update((name + '.' + key).encode() + b'\0')
                h.update(str(tuple(value.shape)).encode() + b'\0')
                h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
                count += 1
        return h.hexdigest(), count

    for kind in ['official', 'local']:
        checkpoint = release / 'checkpoints/pytorch_model.pt' if kind == 'official' else args.run / 'checkpoints/last.ckpt'
        mc = yaml.safe_load((release / 'dino_large_vae.yaml').read_text())['model'] if kind == 'official' else cfg['model']
        common = dict(context_dim=mc['dim'], input_dim=x.shape[-1], num_heads=mc['num_heads'],
                      dropout=mc['dropout'], code_dim=mc['code_dim'], num_queries=mc['num_queries'],
                      ffn_expansion_factor=mc['ffn_expansion_factor'],
                      grid_hw=tuple(int(v / mc['patch_size']) for v in mc['image_hw']))
        modules = {
            'encoder': LAMEncoder(**common, num_layers=mc['enc_layers'], num_frames=mc['num_frames'],
                                  modal_mask=mc['enc_modal_mask'], add_state=mc['enc_add_state'],
                                  max_state_dim=mc['max_state_dim'], num_embodiments=mc['num_embodiments']),
            'vq': VAEQuantizer(code_dim=mc['code_dim'], **mc['vq_kwargs']),
            'decoder': LAMDecoder_v2(**common, num_layers=mc['dec_layers']),
        }
        saved = torch.load(checkpoint, weights_only=True, map_location='cpu', mmap=True)
        if kind == 'local':
            assert saved['global_step'] == 10000
        for name, module in modules.items():
            prefix = 'lam.' + name + '.'
            state = {key[len(prefix):]: value for key, value in saved['state_dict'].items() if key.startswith(prefix)}
            module.load_state_dict(state, strict=True)
            module.eval().requires_grad_(False)
        del state, saved
        before_hash, tensor_count = fingerprint(modules)
        assert tensor_count == 439
        for module in modules.values():
            module.cuda()
        encoder, vq, decoder = (modules[k] for k in ['encoder', 'vq', 'decoder'])
        records = []
        posterior = []
        for precision, samples, draws in [('native_bf16_medium', 32, 4), ('highest_fp32', 8, 1)]:
            native = precision == 'native_bf16_medium'
            torch.set_float32_matmul_precision('medium' if native else 'highest')
            amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16) if native else nullcontext()
            for offset in range(0, samples, 4):
                sl = slice(offset, offset + 4)
                xx, yy = x[sl].cuda(), y[sl].cuda()
                with torch.no_grad(), amp():
                    frozen_nodes = encoder(torch.cat([xx, yy], 1).float())
                for path, draw in [('sampled', d) for d in range(draws)] + [('mu_only', -1)]:
                    nodes = frozen_nodes.float().detach().requires_grad_(True)
                    # Official VAE executes outside mixed precision.
                    with torch.autocast('cuda', enabled=False):
                        mu, lv = vq._encode(nodes)
                        sigma = (.5 * lv).exp()
                        noise = gaussian[draw, sl].cuda() if draw >= 0 else torch.zeros_like(mu)
                        z = mu + sigma * noise if draw >= 0 else mu
                        kl_rows = .5 * (mu.square() + lv.exp() - 1 - lv).sum(-1).squeeze(-1)
                    with amp():
                        prediction = decoder(xx, z)
                        image_rows = {
                            'smooth_l1': F.smooth_l1_loss(prediction.float(), yy.float(), beta=.1, reduction='none').flatten(1).mean(1),
                            'mse': (prediction.float() - yy.float()).square().flatten(1).mean(1),
                        }
                    # Differentiate sum_i L_i: each row gives dL_i/dactivation_i.
                    # The actual B4 mean-batch gradient is exactly this divided by4.
                    kl_grad = torch.autograd.grad(kl_rows.sum(), [mu, lv, nodes], retain_graph=True)
                    if draw == 0:
                        for i in range(4):
                            posterior.append({'precision': precision, 'row_index': offset+i,
                                'mu': mu[i].detach().flatten().tolist(), 'sigma': sigma[i].detach().flatten().tolist(),
                                'sampled_z_first_draw': z[i].detach().flatten().tolist()})
                    for loss_name, image in image_rows.items():
                        image_grad = torch.autograd.grad(image.sum(), [z, mu, lv, nodes], retain_graph=True, allow_unused=True)
                        for i in range(4):
                            g = [torch.zeros_like(lv[i]) if value is None else value[i] for value in image_grad]
                            row = {'precision': precision, 'path': path, 'draw': draw, 'loss': loss_name,
                                'row_index': offset+i, **cache['rows'][offset+i],
                                'image_loss': float(image[i].detach()), 'unweighted_kl': float(kl_rows[i].detach()),
                                'z_norm': float(z[i].detach().norm()), 'mu_norm': float(mu[i].detach().norm()),
                                'sigma_norm': float(sigma[i].detach().norm()),
                                'image_dz_norm_per_example': float(g[0].norm()),
                                'image_dz_sigma_scaled_norm_per_example': float((g[0]*sigma[i].detach()).norm()),
                                'image_dmu_norm_per_example': float(g[1].norm()),
                                'image_dlogvar_norm_per_example': float(g[2].norm()),
                                'image_dnodes_norm_per_example': float(g[3].norm()),
                                'image_batch_mean_dz_full_batch_norm': float(image_grad[0].norm()/4),
                                'image_dlogvar_connected': image_grad[2] is not None,
                                'kl_comparisons': {}}
                            for beta in [5e-5, 1e-5]:
                                comp = {}
                                for name, image_g, kl_g in zip(['mu','logvar','nodes'], g[1:], kl_grad):
                                    k = kl_g[i] * beta
                                    denom = image_g.norm() * k.norm()
                                    comp[name] = {'weighted_kl_gradient_norm': float(k.norm()),
                                        'kl_over_image_gradient_norm': float(k.norm()/image_g.norm()) if float(image_g.norm()) else None,
                                        'cosine': float((image_g*k).sum()/denom) if float(denom) else None,
                                        'combined_gradient_norm': float((image_g+k).norm())}
                                row['kl_comparisons'][str(beta)] = comp
                            records.append(row)
                    del prediction, image_rows, image_grad, kl_grad, nodes, mu, lv, sigma, z, kl_rows
                if offset % 8 == 0:
                    print(json.dumps({'phase':'gradient_probe','model':kind,'precision':precision,'pairs_completed':offset+4}),flush=True)
        assert all(p.grad is None for module in modules.values() for p in module.parameters())
        after_hash, _ = fingerprint(modules)
        assert before_hash == after_hash
        summaries = []
        for precision in ['native_bf16_medium','highest_fp32']:
            for path in ['sampled','mu_only']:
                for loss in ['smooth_l1','mse']:
                    rs = [r for r in records if (r['precision'],r['path'],r['loss'])==(precision,path,loss)]
                    keys = ['image_loss','unweighted_kl','z_norm','mu_norm','sigma_norm',
                        'image_dz_norm_per_example','image_dz_sigma_scaled_norm_per_example','image_dmu_norm_per_example',
                        'image_dlogvar_norm_per_example','image_dnodes_norm_per_example']
                    summary = {'precision':precision,'path':path,'loss':loss,'sample_draw_count':len(rs),
                        **{key:sum(r[key] for r in rs)/len(rs) for key in keys},'kl_comparisons':{}}
                    for beta in ['5e-05','1e-05']:
                        summary['kl_comparisons'][beta] = {}
                        for level in ['mu','logvar','nodes']:
                            summary['kl_comparisons'][beta][level] = {}
                            for field in ['weighted_kl_gradient_norm','kl_over_image_gradient_norm','cosine','combined_gradient_norm']:
                                values=[r['kl_comparisons'][beta][level][field] for r in rs if r['kl_comparisons'][beta][level][field] is not None]
                                summary['kl_comparisons'][beta][level][field] = sum(values)/len(values) if values else None
                    summaries.append(summary)
        model_reports[kind] = {'checkpoint':str(checkpoint),'strictly_loaded_state_tensors':tensor_count,
            'state_sha256_before':before_hash,'state_sha256_after':after_hash,'unchanged':True,
            'summary':summaries,'rows':records,'posterior_first_draw':posterior}
        write_json(args.output.with_name(args.output.stem+'-partial.json'), {'complete':False,'reports':model_reports})
        print(json.dumps({'phase':'model_complete','model':kind,'summary':summaries}),flush=True)
        del encoder,vq,decoder,module,modules
        torch.cuda.empty_cache()
    report = {'complete':True,'reports':model_reports,'source_commit':commit,'source_hashes':source_hashes,
        'source':str(source),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'epsilon_sha256':hashlib.sha256(gaussian.numpy().tobytes()).hexdigest(),
        'feature_cache':str(cache_path),'elapsed_seconds':time.monotonic()-start_time,
        'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'scope':'Frozen released/local checkpoint activation gradients; no optimizer or parameter updates. Same deterministic validation features, not augmented training batches. Per-example gradients differentiate a feature-mean loss per example; actual mean-batch gradients divide by B4. Raw norms depend on latent coordinates; sigma-scaled norms additionally reported. KL depends on mu/logvar, not sampled z directly. g omitted because official14/local16 state contracts are not established as compatible. Highest-FP32 subset8 and1draw only; native32pairs4draws.'}
    write_json(args.output, report)
    summary_path = ROOT / 'research/results/robotwin-z-underuse-20261114/released-z-gradients-summary.json'
    write_json(summary_path, {**{k:v for k,v in report.items() if k!='reports'},
        'reports':{kind:{k:v for k,v in value.items() if k not in {'rows','posterior_first_draw'}} for kind,value in model_reports.items()}})


if __name__ == '__main__':
    main()
