"""Standalone cached-feature evaluator with per-sample valid state masks.

Derived from our research evaluator, not upstream LaWAM. The previous
experiment evaluator is retained unchanged for reproducibility.
"""
import json
from research.evaluate_robotwin_loss_ablation_fp32_v2 import native_feature_views
from research.lawam_state_evaluation import masked_state_mse

def evaluate_model(model, cache, batch_size, output, *, official):
    import torch
    mu_rows, lv_rows, node_rows, current_mu_rows, current_lv_rows = [], [], [], [], []
    for start in range(0, len(cache['rows']), batch_size):
        sl = slice(start, start + batch_size)
        x, y = cache['x'][sl].cuda(), cache['y'][sl].cuda()
        encoder_input, x, y = native_feature_views(x, y)
        nodes = model.encoder(encoder_input)
        mu, lv = model.vq._encode(nodes)
        current_encoder_input, _, _ = native_feature_views(x, x)
        current_nodes = model.encoder(current_encoder_input)
        current_mu, current_lv = model.vq._encode(current_nodes)
        mu_rows.append(mu.cpu()); lv_rows.append(lv.cpu()); node_rows.append(nodes.cpu())
        current_mu_rows.append(current_mu.cpu()); current_lv_rows.append(current_lv.cpu())
        if (start + len(x)) % 128 == 0:
            print(json.dumps({'phase': 'FP32_encode', 'model': output.name, 'samples': start + len(x)}), flush=True)
    mu, lv, nodes = map(torch.cat, (mu_rows, lv_rows, node_rows))
    current_mu, current_lv = map(torch.cat, (current_mu_rows, current_lv_rows))
    assert mu.dtype == lv.dtype == nodes.dtype == torch.float32
    mean = mu.mean(0, keepdim=True)
    donors = torch.tensor([row['donor_index'] for row in cache['rows']])
    variants = {'real': mu, 'zero': torch.zeros_like(mu), 'mean': mean.expand_as(mu),
                'swapped': mu[donors], 'current_only': current_mu}
    rows = [dict(row, metrics={}) for row in cache['rows']]
    for start in range(0, len(cache['rows']), batch_size):
        sl = slice(start, start + batch_size)
        x, y, states, delta, emb, top = [cache[key][sl].cuda() for key in
                                        ('x', 'y', 'states', 'delta', 'emb', 'high_change_patch_indices')]
        _, x, y = native_feature_views(x, y)
        real_pred = None
        for name, code in variants.items():
            z = code[sl].cuda()
            pred = model.decoder(x, z)
            if name == 'real':
                real_pred = pred
            err = (pred - y).square()
            metrics = {'visual_mse': err.flatten(1).mean(1),
                       'high_change_patch_mse': err.mean(-1).squeeze(1).gather(1, top).mean(1),
                       'output_difference_mse': (pred - real_pred).square().flatten(1).mean(1)}
            if not official:
                sp = model.state_decoder(z, states[:, :1], emb)
                metrics['state_mse'] = masked_state_mse(sp, delta, cache['mask'][sl].to(delta.device))
                for group, columns in cache.get('state_groups', {}).items():
                    metrics[group+'_state_mse'] = masked_state_mse(sp[:, columns], delta[:, columns], cache['mask'][sl][:, columns].to(delta.device))
            for key, value in metrics.items():
                assert torch.isfinite(value).all(), (output.name, name, key)
                for j, scalar in enumerate(value.cpu().tolist()):
                    rows[start + j]['metrics'][name + '_' + key] = scalar
        persistence = (x - y).square()
        for j in range(len(x)):
            rows[start + j]['metrics'].update(
                persistence_visual_mse=float(persistence[j].mean()),
                persistence_high_change_patch_mse=float(persistence[j].mean(-1).squeeze(0)[top[j]].mean()))
            if not official:
                rows[start + j]['metrics']['persistence_state_mse'] = float(masked_state_mse(torch.zeros_like(delta[j:j+1]), delta[j:j+1], cache['mask'][start+j:start+j+1].to(delta.device))[0])
        if (start + len(x)) % 128 == 0:
            print(json.dumps({'phase': 'FP32_interventions', 'model': output.name, 'samples': start + len(x)}), flush=True)
    output.mkdir(parents=True, exist_ok=True)
    torch.save({'mu': mu, 'logvar': lv, 'nodes': nodes, 'global512_mean': mean,
                'current_only_mu': current_mu, 'current_only_logvar': current_lv,
                'row_identity_sha256': cache['reference_identity_sha256']}, output / 'posterior-codes.pt')
    signal = float(mu.double().flatten(1).var(0, unbiased=False).sum())
    noise = float(lv.double().exp().flatten(1).sum(1).mean())
    geometry = {'mu_variance_sum': signal, 'noise_variance_sum': noise,
                'noise_fraction': noise / (noise + signal),
                'pre_norm_node_coordinate_std_mean': float(nodes.double().flatten(1).std(-1, unbiased=False).mean()),
                'mean_code_norm': float(mu.double().flatten(1).norm(dim=1).mean()),
                'mean_distance_to_current_only_mu': float((mu - current_mu).double().flatten(1).norm(dim=1).mean())}
    return {'rows': rows, 'geometry': geometry}

