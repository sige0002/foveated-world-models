"""CPU-only, frozen local g residual decomposition on the existing FP32 512 pairs.

Strictly load only the ten saved state-decoder tensors. Check all five intervention
rows against the preceding native-verified full-model evaluation before analysis.
The input is posterior mean mu, not sampled training z; official g is not used.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
GROUP = Path('/media/sadasue/SSD1/foveated-world-models/runs/robotwin-loss-ablation-b4-20261114')
SOURCE = Path('/media/sadasue/SSD1/foveated-world-models/research/official-data-contract-audit-20261113/LaWAM-robotwin-delta2')
GROUPS = {'xyz': [0, 1, 2, 8, 9, 10], 'quaternion': [3, 4, 5, 6, 11, 12, 13, 14], 'gripper': [7, 15]}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def tensor_hash(x):
    x = x.detach().cpu().contiguous()
    h = hashlib.sha256(str(tuple(x.shape)).encode() + b'\0' + str(x.dtype).encode() + b'\0')
    h.update(x.reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def module_hash(module):
    h = hashlib.sha256()
    for name, value in module.state_dict().items():
        h.update(name.encode() + b'\0' + bytes.fromhex(tensor_hash(value)))
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--arm', choices=['baseline', 'kl_mse'], default='kl_mse')
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    started = time.monotonic()
    torch.set_num_threads(2)
    torch.set_float32_matmul_precision('highest')
    sys.path.insert(0, str(SOURCE))
    from latent_action_model.core.utils.lam_decoder import StatePredictor

    evaluation = GROUP / 'evaluations/final3000-highest-fp32-v2'
    report_path = evaluation / args.arm / 'results.json'
    code_path = evaluation / args.arm / 'posterior-codes.pt'
    cache_path = GROUP / 'evaluations/final3000-highest-fp32/features512-highest-fp32.pt'
    report = json.loads(report_path.read_text())
    assert report['complete'] and report['unchanged'] and report['global_step'] == 3000
    checkpoint = GROUP / args.arm / 'checkpoints/final.ckpt'
    # mmap does not construct/touch the full encoder, visual decoder or Adam states.
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False, mmap=True)
    assert saved['global_step'] == 3000
    hp = saved['hyper_parameters']
    assert hp['state_output_scale'] == 2 and hp['lambda_aux'] == 1 and hp['state_loss_type'] == 'l2'
    state = {name.removeprefix('lam.state_decoder.'): value for name, value in saved['state_dict'].items()
             if name.startswith('lam.state_decoder.')}
    assert len(state) == 10
    with torch.device('meta'):
        model = StatePredictor(latent_dim=hp['dim'], dropout=hp['dropout'],
                               num_embodiments=hp['num_embodiments'], num_queries=hp['num_queries'],
                               max_state_dim=hp['max_state_dim'], code_dim=hp['code_dim'], output_scale=2)
    model.load_state_dict(state, strict=True, assign=True)
    model.eval().requires_grad_(False)
    before = module_hash(model)
    assert before == report['module_hashes_before']['state_decoder']['sha256']
    cache = torch.load(cache_path, map_location='cpu', weights_only=False, mmap=True)
    codes = torch.load(code_path, map_location='cpu', weights_only=False)
    assert codes['row_identity_sha256'] == cache['reference_identity_sha256']
    assert len(cache['rows']) == len(report['rows']) == 512
    for left, right in zip(cache['rows'], report['rows']):
        assert all(left[k] == right[k] for k in ('episode', 'base', 'donor_episode', 'donor_base', 'donor_index'))
    assert torch.equal(cache['delta'], cache['states'][:, 1] - cache['states'][:, 0])
    assert cache['mask'][:, :16].all() and not cache['mask'][:, 16:].any()
    mu = codes['mu']
    donors = torch.tensor([row['donor_index'] for row in cache['rows']])
    variants = {'real': mu, 'zero': torch.zeros_like(mu), 'mean': mu.mean(0, keepdim=True).expand_as(mu),
                'swapped': mu[donors], 'current_only': codes['current_only_mu']}
    predictions, parity = {}, {}
    with torch.inference_mode():
        for name, z in variants.items():
            pred = torch.cat([model(z[i:i+2], cache['states'][i:i+2, :1], cache['emb'][i:i+2])
                              for i in range(0, 512, 2)])[:, :16]
            predictions[name] = pred
            mse = (pred - cache['delta'][:, :16]).square().mean(1)
            expected = torch.tensor([row['metrics'][name + '_state_mse'] for row in report['rows']])
            torch.testing.assert_close(mse, expected, rtol=0, atol=1e-6)
            parity[name] = {'samples': 512, 'max_abs_row_mse_difference': float((mse - expected).abs().max()),
                            'atol': 1e-6, 'rtol': 0, 'passed': True}
            print(json.dumps({'phase': 'matched_prior_state_mse', 'variant': name, **parity[name]}), flush=True)
    predictions['no_change'] = torch.zeros_like(predictions['real'])
    gradients = {group: [] for group in GROUPS}
    for i in range(0, 512, 2):
        leaf_mu = mu[i:i+2].detach().clone().requires_grad_(True)
        pred = model(leaf_mu, cache['states'][i:i+2, :1], cache['emb'][i:i+2])[:, :16]
        error = (pred - cache['delta'][i:i+2, :16]).square()
        parts = {group: error[:, indices].sum(1).div(16).mean() for group, indices in GROUPS.items()}
        torch.testing.assert_close(sum(parts.values()), error.mean())
        for group, loss in parts.items():
            # Each loss is the group's contribution to the all16 batch mean.
            # Undo 1/B to compare true per-example dL_i/dmu, not raw batch gradients.
            grad = torch.autograd.grad(loss, leaf_mu, retain_graph=True)[0].detach() * len(leaf_mu)
            assert torch.isfinite(grad).all()
            gradients[group].append(grad)
    gradients = {group: torch.cat(values).flatten(1) for group, values in gradients.items()}
    assert all(parameter.grad is None for parameter in model.parameters())
    episodes = np.array([row['episode'] for row in cache['rows']])
    unique, inverse = np.unique(episodes, return_inverse=True)
    counts = np.bincount(inverse)
    # Same paired episode resamples for every comparison, keeping variable cluster size.
    draws = np.random.default_rng(20261114).integers(0, len(unique), (4000, len(unique)))
    denominators = counts[draws].sum(1)

    def estimate(values):
        values = np.asarray(values, dtype=np.float64)
        sums = np.bincount(inverse, weights=values, minlength=len(unique))
        means = sums[draws].sum(1) / denominators
        return {'estimate': float(values.mean()), 'ci95': np.quantile(means, [.025, .975]).tolist()}

    errors = {name: (pred.double() - cache['delta'][:, :16].double()).square().numpy()
              for name, pred in predictions.items()}
    summaries, contrasts = {}, {}
    for name, error in errors.items():
        total = error.mean(1)
        summaries[name] = {'all16_mse': estimate(total), 'groups': {}}
        for group, indices in GROUPS.items():
            contribution = error[:, indices].sum(1) / 16
            summaries[name]['groups'][group] = {
                'dimensions': indices, 'per_coordinate_mse': estimate(error[:, indices].mean(1)),
                'contribution_to_all16_mse': estimate(contribution),
                'fraction_of_total_error': float(contribution.mean() / total.mean())}
        if name != 'real':
            diff = error - errors['real']
            contrasts[name + '_minus_real'] = {'all16_mse': estimate(diff.mean(1)), 'groups': {
                group: {'per_coordinate_mse': estimate(diff[:, indices].mean(1)),
                        'contribution_to_all16_mse': estimate(diff[:, indices].sum(1) / 16)}
                for group, indices in GROUPS.items()}}
    after = module_hash(model)
    assert before == after and not torch.cuda.is_initialized()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.save({'predictions': predictions, 'group_per_example_mu_gradients': gradients,
                'target': cache['delta'][:, :16].clone(),
                'rows': cache['rows'], 'row_identity_sha256': cache['reference_identity_sha256']},
               args.output / 'predictions.pt')
    result = {'complete': True, 'arm': args.arm, 'device': 'cpu', 'cuda_initialized': False,
              'samples': 512, 'episodes': len(unique), 'precision': 'FP32, highest',
              'state_decoder_sha256_before': before, 'state_decoder_sha256_after': after,
              'source_sha256': digest(__file__), 'decoder_source_sha256': digest(SOURCE / 'latent_action_model/core/utils/lam_decoder.py'),
              'preceding_report_sha256': digest(report_path), 'posterior_codes_sha256': digest(code_path),
              'feature_cache_sha256': digest(cache_path),
              'row_identity_sha256': cache['reference_identity_sha256'], 'strictly_loaded_tensors': 10,
              'prior_native_verified_full_model_parity': parity,
              'summaries': summaries, 'control_minus_real': contrasts,
              'real_mu_group_gradients': {
                  'definition': 'Per-example derivative of sum of group squared residuals / 16, all groups equally normalized; frozen g, real posterior mean, no parameter backward or update.',
                  'groups': {group: {'per_example_norm': estimate(grad.double().norm(dim=1).numpy()),
                                     'aggregate_gradient_norm': float(grad.double().mean(0).norm())}
                             for group, grad in gradients.items()},
                  'pairwise_cosines_mean': {a + '__' + b: float(torch.nn.functional.cosine_similarity(gradients[a].double(), gradients[b].double(), dim=1).mean())
                                            for a, b in [('xyz', 'quaternion'), ('xyz', 'gripper'), ('quaternion', 'gripper')]},
                  'total_per_example_norm': estimate(sum(gradients.values()).double().norm(dim=1).numpy())},
              'limitations': ['Posterior mean mu at saved step3000 on validation inputs, not sampled training z or training gradients.',
                              'Local g only. Official released g has a different state contract and is not evaluated.',
                              'Error share does not establish which loss component caused encoder concentration.',
                              'Quaternion values use the existing normalized component-MSE target, not an SO(3) distance.',
                              'Full checkpoint file is mmap-read; exact ten loaded tensor hash is bound to completed evaluation.'],
              'elapsed_seconds': time.monotonic() - started}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'complete': True, 'output': str(args.output / 'results.json'), 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
