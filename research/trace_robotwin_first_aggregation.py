"""Frozen first-aggregation audit; no training and no modifications to active sources.

Use the source LAMEncoder with zero subsequent Transformer layers, load all of
its retained tensors strictly, and invoke its unchanged project/position/cross
attention modules. Compare observations to unobserved forwards and to the saved
full-encoder first-block output. Q/K/V diagnostics are separate recalculations.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import evaluate_robotwin_loss_ablation_fp32_v2 as api
from research.trace_robotwin_layer_covariance import covariance_stats
from research.train_robotwin_stage1 import write_json

GROUP = api.SSD_RUNS / 'robotwin-loss-ablation-b4-20261114'
REFERENCE = GROUP / 'evaluations/final3000-highest-fp32-v2'
LAYER_REFERENCE = GROUP / 'evaluations/layer-covariance64-highest-fp32-20261114'


def weight_spectrum(weight):
    import torch
    w = weight.detach().cpu().double()
    s = torch.linalg.svdvals(w)
    e = s.square()
    p = e / e.sum()
    return {'shape': list(w.shape), 'frobenius_norm': float(w.norm()),
            'largest_singular_value': float(s[0]), 'top_singular_energy_fraction': float(p[0]),
            'participation_ratio': float(1 / p.square().sum()),
            'stable_rank': float(e.sum() / e[0]),
            'entropy_effective_rank': float((-(p * p.clamp_min(1e-300).log()).sum()).exp()),
            'largest_ten_singular_values': s[:10].tolist()}


def geometry(value):
    """Rows always denote examples, never independent visual tokens."""
    x = value.flatten(1)
    result = {'flattened_sample': covariance_stats(x), 'tensor_shape': list(value.shape)}
    if value.ndim == 3 and value.shape[1] > 1:
        result['token_mean_per_sample'] = covariance_stats(value.mean(1))
    return result


def load_frontend(source, request, config, checkpoint):
    import torch
    from latent_action_model.core.utils.lam_encoder import LAMEncoder
    mc = config
    encoder = LAMEncoder(context_dim=mc['dim'], input_dim=768, num_layers=0,
        num_heads=mc['num_heads'], dropout=mc['dropout'], code_dim=mc['code_dim'],
        num_queries=mc['num_queries'], ffn_expansion_factor=mc['ffn_expansion_factor'],
        grid_hw=tuple(int(v / mc['patch_size']) for v in mc['image_hw']),
        num_frames=mc['num_frames'], modal_mask=mc['enc_modal_mask'],
        add_state=mc['enc_add_state'], max_state_dim=mc['max_state_dim'],
        num_embodiments=mc['num_embodiments'])
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    state = {k.removeprefix('lam.encoder.'): v for k, v in saved['state_dict'].items()
             if k.startswith('lam.encoder.') and '.layers.' not in k}
    assert len(state) == 17
    encoder.load_state_dict(state, strict=True)
    encoder.eval().requires_grad_(False)
    assert len(encoder.QFormer.layers) == 0 and not encoder.add_state
    assert encoder.QFormer.num_queries == 1
    del saved, state
    return encoder


def weight_report(encoder):
    import torch
    block = encoder.QFormer.q_cross_attn
    q, k, v = block.attn.in_proj_weight.chunk(3)
    weights = {'project_in': encoder.project_in.weight, 'attention_Q': q,
               'attention_K': k, 'attention_V': v,
               'attention_out': block.attn.out_proj.weight,
               'ffn_linear1': block.ff[0].weight, 'ffn_linear2': block.ff[2].weight}
    return {'spectra': {name: weight_spectrum(w) for name, w in weights.items()},
            'learned_query_norm': float(encoder.QFormer.queries.norm()),
            'layernorm': {name: {'gamma_min': float(mod.weight.min()), 'gamma_max': float(mod.weight.max()),
                'gamma_mean': float(mod.weight.mean()), 'beta_norm': float(mod.bias.norm()), 'eps': mod.eps}
                for name, mod in [('context_norm1', block.norm1), ('query_norm2', block.norm2)]},
            'cuda_initialized': torch.cuda.is_initialized()}


def prefix_forward(encoder, features):
    projected = encoder.project_in(features)
    context = encoder.pos_embed(projected)
    b, _, _, d = context.shape
    query = encoder.QFormer.queries.expand(b, -1, -1)
    return encoder.QFormer.q_cross_attn(query, context.reshape(b, -1, d))


@contextmanager
def observe(encoder, capture):
    block = encoder.QFormer.q_cross_attn
    named = [('project_in', encoder.project_in), ('position', encoder.pos_embed),
             ('context_norm1', block.norm1), ('attention', block.attn),
             ('query_norm2', block.norm2), ('ffn_linear1', block.ff[0]),
             ('ffn_gelu', block.ff[1]), ('ffn_linear2', block.ff[2]),
             ('ffn', block.ff), ('block', block)]
    originals = []
    try:
        for name, module in named:
            assert 'forward' not in module.__dict__
            assert not module._forward_hooks and not module._forward_pre_hooks
            original = module.forward
            originals.append(module)
            def wrapped(*args, _name=name, _original=original, **kwargs):
                output = _original(*args, **kwargs)
                assert _name not in capture
                capture[_name] = {'input': args[0].detach().clone(),
                                 'output': (output[0] if isinstance(output, tuple) else output).detach().clone()}
                if _name == 'attention':
                    capture[_name]['weights'] = output[1].detach().clone()
                return output
            module.forward = wrapped
        yield
    finally:
        for module in originals:
            del module.forward


def attention_diagnostics(encoder, capture):
    """Separate functional replay of current installed PyTorch need_weights path."""
    import torch
    from torch.nn import functional as F
    attn = encoder.QFormer.q_cross_attn.attn
    assert attn.batch_first and attn.dropout == 0 and attn._qkv_same_embed_dim
    assert attn.bias_k is None and attn.bias_v is None and not attn.add_zero_attn
    query = capture['attention']['input'].transpose(0, 1)
    kv = capture['context_norm1']['output'].transpose(0, 1)
    q, k, v = F._in_projection_packed(query, kv, kv, attn.in_proj_weight, attn.in_proj_bias)
    target, batch, dim = q.shape
    heads, hd = attn.num_heads, dim // attn.num_heads
    qh = q.view(target, batch * heads, hd).transpose(0, 1)
    kh = k.view(k.shape[0], batch * heads, hd).transpose(0, 1)
    vh = v.view(v.shape[0], batch * heads, hd).transpose(0, 1)
    logits = torch.bmm(qh * math.sqrt(1.0 / hd), kh.transpose(-2, -1))
    weights = logits.softmax(-1)
    weighted_v = torch.bmm(weights, vh)
    merged = weighted_v.transpose(0, 1).contiguous().view(target * batch, dim)
    output = F.linear(merged, attn.out_proj.weight, attn.out_proj.bias).view(target, batch, dim).transpose(0, 1)
    probabilities = weights.view(batch, heads, target, k.shape[0])
    averaged = probabilities.mean(1)
    difference = float((output - capture['attention']['output']).abs().max())
    weights_difference = float((averaged - capture['attention']['weights']).abs().max())
    torch.testing.assert_close(output, capture['attention']['output'], atol=2e-5, rtol=1e-6)
    torch.testing.assert_close(averaged, capture['attention']['weights'], atol=1e-6, rtol=1e-6)
    return {'Q': q.transpose(0, 1), 'K': k.transpose(0, 1), 'V': v.transpose(0, 1),
            'attention_logits': logits.view(batch, heads, -1),
            'attention_probabilities': probabilities[:, :, 0, :],
            'attention_weighted_V': merged.view(target, batch, dim).transpose(0, 1),
            'manual_output': output}, {'output_max_abs_difference': difference,
                                      'averaged_weights_max_abs_difference': weights_difference}


def trace(encoder, cache, model_name, samples, batch_size):
    import torch
    captured = {}
    parity = []
    previous = torch.load(LAYER_REFERENCE / (model_name + '-layer-activations.pt'),
                          map_location='cpu', weights_only=True)
    for condition in ('real', 'current_only'):
        chunks = {}
        for start in range(0, samples, batch_size):
            stop = min(start + batch_size, samples)
            x = cache['x'][start:stop].cuda()
            y = cache['y'][start:stop].cuda() if condition == 'real' else x
            features, _, _ = api.native_feature_views(x, y)
            reference = prefix_forward(encoder, features)
            observation = {}
            with observe(encoder, observation):
                observed = prefix_forward(encoder, features)
            torch.testing.assert_close(observed, reference, atol=0, rtol=0)
            replay, attn_parity = attention_diagnostics(encoder, observation)
            saved_reference = previous[condition]['cross_attention_query'][start:stop]
            saved_difference = float((observed[:, 0].cpu() - saved_reference).abs().max())
            torch.testing.assert_close(observed[:, 0].cpu(), saved_reference, atol=0, rtol=0)
            values = {'visual_features': features.flatten(1, 2),
                'project_in': observation['project_in']['output'].flatten(1, 2),
                'with_position': observation['position']['output'].flatten(1, 2),
                'context_norm1': observation['context_norm1']['output'],
                'query_before_attention': observation['attention']['input'],
                'attention_Q': replay['Q'], 'attention_K': replay['K'], 'attention_V': replay['V'],
                'attention_weighted_V': replay['attention_weighted_V'],
                'attention_out_projection': observation['attention']['output'],
                'query_plus_attention': observation['query_norm2']['input'],
                'query_norm2': observation['query_norm2']['output'],
                'ffn_linear1': observation['ffn_linear1']['output'],
                'ffn_gelu': observation['ffn_gelu']['output'],
                'ffn_linear2': observation['ffn_linear2']['output'],
                'query_plus_ffn': observed,
                'attention_probabilities': replay['attention_probabilities'],
                'attention_logits': replay['attention_logits']}
            for name, value in values.items():
                chunks.setdefault(name, []).append(value.detach().cpu().clone())
            parity.append({'condition': condition, 'start': start, 'samples': stop - start,
                'observer_max_abs_difference': 0.0, 'saved_full_encoder_max_abs_difference': saved_difference,
                'manual_attention_replay': attn_parity})
            del x, y, features, reference, observed, replay, observation, values
            print(json.dumps({'phase': 'capture', 'model': model_name, 'condition': condition, 'samples': stop}), flush=True)
        captured[condition] = {name: torch.cat(values) for name, values in chunks.items()}
    stats = {}
    for condition, tensors in captured.items():
        stages = {name: geometry(value) for name, value in tensors.items() if name not in ('attention_logits', 'attention_probabilities')}
        probabilities = tensors['attention_probabilities'].double()
        entropy = -(probabilities * probabilities.clamp_min(1e-300).log()).sum(-1)
        maximum = probabilities.max(-1).values
        linear1 = tensors['ffn_linear1'].double()
        phi = torch.exp(-.5 * linear1.square()) / math.sqrt(2 * math.pi)
        derivative = .5 * (1 + torch.erf(linear1 / math.sqrt(2))) + linear1 * phi
        stats[condition] = {'stages': stages, 'attention': {
            'normalized_entropy_mean': float(entropy.mean() / math.log(probabilities.shape[-1])),
            'effective_token_count_mean': float(entropy.exp().mean()),
            'max_probability_mean': float(maximum.mean()), 'max_probability_p99': float(maximum.quantile(.99)),
            'future_frame_mass_mean': float(probabilities[:, :, probabilities.shape[-1] // 2:].sum(-1).mean()),
            'per_head_normalized_entropy': (entropy.mean(0) / math.log(probabilities.shape[-1])).tolist(),
            'per_head_max_probability_mean': maximum.mean(0).tolist()},
            'gelu': {'preactivation_mean': float(linear1.mean()), 'preactivation_std': float(linear1.std()),
                'preactivation_quantiles_01_50_99': torch.quantile(linear1.flatten(), torch.tensor([.01, .5, .99], dtype=torch.float64)).tolist(),
                'fraction_below_minus3': float((linear1 < -3).double().mean()),
                'fraction_abs_derivative_below_0p01': float((derivative.abs() < .01).double().mean())}}
    return captured, {'conditions': stats, 'parity': parity,
        'native_outputs_bitwise_equal_to_unobserved_and_previous_full_encoder': True}


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--samples', type=int, choices=(16, 32, 64), default=16)
    parser.add_argument('--batch-size', type=int, choices=(1, 2, 4), default=4)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    out = args.output.resolve()
    assert not out.exists() and out.is_relative_to(api.SSD_RUNS.resolve())
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(out.parent), '-o', 'TARGET,SOURCE,FSTYPE'], text=True))
    assert mount['filesystems'][0]['target'] == '/media/sadasue/SSD1'
    cfg = yaml.safe_load((GROUP / 'kl_mse/resolved_config.yaml').read_text())
    request = yaml.safe_load((GROUP / 'kl_mse/requested_config.yaml').read_text())
    contract = api.read_json(GROUP / 'kl_mse/data_contract.json')
    reference = api.read_json(REFERENCE / 'results.json')
    source = Path(request['official_source'])
    assert reference['complete'] and api.digest(Path(api.__file__)) == reference['protocol']['script_sha256']
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == contract['source_commit']
    assert hashlib.sha256(subprocess.check_output(['git', '-C', str(source), 'diff', '--binary'])).hexdigest() == contract['source_patch_sha256']
    for relative, expected in reference['protocol']['verified_visual_source_equivalence'].items():
        if ':' not in relative:
            assert api.digest(source / relative) == expected
    os.environ.update(cfg['environment'])
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    sys.path[:0] = [str(source), *request['dependency_roots']]
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    release = Path(reference['protocol']['release'])
    layer_reference = api.read_json(LAYER_REFERENCE / 'results.json')
    assert layer_reference['complete'] and layer_reference['protocol']['samples'] == 64
    out.mkdir()
    protocol = {'samples': args.samples, 'batch_size': args.batch_size,
        'selection': 'fixed first N of the same highest-FP32 512 evaluation pairs',
        'source_commit': contract['source_commit'], 'source_patch_sha256': contract['source_patch_sha256'],
        'script_sha256': api.digest(__file__), 'source': str(source), 'mount': mount,
        'full_encoder_reference': str(LAYER_REFERENCE), 'full_encoder_reference_sha256': api.digest(LAYER_REFERENCE / 'results.json'),
        'scope': 'Frozen first aggregation only; source project_in/position/CrossAttentionBlock invoked unchanged. Later layers not instantiated, no optimizer.',
        'torch_version': torch.__version__}
    write_json(out / 'protocol.json', protocol)
    reports = {}
    for model_name, key in [('official', 'official'), ('combined3000', 'kl_mse')]:
        config = yaml.safe_load((release / 'dino_large_vae.yaml').read_text())['model'] if key == 'official' else cfg['model']
        checkpoint = release / 'checkpoints/pytorch_model.pt' if key == 'official' else GROUP / key / 'checkpoints/final.ckpt'
        assert api.digest(checkpoint) == reference['models'][key]['checkpoint_sha256']
        encoder = load_frontend(source, request, config, checkpoint)
        reports[model_name] = {'checkpoint': str(checkpoint), 'checkpoint_sha256': api.digest(checkpoint),
                               'weight_analysis': weight_report(encoder)}
        write_json(out / 'weights-partial.json', reports)
        del encoder; gc.collect()
    if args.prepare_only:
        assert not torch.cuda.is_initialized()
        write_json(out / 'preparation.json', {'protocol': protocol, 'models': reports,
            'cuda_initialized': False, 'estimated_gpu_peak_bytes': 1024**3,
            'estimate_basis': 'Only retained 17 encoder tensors (~32MiB), B4 frontend activations; no DINO/24-layer encoder/decoder. Bound to confirm in frozen16 preflight.'})
        print(json.dumps({'prepared': True, 'output': str(out), 'cuda_initialized': False}), flush=True)
        return
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision('highest')
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
    assert available >= 8 * 1024**3 and torch.cuda.is_available()
    protocol.update(host_available_before=available, cuda_free_before=torch.cuda.mem_get_info()[0],
        gpu_before=subprocess.check_output(['nvidia-smi'], text=True),
        precision='highest FP32 cached DINO and frontend; no autocast; TF32 disabled; CPUfloat64 spectra')
    cache_path = Path(reference['protocol']['cache'])
    assert api.digest(cache_path) == api.read_json(REFERENCE / 'feature-cache-manifest.json')['sha256']
    cache = torch.load(cache_path, weights_only=True, map_location='cpu', mmap=True)
    assert cache['contract'] == contract and cache['precision'] == 'highest FP32; TF32 disabled'
    assert cache['rows'][:64] == layer_reference['protocol']['rows']
    protocol.update(feature_cache=str(cache_path), feature_cache_sha256=api.digest(cache_path), rows=cache['rows'][:args.samples])
    write_json(out / 'protocol.json', protocol)
    start = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for model_name, key in [('official', 'official'), ('combined3000', 'kl_mse')]:
            config = yaml.safe_load((release / 'dino_large_vae.yaml').read_text())['model'] if key == 'official' else cfg['model']
            encoder = load_frontend(source, request, config, Path(reports[model_name]['checkpoint']))
            before = {name: api.tensor_hash(tensor) for name, tensor in encoder.state_dict().items()}
            encoder.cuda()
            captured, result = trace(encoder, cache, model_name, args.samples, args.batch_size)
            after = {name: api.tensor_hash(tensor) for name, tensor in encoder.state_dict().items()}
            assert before == after and all(p.grad is None for p in encoder.parameters())
            reports[model_name].update(result, weight_hashes_before=before, weight_hashes_after=after)
            torch.save(captured, out / (model_name + '-activations.pt'))
            write_json(out / (model_name + '.json'), reports[model_name])
            print(json.dumps({'phase': 'complete_model', 'model': model_name,
                'top_pc_by_stage': {name: value['flattened_sample']['top_pc_fraction'] for name, value in result['conditions']['real']['stages'].items()}}), flush=True)
            del captured, encoder; gc.collect(); torch.cuda.empty_cache()
    write_json(out / 'results.json', {'complete': True, 'protocol': protocol, 'models': reports,
        'elapsed_seconds': time.monotonic() - start, 'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(),
        'limitations': ['Same fixed subset, not independent episodes or a training-history measurement.',
            'Token stages flatten token coordinates per example; pooled-token summaries are separately labeled.',
            'Independent QKV replay is checked against actual native MHA outputs; it does not replace the observed model forward.',
            'Concentrated covariance does not by itself establish parameter rank collapse, semantic dimensions, or historical training cause.']})
    print(json.dumps({'complete': True, 'output': str(out)}), flush=True)


if __name__ == '__main__':
    main()
