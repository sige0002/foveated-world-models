"""Frozen centered layer covariance on the shared highest-FP32 validation cache.

Observe original forwards with temporary instance wrappers, avoiding PyTorch's
forward-hook fastpath exclusion. No training or checkpoint writes occur.
"""
from contextlib import contextmanager
from copy import deepcopy
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import evaluate_robotwin_loss_ablation_fp32_v2 as api
from research.train_robotwin_stage1 import write_json

GROUP = api.SSD_RUNS / 'robotwin-loss-ablation-b4-20261114'
REFERENCE = GROUP / 'evaluations/final3000-highest-fp32-v2'


def covariance_stats(values):
    import torch
    x = values.double().cpu()
    assert x.ndim == 2 and len(x) >= 2 and torch.isfinite(x).all()
    centered = x - x.mean(0, keepdim=True)
    eigenvalues = torch.linalg.eigvalsh(centered @ centered.T / (len(x) - 1)).flip(0).clamp_min(0)
    total = float(eigenvalues.sum())
    energy = float(x.square().sum(-1).mean())
    common = float(x.mean(0).square().sum())
    return {'samples': len(x), 'features': x.shape[1], 'sample_dimension_rank_ceiling': min(len(x) - 1, x.shape[1]),
        'centered_covariance_eigenvalues': eigenvalues.tolist(), 'centered_trace': total,
        'top_pc_fraction': float(eigenvalues[0]) / total if total > 0 else None,
        'participation_ratio': total**2 / float(eigenvalues.square().sum()) if total > 0 else None,
        'components_for_90pct': int(torch.searchsorted(eigenvalues.cumsum(0), .9 * eigenvalues.sum())) + 1 if total > 0 else 0,
        'eigenvalues_above_relative_1e_10': int((eigenvalues > eigenvalues[0] * 1e-10).sum()) if total > 0 else 0,
        'uncentered_mean_squared_norm': energy, 'common_mean_squared_norm': common,
        'between_example_mean_squared_norm': float(centered.square().sum(-1).mean()),
        'between_example_fraction_of_uncentered_energy': float(centered.square().sum(-1).mean()) / energy if energy else None}


@contextmanager
def observe_forwards(named_modules, capture):
    """Leave module hook dictionaries and native fused-forward eligibility intact."""
    originals = []
    try:
        for name, module in named_modules:
            assert 'forward' not in module.__dict__
            original = module.forward
            originals.append(module)
            def wrapped(*args, _name=name, _original=original, **kwargs):
                output = _original(*args, **kwargs)
                # Clone only the final query, not the full context-token storage.
                assert _name not in capture
                capture[_name] = output[:, -1, :].detach().clone()
                return output
            module.forward = wrapped
        yield
    finally:
        for module in originals:
            del module.forward


def trace_model(model, cache, samples, batch_size):
    import torch
    names = [('cross_attention_query', model.encoder.QFormer.q_cross_attn)]
    names += [(f'layer_{i+1:02d}_query', layer) for i, layer in enumerate(model.encoder.QFormer.layers)]
    names += [('encoder_out_proj', model.encoder.out_proj), ('vae_pre_norm', model.vq.pre_norm), ('posterior_mu', model.vq.mu)]
    assert model.encoder.QFormer.num_queries == 1 and not model.encoder.add_state
    assert all(not m._forward_hooks and not m._forward_pre_hooks for _, m in names)
    assert torch.backends.mha.get_fastpath_enabled()
    captured = {}
    parity = []
    with torch.inference_mode():
        # Full public route parity is separate from observer/no-observer parity.
        video = cache['parity_videos'].cuda()
        native = model.get_latent_action(video, None, video,
            embodiment_ids=torch.ones(len(video), device='cuda', dtype=torch.long), predict_future_frame=False)
        features, _, _ = api.native_feature_views(cache['x'][:len(video)].cuda(), cache['y'][:len(video)].cuda())
        native_mu, _ = model.vq._encode(model.encoder(features))
        native_difference = float((native_mu - native['quantized']).abs().max())
        torch.testing.assert_close(native_mu, native['quantized'], atol=1e-6, rtol=0)
        del native, native_mu, video, features
        for condition in ('real', 'current_only'):
            chunks = {name: [] for name, _ in names}
            for offset in range(0, samples, batch_size):
                sl = slice(offset, min(offset + batch_size, samples))
                x = cache['x'][sl].cuda()
                y = cache['y'][sl].cuda() if condition == 'real' else x
                features, _, _ = api.native_feature_views(x, y)
                reference_nodes = model.encoder(features)
                reference_mu, _ = model.vq._encode(reference_nodes)
                observed = {}
                with observe_forwards(names, observed):
                    nodes = model.encoder(features)
                    mu, _ = model.vq._encode(nodes)
                assert list(observed) == list(chunks)
                node_difference = float((nodes - reference_nodes).abs().max())
                mu_difference = float((mu - reference_mu).abs().max())
                # Fail closed: these wrappers must not change the actual forward.
                torch.testing.assert_close(nodes, reference_nodes, atol=0, rtol=0)
                torch.testing.assert_close(mu, reference_mu, atol=0, rtol=0)
                torch.testing.assert_close(observed['posterior_mu'], mu[:, -1], atol=0, rtol=0)
                parity.append({'condition': condition, 'offset': offset, 'samples': len(x),
                               'nodes_max_abs_difference': node_difference, 'mu_max_abs_difference': mu_difference})
                # One CPU transfer after the full original computation.
                widths = [observed[name].shape[-1] for name in chunks]
                joined = torch.cat([observed[name] for name in chunks], dim=-1).cpu()
                for (name, _), value in zip(names, joined.split(widths, dim=-1)):
                    chunks[name].append(value.clone())
                del x, y, features, nodes, mu, reference_nodes, reference_mu, observed, joined
                if (offset + batch_size) % 16 == 0:
                    print(json.dumps({'phase': 'capture', 'condition': condition, 'samples': min(offset + batch_size, samples)}), flush=True)
            captured[condition] = {name: torch.cat(values) for name, values in chunks.items()}
    rows = []
    for name, _ in names:
        real, current = captured['real'][name], captured['current_only'][name]
        difference = current.double() - real.double()
        row = {'layer': name, 'real': covariance_stats(real), 'current_only': covariance_stats(current),
               'future_replacement_difference': covariance_stats(difference),
               'future_replacement_rms_l2': float(difference.square().sum(-1).mean().sqrt())}
        if name in ('vae_pre_norm', 'posterior_mu'):
            row['analytical_affine_hyperplane_dimension_ceiling'] = 31
        rows.append(row)
    return captured, {'layers': rows, 'native_cached_mu_max_abs_difference': native_difference,
        'observer_no_observer_parity': parity, 'all_observer_outputs_bitwise_equal': True,
        'transformer_fastpath_remained_enabled': torch.backends.mha.get_fastpath_enabled()}


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--models', nargs='+', choices=('official', 'combined3000', 'baseline3000'), default=['official', 'combined3000'])
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--batch-size', type=int, default=4)
    args = parser.parse_args()
    assert args.samples == 64 and 1 <= args.batch_size <= 8 and len(set(args.models)) == len(args.models)
    out = args.output.resolve()
    assert out.is_relative_to(api.SSD_RUNS.resolve()) and not out.exists()
    mount = json.loads(subprocess.check_output(['findmnt', '-J', '-T', str(out.parent), '-o', 'TARGET,SOURCE,FSTYPE'], text=True))
    assert mount['filesystems'][0]['target'] == '/media/sadasue/SSD1'
    cfg = yaml.safe_load((GROUP / 'kl_mse/resolved_config.yaml').read_text())
    request = yaml.safe_load((GROUP / 'kl_mse/requested_config.yaml').read_text())
    contract = api.read_json(GROUP / 'kl_mse/data_contract.json')
    reference = api.read_json(REFERENCE / 'results.json')
    assert reference['complete'] and api.digest(Path(api.__file__)) == reference['protocol']['script_sha256']
    source = Path(request['official_source'])
    assert subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() == contract['source_commit']
    assert hashlib.sha256(subprocess.check_output(['git', '-C', str(source), 'diff', '--binary'])).hexdigest() == contract['source_patch_sha256']
    for relative, expected in reference['protocol']['verified_visual_source_equivalence'].items():
        if ':' not in relative:
            assert api.digest(source / relative) == expected
    os.environ.update(cfg['environment'])
    sys.path[:0] = [str(source), *request['dependency_roots']]
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))) * 1024
    assert available >= 24 * 1024**3 and torch.cuda.is_available()
    cuda_free_before = torch.cuda.mem_get_info()[0]
    resource_probe = {'cuda_free_before': cuda_free_before, 'device': torch.cuda.get_device_name()}
    if cuda_free_before < 12 * 1024**3:
        # GB10 shares Linux RAM. cudaMemGetInfo can exclude reclaimable file
        # cache; prove actual headroom without global cache eviction or touching
        # another job. This exceeds the prior frozen B16 probe's3.91GiB peak.
        assert 'GB10' in resource_probe['device']
        temporary = torch.empty(6 * 1024**3, dtype=torch.uint8, device='cuda')
        temporary.zero_(); torch.cuda.synchronize()
        resource_probe.update(bounded_allocation_and_touch_bytes=temporary.numel(), passed=True)
        del temporary; torch.cuda.empty_cache(); torch.cuda.synchronize()
        resource_probe['cuda_free_after_release'] = torch.cuda.mem_get_info()[0]
    cache_path = Path(reference['protocol']['cache'])
    assert api.digest(cache_path) == api.read_json(REFERENCE / 'feature-cache-manifest.json')['sha256']
    cache = torch.load(cache_path, weights_only=True, map_location='cpu')
    assert len(cache['rows']) == 512 and cache['contract'] == contract and cache['precision'] == 'highest FP32; TF32 disabled'
    for key, value in cache['tensor_sha256'].items():
        assert api.tensor_hash(cache[key]) == value
    protocol = {'samples': args.samples, 'batch_size': args.batch_size, 'models': args.models,
        'row_selection': 'fixed first64 of the shared512 validation pairs; no data-dependent selection',
        'rows': cache['rows'][:args.samples], 'feature_cache': str(cache_path), 'feature_cache_sha256': api.digest(cache_path),
        'source': str(source), 'source_commit': contract['source_commit'], 'source_patch_sha256': contract['source_patch_sha256'],
        'script_sha256': api.digest(__file__), 'evaluator_sha256': api.digest(api.__file__),
        'precision': 'highest FP32 DINO cache/encoder/VAE; TF32 off; no autocast; centered spectra computed CPUfloat64',
        'mount': mount, 'host_available_before': available, 'resource_probe': resource_probe,
        'gpu_before': subprocess.check_output(['nvidia-smi'], text=True),
        'observer': 'Temporary instance-forward wrappers invoke exact original bound methods; no registered hooks, replacements of mathematical layers, or optimizer.'}
    out.mkdir()
    write_json(out / 'protocol.json', protocol)
    started = time.monotonic(); torch.cuda.reset_peak_memory_stats()
    reports = {}
    release = Path(reference['protocol']['release'])
    for name in args.models:
        key = {'official': 'official', 'combined3000': 'kl_mse', 'baseline3000': 'baseline'}[name]
        official = name == 'official'
        mc = yaml.safe_load((release / 'dino_large_vae.yaml').read_text())['model'] if official else yaml.safe_load((GROUP / key / 'resolved_config.yaml').read_text())['model']
        if official: mc['vision_model_id'] = cfg['model']['vision_model_id']
        checkpoint = release / 'checkpoints/pytorch_model.pt' if official else GROUP / key / 'checkpoints/final.ckpt'
        model, evidence = api.load_model(checkpoint, deepcopy(mc), contract, official=official)
        before = api.module_hashes(model)
        assert evidence['checkpoint_sha256'] == reference['models'][key]['checkpoint_sha256']
        assert before == reference['models'][key]['module_hashes_before']
        model.cuda()
        captured, result = trace_model(model, cache, args.samples, args.batch_size)
        after = api.module_hashes(model)
        assert before == after
        torch.save(captured, out / (name + '-layer-activations.pt'))
        reports[name] = {**evidence, **result, 'module_hashes_before': before, 'module_hashes_after': after, 'unchanged': True}
        write_json(out / (name + '.json'), reports[name])
        write_json(out / 'partial.json', {'complete': False, 'protocol': protocol, 'models': reports})
        print(json.dumps({'phase': 'model_complete', 'model': name,
            'top_pc_by_layer': {r['layer']: r['real']['top_pc_fraction'] for r in result['layers']}}), flush=True)
        del model, captured; gc.collect(); torch.cuda.empty_cache()
    result = {'complete': True, 'protocol': protocol, 'models': reports, 'elapsed_seconds': time.monotonic() - started,
        'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated(),
        'limitations': ['Centered between-example covariance, not uncentered mean/norm, defines the reported spectrum.',
            '64 samples bound observed centered rank by63; small eigenvalues and PR depend on the fixed sample subset.',
            '32D LayerNorm output lies in an affine31D hyperplane (up to floating arithmetic); its linear mu projection cannot increase this dimension.',
            'Future-replacement difference is an intervention on cached visual input, not a training gradient or a parameter-update history.',
            'No matching six-update trained checkpoint was retained. Historical onset and fresh initialization are not inferred from these trained snapshots.']}
    write_json(out / 'results.json', result)
    print(json.dumps({'complete': True, 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    main()
