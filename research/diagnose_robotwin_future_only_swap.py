"""Frozen future-only IDM intervention; no learning and no automatic launch.

Keep current x_i fixed in IDM and decoder. Replace only the IDM future y_i
with the already pinned different-episode y_donor. Score against true y_i.
The mismatched endpoint pair can be off distribution, not a physical action.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import evaluate_robotwin_loss_ablation_fp32_v2 as base

MODELS = ('baseline', 'kl_mse', 'official')
BASE_SHA256 = 'ff02f76a61305c80bb036d0aca471d675387975cd7ea59454da7479018855758'
PIN_KEYS = ('config_sha256', 'source_commit', 'local_patch_sha256',
            'source_files_sha256', 'verified_visual_source_equivalence',
            'validation_pair_and_donor_sha256', 'dataset_info_sha256', 'release_config_sha256')


def intervals(reports, draws=4000):
    """Same target-episode resamples across models; fixed donor assignment."""
    import numpy as np
    rows = reports['baseline']['rows']
    episodes = np.array([row['episode'] for row in rows])
    unique, inverse = np.unique(episodes, return_inverse=True)
    assert len(rows) == 512 and len(unique) == 415
    counts = np.bincount(inverse)
    picks = np.random.default_rng(1942).integers(0, len(unique), size=(draws, len(unique)))
    denominator = counts[picks].sum(1)
    result = {}
    for name, report in reports.items():
        assert [row['episode'] for row in report['rows']] == episodes.tolist()
        metrics = {}
        for key in report['rows'][0]['metrics']:
            values = np.array([row['metrics'][key] for row in report['rows']], dtype=np.float64)
            assert np.isfinite(values).all()
            sums = np.bincount(inverse, weights=values)
            distribution = sums[picks].sum(1) / denominator
            metrics[key] = {'estimate': float(values.mean()),
                            'ci95': np.quantile(distribution, [.025, .975]).tolist()}
        result[name] = {'samples': 512, 'episodes': 415, 'metrics': metrics}
    return result


def evaluate(model, cache, reference, reference_codes, *, official):
    import torch
    rows, means, altered_means = [], [], []
    maximum_reference_difference = {}
    identity_keys = ('episode', 'base', 'donor_index', 'donor_episode', 'donor_base')
    donors = torch.tensor([row['donor_index'] for row in cache['rows']])
    for start in range(0, 512, 2):
        sl = slice(start, start + 2)
        current = cache['x'][sl].cuda()
        true_future = cache['y'][sl].cuda()
        donor_future = cache['y'][donors[sl]].cuda()
        real_input, decoder_current, teacher_future = base.native_feature_views(current, true_future)
        altered_input, _, _ = base.native_feature_views(current, donor_future)
        assert torch.equal(real_input[:, :1], altered_input[:, :1])
        assert torch.equal(teacher_future, true_future)
        mu, _ = model.vq._encode(model.encoder(real_input))
        changed_mu, _ = model.vq._encode(model.encoder(altered_input))
        # Identical frozen model, features, B2 and native views as completed v2.
        torch.testing.assert_close(mu.cpu(), reference_codes['mu'][sl], rtol=0, atol=1e-6)
        maximum_reference_difference['mu'] = max(maximum_reference_difference.get('mu', 0.),
            float((mu.cpu() - reference_codes['mu'][sl]).abs().max()))
        real_pred = model.decoder(decoder_current, mu)
        changed_pred = model.decoder(decoder_current, changed_mu)
        real_error = (real_pred - teacher_future).square().flatten(1).mean(1)
        changed_error = (changed_pred - teacher_future).square().flatten(1).mean(1)
        metrics = {
            'real_visual_mse': real_error,
            'future_swapped_visual_mse': changed_error,
            'future_swapped_minus_real_visual_mse': changed_error - real_error,
            'output_difference_mse': (changed_pred - real_pred).square().flatten(1).mean(1),
            'mu_distance_l2': (changed_mu - mu).flatten(1).norm(dim=1),
            'mu_difference_mse': (changed_mu - mu).square().flatten(1).mean(1),
            'donor_future_distance_mse': (donor_future - teacher_future).square().flatten(1).mean(1),
        }
        if not official:
            states, delta, emb = [cache[key][sl].cuda() for key in ('states', 'delta', 'emb')]
            real_state = model.state_decoder(mu, states[:, :1], emb)
            changed_state = model.state_decoder(changed_mu, states[:, :1], emb)
            se = (real_state[:, :16] - delta[:, :16]).square().mean(1)
            ce = (changed_state[:, :16] - delta[:, :16]).square().mean(1)
            metrics.update(real_state_mse=se, future_swapped_state_mse=ce,
                           future_swapped_minus_real_state_mse=ce - se)
        for key, values in metrics.items():
            assert torch.isfinite(values).all(), key
        for j in range(len(current)):
            row = deepcopy(cache['rows'][start + j])
            prior = reference['rows'][start + j]
            assert all(row[key] == prior[key] for key in identity_keys)
            row['metrics'] = {key: float(value[j]) for key, value in metrics.items()}
            for key in ('real_visual_mse', *(() if official else ('real_state_mse',))):
                difference = abs(row['metrics'][key] - prior['metrics'][key])
                maximum_reference_difference[key] = max(maximum_reference_difference.get(key, 0.), difference)
                assert difference <= 1e-7, (start + j, key, difference)
            rows.append(row)
        means.append(mu.cpu()); altered_means.append(changed_mu.cpu())
        if len(rows) % 128 == 0:
            print(json.dumps({'phase': 'future_only_swap', 'model': reference['kind'],
                              'samples': len(rows)}), flush=True)
    return {'rows': rows, 'reference_parity': {'maximum_absolute_difference': maximum_reference_difference,
             'mu_atol': 1e-6, 'scalar_MSE_atol': 1e-7, 'rtol': 0}}, {
                 'real_mu': torch.cat(means), 'future_swapped_mu': torch.cat(altered_means),
                 'identity_sha256': cache['reference_identity_sha256']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs-root', type=Path, required=True)
    parser.add_argument('--reference-evaluation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    assert base.digest(base.__file__) == BASE_SHA256
    reference_dir = args.reference_evaluation.resolve()
    assert reference_dir.is_relative_to(base.SSD_RUNS.resolve())
    assert args.output.resolve() != reference_dir
    original_protocol = base.read_json(reference_dir / 'protocol.json')
    assert original_protocol['script_sha256'] == BASE_SHA256
    assert original_protocol['batch_size'] == 2 and original_protocol['samples'] == 512
    args.cache = Path(original_protocol['cache'])
    args.release = Path(original_protocol['release'])
    args.batch_size, args.bootstrap = 2, 4000
    runs, configs, requests, contracts, identities, donors, release_cfg, prepared = base.prepare(args)
    assert all(prepared[key] == original_protocol[key] for key in PIN_KEYS)
    protocol = {**prepared, 'script_sha256': base.digest(__file__), 'evaluator_v2_sha256': BASE_SHA256,
        'models': list(MODELS), 'reference_evaluation': str(reference_dir),
        'reference_protocol_sha256': base.digest(reference_dir / 'protocol.json'),
        'intervention': 'IDM(x_i,y_i) versus IDM(x_i,y_donor), same current x_i in IDM and decoder; score both against true future y_i.',
        'scope': 'Frozen diagnostic. Cross-episode future endpoints can be off distribution, not physically realizable transitions or action labels. This is not a semantic repair test.',
        'bootstrap_scope': '4000 paired target-episode cluster resamples; fixed weights and one fixed donor permutation; no training-seed/donor/precision uncertainty.',
        'complete_reference_available': (reference_dir / 'results.json').is_file()}
    base.write_json(args.output / 'diagnostic-protocol.json', protocol)
    if args.prepare_only:
        print(json.dumps({'prepared_only': True, 'GPU_used': False, 'protocol': protocol}), flush=True)
        return
    if (args.output / 'results.json').exists():
        raise FileExistsError(args.output / 'results.json')
    final_reference = base.read_json(reference_dir / 'results.json')
    assert final_reference['complete'] and final_reference['protocol'] == original_protocol
    protocol['reference_results_sha256'] = base.digest(reference_dir / 'results.json')
    reference_models = {name: base.read_json(reference_dir / name / 'results.json') for name in MODELS}
    for name, result in reference_models.items():
        assert result['complete'] and result['unchanged'] and len(result['rows']) == 512
        assert result['module_hashes_before'] == result['module_hashes_after']
        assert result['module_hashes_before'] == final_reference['models'][name]['module_hashes_before']
    os.environ.update(configs['baseline']['environment'])
    sys.path[:0] = [prepared['source'], *requests['baseline']['dependency_roots']]
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.is_available()
    available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith('MemAvailable:'))) * 1024
    free, _ = torch.cuda.mem_get_info()
    if available < 16 * 1024**3 or free < 8 * 1024**3:
        raise RuntimeError('Insufficient headroom; no model loaded and no other jobs changed')
    manifest = base.read_json(reference_dir / 'feature-cache-manifest.json')
    assert base.digest(args.cache) == manifest['sha256']
    cache = torch.load(args.cache, weights_only=True, map_location='cpu')
    reports = {}
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for name in MODELS:
            official = name == 'official'
            cfg = deepcopy(release_cfg if official else configs[name]['model'])
            cfg['vision_model_id'] = configs['baseline']['model']['vision_model_id']
            checkpoint = args.release / 'checkpoints/pytorch_model.pt' if official else runs[name] / 'checkpoints/final.ckpt'
            model, evidence = base.load_model(checkpoint, cfg, contracts['baseline'], official=official)
            assert evidence['checkpoint_sha256'] == reference_models[name]['checkpoint_sha256']
            before = base.module_hashes(model)
            assert before == reference_models[name]['module_hashes_before']
            base.validate_cache(cache, identities, contracts['baseline'], before['vision_encoder'])
            model.cuda()
            native_parity = base.native_route_parity(model, cache, official=official)
            reference_codes = torch.load(reference_dir / name / 'posterior-codes.pt', weights_only=True, map_location='cpu')
            result, codes = evaluate(model, cache, reference_models[name], reference_codes, official=official)
            after = base.module_hashes(model)
            assert before == after
            result.update(kind=name, complete=True, **evidence, module_hashes_before=before,
                          module_hashes_after=after, unchanged=True, native_api_parity=native_parity,
                          reference_rows_sha256=base.digest(reference_dir / name / 'results.json'))
            out = args.output / name
            out.mkdir(exist_ok=True)
            base.write_json(out / 'results.json', result)
            torch.save(codes, out / 'posterior-codes.pt')
            reports[name] = result
            del model, reference_codes
            gc.collect(); torch.cuda.empty_cache()
    assert base.digest(reference_dir / 'results.json') == protocol['reference_results_sha256']
    result = {'complete': True, 'protocol': protocol, 'summary': intervals(reports),
              'elapsed_seconds': time.monotonic() - started,
              'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
    base.write_json(args.output / 'results.json', result)
    print(json.dumps({'complete': True, 'output': str(args.output / 'results.json')}), flush=True)


if __name__ == '__main__':
    main()
