"""CPU decomposition of saved first-block observations; no model execution."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
import torch
from trace_robotwin_layer_covariance import covariance_stats


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def centered(x):
    return x - x.mean(0, keepdim=True)


def covariance_energy(x):
    return centered(x).square().sum() / (len(x) - 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    ref = json.loads((args.source / 'results.json').read_text())
    assert ref['complete'] and ref['protocol']['samples'] == 64
    reports, source_hashes = {}, {}
    for name in ('official', 'combined3000'):
        path = args.source / (name + '-activations.pt')
        activation = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        source_hashes[name] = sha(path)
        condition_reports = {}
        for condition in ('real', 'current_only'):
            values = activation[condition]
            p = values['attention_probabilities'].double()
            n, h, tokens = p.shape
            v = values['attention_V'].double().reshape(n, tokens, h, -1).permute(0, 2, 1, 3)
            assert (n, h, tokens) == (64, 16, 512)
            dh = v.shape[-1]
            weighted = torch.einsum('bhs,bhsd->bhd', p, v)
            native = values['attention_weighted_V'].double().reshape(n, h, dh)
            torch.testing.assert_close(weighted, native, atol=3e-5, rtol=1e-6)
            argmax = p.argmax(-1)
            sample_index = torch.arange(n)[:, None]
            selected_inputs = {key: values[key].double()[sample_index, argmax]
                for key in ['visual_features', 'project_in', 'context_norm1']}
            raw_norm = values['visual_features'].double().norm(dim=-1)
            head_reports = []
            total_trace = float(covariance_energy(native.flatten(1)))
            for i in range(h):
                counts = torch.bincount(argmax[:, i], minlength=tokens)
                spatial_counts = torch.bincount(argmax[:, i] % (tokens // 2), minlength=tokens // 2)
                head_energy = centered(native[:, i]).square().sum(-1)
                order = head_energy.argsort(descending=True)
                head_reports.append({'head': i,
                    'weighted_V': covariance_stats(native[:, i]),
                    'weighted_V_trace_fraction': float(covariance_energy(native[:, i])) / total_trace,
                    'V_all_tokens_per_sample': covariance_stats(v[:, i].flatten(1)),
                    'argmax_selected_token_geometry': {key: covariance_stats(value[:, i]) for key, value in selected_inputs.items()},
                    'argmax_selected_V_head': covariance_stats(v[torch.arange(n), i, argmax[:, i]]),
                    'raw_feature_norm_percentile_of_selected_token_mean': float((raw_norm <= raw_norm[torch.arange(n), argmax[:, i]][:, None]).double().mean()),
                    'largest_one_example_variance_share': float(head_energy.max() / head_energy.sum()),
                    'largest_four_examples_variance_share': float(head_energy[order[:4]].sum() / head_energy.sum()),
                    'weighted_V_excluding_four_largest_energy_examples': covariance_stats(native[order[4:], i]),
                    'argmax_unique_token_positions': int((counts > 0).sum()),
                    'argmax_modal_token_position': int(counts.argmax()),
                    'argmax_modal_token_fraction': float(counts.max()) / n,
                    'argmax_same_token_probability_across_distinct_samples': float((counts * (counts - 1)).sum()) / (n * (n - 1)),
                    'argmax_same_spatial_position_probability_across_distinct_samples': float((spatial_counts * (spatial_counts - 1)).sum()) / (n * (n - 1)),
                    'argmax_future_frame_fraction': float((argmax[:, i] >= tokens // 2).double().mean()),
                    'future_frame_mass': float(p[:, i, tokens // 2:].sum(-1).mean()),
                    'logit_std_across_tokens_mean': float(values['attention_logits'][:, i].double().std(-1).mean())})
            agreement = argmax[:, :, None] == argmax[:, None, :]
            off_diagonal = ~torch.eye(h, dtype=torch.bool)
            pbar, vbar = p.mean(0, keepdim=True), v.mean(0, keepdim=True)
            dp, dv = p - pbar, v - vbar
            terms = {
                'value_variation_at_fixed_mean_attention': torch.einsum('bhs,bhsd->bhd', pbar.expand_as(p), dv).flatten(1),
                'attention_variation_at_fixed_mean_values': torch.einsum('bhs,bhsd->bhd', dp, vbar.expand_as(v)).flatten(1),
                'attention_value_interaction': centered(torch.einsum('bhs,bhsd->bhd', dp, dv).flatten(1))}
            actual = centered(weighted.flatten(1))
            reconstruction = sum(terms.values())
            torch.testing.assert_close(reconstruction, actual, atol=1e-10, rtol=1e-10)
            energy = actual.square().sum()
            decomposition = {key: {'geometry': covariance_stats(value),
                'own_variance_over_actual_variance': float(value.square().sum() / energy)} for key, value in terms.items()}
            crosses = {}
            keys = list(terms)
            for i, left in enumerate(keys):
                for right in keys[i + 1:]:
                    crosses[left + '__' + right] = float(2 * (terms[left] * terms[right]).sum() / energy)
            total = sum(v['own_variance_over_actual_variance'] for v in decomposition.values()) + sum(crosses.values())
            assert abs(total - 1) < 1e-12
            x = values['query_plus_attention'].double().flatten(1)
            zero_centered = x - x.mean(-1, keepdim=True)
            eps = ref['models'][name]['weight_analysis']['layernorm']['query_norm2']['eps']
            normalized = zero_centered / (zero_centered.square().mean(-1, keepdim=True) + eps).sqrt()
            query, ff = values['query_norm2'].double().flatten(1), values['ffn_linear2'].double().flatten(1)
            qcenter, fcenter = centered(query), centered(ff)
            output = centered(values['query_plus_ffn'].double().flatten(1))
            output_energy = output.square().sum()
            condition_reports[condition] = {'heads': head_reports,
                'between_heads_same_argmax_fraction': float(agreement[:, off_diagonal].double().mean()),
                'argmax_indices': argmax.tolist(),
                'weighted_V_max_head_trace_fraction': max(v['weighted_V_trace_fraction'] for v in head_reports),
                'decomposition_of_centered_weighted_V': decomposition,
                'decomposition_cross_terms_over_actual_variance': crosses,
                'decomposition_sum': total,
                'manual_weighted_V_max_abs_difference': float((weighted - native).abs().max()),
                'norm2_decomposition': {key: covariance_stats(val) for key, val in [
                    ('input', x), ('coordinate_centering_only', zero_centered),
                    ('divide_by_per_sample_std_without_affine', normalized), ('learned_affine_output', query)]},
                'residual_ffn_variance_terms_over_output_variance': {
                    'normalized_query': float(qcenter.square().sum() / output_energy),
                    'ffn': float(fcenter.square().sum() / output_energy),
                    'twice_cross': float(2 * (qcenter * fcenter).sum() / output_energy)}}
            del v, p, weighted, native, terms, dv, dp, vbar, pbar
        real_argmax = torch.tensor(condition_reports['real']['argmax_indices'])
        curr_argmax = torch.tensor(condition_reports['current_only']['argmax_indices'])
        condition_reports['real_current_only_argmax_agreement'] = {
            'exact_token_fraction': float((real_argmax == curr_argmax).double().mean()),
            'same_spatial_position_fraction': float(((real_argmax % 256) == (curr_argmax % 256)).double().mean())}
        checkpoint = torch.load(ref['models'][name]['checkpoint'], weights_only=True, mmap=True, map_location='cpu')
        weight = checkpoint['state_dict']['lam.encoder.QFormer.q_cross_attn.attn.out_proj.weight'].double()
        v_weight = checkpoint['state_dict']['lam.encoder.QFormer.q_cross_attn.attn.in_proj_weight'].chunk(3)[2].double()
        head_weight_spectra = []
        for i in range(16):
            singular = torch.linalg.svdvals(v_weight[64 * i:64 * (i + 1)])
            energy = singular.square()
            proportion = energy / energy.sum()
            head_weight_spectra.append({'head': i, 'top_singular_energy_fraction': float(proportion[0]),
                'participation_ratio': float(1 / proportion.square().sum()),
                'frobenius_norm': float(energy.sum().sqrt()), 'spectral_norm': float(singular[0])})
        condition_reports['V_weight_by_head'] = head_weight_spectra
        posterior_dir = args.source.parent / 'final3000-highest-fp32-v2' / ('official' if name == 'official' else 'kl_mse')
        posterior_result = json.loads((posterior_dir / 'results.json').read_text())
        assert [[r['episode'], r['base']] for r in posterior_result['rows'][:64]] == [
            [r['episode'], r['base']] for r in ref['protocol']['rows']]
        posterior = torch.load(posterior_dir / 'posterior-codes.pt', map_location='cpu', weights_only=True)
        mu = centered(posterior['mu'][:64].double().flatten(1))
        mu_scores = torch.linalg.svd(mu, full_matrices=False)[0][:, 0]
        head_pc_correlations = []
        weighted_heads = activation['real']['attention_weighted_V'].double().reshape(64, 16, 64)
        for i in range(16):
            head_scores = torch.linalg.svd(centered(weighted_heads[:, i]), full_matrices=False)[0][:, 0]
            correlation = float(head_scores @ mu_scores)
            head_pc_correlations.append({'head': i, 'absolute_correlation': abs(correlation), 'r_squared': correlation ** 2})
        condition_reports['weighted_V_head_PC1_vs_final_mu_PC1'] = head_pc_correlations
        condition_reports['posterior_codes_sha256'] = sha(posterior_dir / 'posterior-codes.pt')
        x = centered(activation['real']['attention_weighted_V'].double().flatten(1))
        _, singular, right = torch.linalg.svd(x, full_matrices=False)
        eigen = singular.square() / (len(x) - 1)
        gains = (right @ weight.T).square().sum(-1)
        contribution = eigen * gains
        contribution /= contribution.sum()
        condition_reports['attention_output_projection_real'] = {
            'weighted_V_covariance_eigenvalues_descending': eigen.tolist(),
            'input_pc_squared_directional_gain': gains.tolist(),
            'output_variance_contribution_by_input_pc': contribution.tolist(),
            'first_input_pc_output_variance_fraction': float(contribution[0]),
            'first_input_pc_gain_over_variance_weighted_remaining_gain': float(gains[0] / ((eigen[1:] * gains[1:]).sum() / eigen[1:].sum()))}
        reports[name] = condition_reports
        del activation, checkpoint, weight
    assert not torch.cuda.is_initialized()
    result = {'source': str(args.source), 'source_results_sha256': sha(args.source / 'results.json'),
        'source_activations_sha256': source_hashes, 'script_sha256': sha(__file__), 'samples': 64,
        'models': reports, 'cuda_initialized': False,
        'limits': ['Per-head comparisons are within each checkpoint; head indices across models have no guaranteed semantic alignment.',
            'Fixed-mean decomposition is an algebraic frozen-data analysis, not a retraining intervention or a historical causal finding.',
            'Individual variance terms may exceed total variance because cross-covariance terms can cancel.',
            'Argmax index refers to frame/spatial token coordinates, not an identified physical object.',
            'A sharp attention distribution or negative GELU preactivation alone does not establish collapse; both models must be compared.']}
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    for name, report in reports.items():
        print(name, 'max_head_variance', report['real']['weighted_V_max_head_trace_fraction'],
              'first_PC_projection_share', report['attention_output_projection_real']['first_input_pc_output_variance_fraction'])
        for key, value in report['real']['decomposition_of_centered_weighted_V'].items():
            print(key, value['own_variance_over_actual_variance'], value['geometry']['top_pc_fraction'])
    print(args.output)


if __name__ == '__main__':
    main()
