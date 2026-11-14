"""Frozen layerwise query geometry for real versus current-only future inputs.

Traces the existing computation. No layer is replaced, and no optimizer runs.
The observed geometry cannot identify which training updates produced it.
"""
import argparse
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
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()
    req = yaml.safe_load((args.run/'requested_config.yaml').read_text())
    cfg = yaml.safe_load((args.run/'resolved_config.yaml').read_text())
    source = Path(req['official_source']).parent/'LaWAM'
    commit = subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    assert commit == req['official_commit']
    assert not subprocess.check_output(['git','-C',str(source),'diff','--binary'])
    os.environ.update(cfg['environment'])
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    sys.path[:0] = [str(source), *req['dependency_roots']]
    import torch
    from latent_action_model.core.utils.lam_encoder import LAMEncoder
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.manual_seed(142)
    cache_path = args.run/'evaluations/z-mechanism-20261114/features.pt'
    cache = torch.load(cache_path, weights_only=True, map_location='cpu')
    x, y = cache['x'].float(), cache['y'].float()
    assert len(x) == 128
    release = args.run.parents[1]/'models/lawam-lam-bd993da'
    started = time.monotonic()
    reports = {}

    def fingerprint(module):
        h = hashlib.sha256()
        for key, value in module.state_dict().items():
            h.update(key.encode()+b'\0')
            h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return h.hexdigest()

    def energy(tensor):
        return float(tensor.square().sum(-1).mean())

    def radial_share(q, displacement):
        radial = q*(q*displacement).sum(-1, keepdim=True)/q.square().sum(-1, keepdim=True).clamp_min(1e-30)
        return energy(radial)/max(energy(displacement), 1e-30)

    for kind in ['official', 'local']:
        checkpoint = release/'checkpoints/pytorch_model.pt' if kind == 'official' else args.run/'checkpoints/last.ckpt'
        mc = yaml.safe_load((release/'dino_large_vae.yaml').read_text())['model'] if kind == 'official' else cfg['model']
        encoder = LAMEncoder(context_dim=mc['dim'], input_dim=x.shape[-1],
            num_heads=mc['num_heads'], dropout=mc['dropout'], code_dim=mc['code_dim'],
            num_queries=mc['num_queries'], ffn_expansion_factor=mc['ffn_expansion_factor'],
            grid_hw=tuple(int(v/mc['patch_size']) for v in mc['image_hw']),
            num_layers=mc['enc_layers'], num_frames=mc['num_frames'], modal_mask=mc['enc_modal_mask'],
            add_state=mc['enc_add_state'], max_state_dim=mc['max_state_dim'], num_embodiments=mc['num_embodiments'])
        saved = torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)
        state = {k.removeprefix('lam.encoder.'):v for k,v in saved['state_dict'].items() if k.startswith('lam.encoder.')}
        assert len(state) == 305
        encoder.load_state_dict(state, strict=True)
        encoder.eval().requires_grad_(False)
        before = fingerprint(encoder)
        del saved,state
        encoder.cuda()
        capture = {}
        handles = []
        def hook(name):
            def collect(module, inputs, output):
                capture[name].append(output[:, -1, :].detach().double().cpu())
            return collect
        names = ['cross_attention']+[f'layer_{i+1:02d}' for i in range(len(encoder.QFormer.layers))]
        handles.append(encoder.QFormer.q_cross_attn.register_forward_hook(hook(names[0])))
        for name, layer in zip(names[1:], encoder.QFormer.layers):
            handles.append(layer.register_forward_hook(hook(name)))
        queries = {}
        with torch.no_grad():
            for condition, future in [('real',y), ('current',x)]:
                capture = {name:[] for name in names}
                for offset in range(0,len(x),4):
                    encoder(torch.cat([x[offset:offset+4],future[offset:offset+4]],1).cuda())
                queries[condition] = {name:torch.cat(values) for name,values in capture.items()}
                print(json.dumps({'phase':'capture','model':kind,'condition':condition,'samples':len(x)}),flush=True)
        for handle in handles:
            handle.remove()
        rows = []
        previous = None
        for name in names:
            q = queries['real'][name]
            delta = queries['current'][name]-q
            mean = q.mean(0,keepdim=True)
            row = {'layer':name, 'query_norm_mean':float(q.norm(dim=-1).mean()),
                'query_energy':energy(q), 'common_mean_energy':energy(mean),
                'between_example_energy':energy(q-mean),
                'between_example_fraction':energy(q-mean)/energy(q),
                'future_replacement_energy':energy(delta),
                'future_replacement_fraction_of_query_energy':energy(delta)/energy(q),
                'future_replacement_radial_fraction':radial_share(q,delta)}
            if previous is not None:
                residual = q-previous
                cross = float(2*(previous*residual).sum(-1).mean())
                change = energy(q)-energy(previous)
                assert abs(change-cross-energy(residual)) <= 1e-6*max(energy(q),1.)
                row.update(residual_energy=energy(residual),
                    residual_common_mean_energy=energy(residual.mean(0,keepdim=True)),
                    twice_query_residual_inner_product=cross, query_energy_increment=change)
            previous = q
            rows.append(row)
        w = encoder.out_proj.weight.detach().double().cpu()
        bias = encoder.out_proj.bias.detach().double().cpu()
        cw, cb = w-w.mean(0,keepdim=True), bias-bias.mean()
        q = queries['real'][names[-1]]
        dq = queries['current'][names[-1]]-q
        p = q@cw.T+cb
        dp = dq@cw.T
        qm = q.mean(0)
        projection = {
            'weight_frobenius_norm':float(w.norm()),
            'centered_weight_spectral_norm':float(torch.linalg.svdvals(cw)[0]),
            'centered_bias_norm':float(cb.norm()),
            'common_query_norm':float(qm.norm()),
            'projected_common_query_without_bias_norm':float((cw@qm).norm()),
            'common_direction_effective_gain':float((cw@qm).norm()/qm.norm()),
            'future_displacement_effective_gain':(energy(dp)/energy(dq))**.5,
            'between_example_effective_gain':(energy((q-q.mean(0))@cw.T)/energy(q-q.mean(0)))**.5,
            'centered_projected_coordinate_std_mean':float(p.square().mean(-1).sqrt().mean()),
            'centered_projected_future_displacement_radial_fraction':radial_share(p,dp),
        }
        after = fingerprint(encoder)
        assert before == after and all(p.grad is None for p in encoder.parameters())
        reports[kind] = {'checkpoint':str(checkpoint), 'state_tensors':305, 'state_sha256_before':before,
            'state_sha256_after':after, 'unchanged':True,'layers':rows,'projection':projection}
        write_json(args.output.with_name(args.output.stem+'-partial.json'),{'complete':False,'reports':reports})
        print(json.dumps({'phase':'model_complete','model':kind,'first':rows[0],'last':rows[-1],'projection':projection}),flush=True)
        del encoder
        torch.cuda.empty_cache()
    write_json(args.output, {'complete':True, 'reports':reports,'samples':len(x),'batch_size':4,
        'source_commit':commit,'precision':'highest FP32 model, float64 stored geometry',
        'feature_cache':str(cache_path),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'elapsed_seconds':time.monotonic()-started,'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),
        'scope':'Frozen trained encoder only, actual released/local10000 weights, same128pairs. Layer hooks leave computation unchanged but may disable fused transformer fastpath. Radial decomposition in query coordinates is diagnostic; only the final centered projection is the actual VAE LayerNorm coordinate. Not a measurement of historical parameter updates or causal training changes.'})


if __name__ == '__main__':
    main()
