"""Frozen B2 diagnostic of cache/native API arithmetic; no optimizer or updates."""
from copy import deepcopy
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research import evaluate_robotwin_loss_ablation_fp32 as ev


def main():
    import yaml
    runs = ev.SSD_RUNS / 'robotwin-loss-ablation-b4-20261114'
    output = runs / 'evaluations/fp32-parity-diagnosis-20261114'
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'results.json').exists():
        raise FileExistsError(output / 'results.json')
    cfg = yaml.safe_load((runs / 'baseline/resolved_config.yaml').read_text())
    request = yaml.safe_load((runs / 'baseline/requested_config.yaml').read_text())
    contract = ev.read_json(runs / 'baseline/data_contract.json')
    os.environ.update(cfg['environment'])
    sys.path[:0] = [request['official_source'], *request['dependency_roots']]
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(42)
    assert torch.cuda.is_available()
    assert torch.cuda.mem_get_info()[0] > 8 * 1024**3
    cache_path = runs / 'evaluations/final3000-highest-fp32/features512-highest-fp32.pt'
    cache = torch.load(cache_path, weights_only=True, map_location='cpu')
    release = ev.SSD_RUNS.parent / 'models/lawam-lam-bd993da'
    release_cfg = yaml.safe_load((release / 'dino_large_vae.yaml').read_text())['model']
    reports = {}
    start = time.monotonic()

    def compare(a, b):
        assert a.shape == b.shape and a.dtype == b.dtype == torch.float32
        diff = a.double() - b.double()
        assert torch.isfinite(diff).all()
        return {'max_abs': float(diff.abs().max()), 'rms': float(diff.square().mean().sqrt()),
                'equal': torch.equal(a, b), 'shape': list(a.shape),
                'strides': [list(a.stride()), list(b.stride())]}

    with torch.inference_mode():
        for name in ('baseline', 'kl_mse', 'official'):
            official = name == 'official'
            mc = deepcopy(release_cfg if official else yaml.safe_load((runs / name / 'resolved_config.yaml').read_text())['model'])
            mc['vision_model_id'] = cfg['model']['vision_model_id']
            checkpoint = release / 'checkpoints/pytorch_model.pt' if official else runs / name / 'checkpoints/final.ckpt'
            model, evidence = ev.load_model(checkpoint, mc, contract, official=official)
            before = ev.module_hashes(model)
            model.cuda()
            videos = cache['parity_videos'].cuda()
            n = len(videos)
            assert n == 2
            states = None if official else cache['states'][:n].cuda()
            mask = None if official else cache['mask'][:n, None].expand(-1, 2, -1).cuda()
            emb = cache['emb'][:n].cuda()
            x, y = cache['x'][:n].cuda(), cache['y'][:n].cuda()
            native = model.get_latent_action(videos, states, videos, state_mask=mask, embodiment_ids=emb, predict_future_frame=True)
            repeat = model.get_latent_action(videos, states, videos, state_mask=mask, embodiment_ids=emb, predict_future_frame=True)
            features = model.vision_encoder.encode(torch.cat([videos, videos], 1), n=-2)
            enc = features[:, :2]
            direct_mu, _ = model.vq._encode(model.encoder(enc, states, embodiment_id=emb))
            cached_mu, _ = model.vq._encode(model.encoder(torch.cat([x, y], 1)))
            cached_pred = model.decoder(x, cached_mu)
            direct_pred = model.decoder(features[:, 2:3], direct_mu)
            x_native_layout = torch.empty_strided(x.shape, features[:, 2:3].stride(), device=x.device, dtype=x.dtype)
            x_native_layout.copy_(x)
            layout_pred = model.decoder(x_native_layout, cached_mu)
            copied_native_pred = model.decoder(native['dec_in'].contiguous(), native['quantized'])
            comparisons = {
                'native_repeated_current': compare(native['dec_in'], repeat['dec_in']),
                'native_repeated_prediction': compare(native['recon'], repeat['recon']),
                'cached_current_vs_native': compare(x, native['dec_in']),
                'cached_future_vs_native': compare(y, native['tgt']),
                'recomputed_current_vs_native': compare(features[:, 2:3], native['dec_in']),
                'cached_mu_vs_native': compare(cached_mu, native['quantized']),
                'direct_mu_vs_native': compare(direct_mu, native['quantized']),
                'cached_prediction_vs_native': compare(cached_pred, native['recon']),
                'direct_prediction_vs_native': compare(direct_pred, native['recon']),
                'layout_matched_prediction_vs_native': compare(layout_pred, native['recon']),
                'contiguous_native_prediction_vs_native': compare(copied_native_pred, native['recon']),
            }
            after = ev.module_hashes(model)
            assert before == after
            reports[name] = {**evidence, 'comparisons': comparisons, 'unchanged': True,
                             'module_hashes_before': before, 'module_hashes_after': after}
            ev.write_json(output / 'partial.json', {'reports': reports, 'complete': False})
            print(json.dumps({'model': name, 'differences': {k: v['max_abs'] for k, v in comparisons.items()}}), flush=True)
            del model, native, repeat, features, cached_pred, direct_pred, layout_pred, copied_native_pred
            gc.collect(); torch.cuda.empty_cache()
    ev.write_json(output / 'results.json', {'complete': True, 'reports': reports,
        'script_sha256': ev.digest(__file__), 'original_evaluator_sha256': ev.digest(ev.__file__),
        'cache_sha256': ev.digest(cache_path), 'elapsed_seconds': time.monotonic() - start,
        'samples': 2, 'precision': 'highest FP32; TF32 disabled; inference_mode',
        'scope': 'Frozen B2 computation-route diagnosis only; no optimizer or updates.'})


if __name__ == '__main__':
    main()
