"""Control target-informed per-example z fitting with constant and shuffled z.

All target access here is diagnostic. No model weights are trained or saved.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.train_robotwin_stage1 import write_json


def main():
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--diagnostics', type=Path, required=True)
    a = p.parse_args()
    cfg = yaml.safe_load((a.run / 'resolved_config.yaml').read_text())
    req = yaml.safe_load((a.run / 'requested_config.yaml').read_text())
    os.environ.update(cfg['environment'])
    os.environ.update(OMP_NUM_THREADS='2', MKL_NUM_THREADS='2')
    sys.path[:0] = [str(Path(req['official_source']).parent / 'LaWAM'), *req['dependency_roots']]
    import torch
    from torch.nn import functional as F
    from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    cache = torch.load(a.diagnostics / 'features.pt', weights_only=True)
    all_reports = {}
    for kind in ['local', 'official']:
        started = time.monotonic()
        d = json.loads((a.diagnostics / kind / 'results.json').read_text())
        tensors = torch.load(a.diagnostics / kind / 'tensors.pt', weights_only=True)
        mc = dict(cfg['model'])
        decoder = LAMDecoder_v2(context_dim=mc['dim'], input_dim=cache['x'].shape[-1],
            num_queries=mc['num_queries'], num_layers=mc['dec_layers'], num_heads=mc['num_heads'],
            dropout=mc['dropout'], ffn_expansion_factor=mc['ffn_expansion_factor'],
            num_embodiments=mc['num_embodiments'], code_dim=mc['code_dim'],
            grid_hw=tuple(i // mc['patch_size'] for i in mc['image_hw']),
            last_ln=mc.get('decoder_last_ln', True)).eval().requires_grad_(False)
        saved = torch.load(d['checkpoint'], weights_only=True, map_location='cpu', mmap=True)
        state = {k.removeprefix('lam.decoder.'): v for k, v in saved['state_dict'].items() if k.startswith('lam.decoder.')}
        decoder.load_state_dict(state, strict=True)
        del state, saved
        decoder.cuda()
        amp = lambda: torch.autocast('cuda', dtype=torch.bfloat16)

        def score(z, n):
            results = {k: [] for k in ['future_mse', 'future_smooth_l1', 'current_mse']}
            with torch.no_grad():
                for start in range(0, n, 4):
                    sl = slice(start, start + 4)
                    xx, yy = cache['x'][sl].cuda(), cache['y'][sl].cuda()
                    with amp():
                        pred = decoder(xx, z[sl].cuda()).float()
                    results['future_mse'].append((pred - yy.float()).square().flatten(1).mean(1).cpu())
                    results['future_smooth_l1'].append(F.smooth_l1_loss(pred, yy.float(), beta=.1, reduction='none').flatten(1).mean(1).cpu())
                    results['current_mse'].append((pred - xx.float()).square().flatten(1).mean(1).cpu())
            return {k: float(torch.cat(v).mean()) for k, v in results.items()}

        codes = tensors['oracle_codes']
        n = len(codes)
        # A circular shift has no fixed points. Verify these 16 cases are from different episodes.
        assert all(cache['rows'][i]['episode'] != cache['rows'][(i - 1) % n]['episode'] for i in range(n))
        mu = tensors['encoding']['real']['mu'][:n]
        z = mu.mean(0, keepdim=True).cuda().clone().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=.05)
        trace = []
        for step in range(81):
            optimizer.zero_grad(set_to_none=True)
            total = 0.
            for start in range(0, n, 4):
                sl = slice(start, start + 4)
                with amp():
                    pred = decoder(cache['x'][sl].cuda(), z.expand(4, -1, -1))
                    loss = F.smooth_l1_loss(pred.float(), cache['y'][sl].float().cuda(), beta=.1)
                total += float(loss.detach()) * 4 / n
                if step < 80:
                    (loss * 4 / n).backward()
            if step in [0, 5, 20, 80]:
                trace.append({'step': step, 'future_smooth_l1': total, 'z_norm': float(z.detach().norm())})
            if step < 80:
                optimizer.step()
        report = dict(original=score(mu, n), individual_oracle=score(codes, n),
                      shuffled_individual_oracle=score(codes.roll(1, 0), n),
                      shared_constant_oracle=score(z.detach().cpu().expand(n, -1, -1), n),
                      shared_trace=trace,
                      current_only_code_128=score(tensors['encoding']['current']['mu'], len(cache['rows'])),
                      scope='16 target-informed fits; one shared constant and individual codes see exactly the same targets. No held-out performance claim.',
                      elapsed_seconds=time.monotonic() - started)
        # Replay both endpoints from the independent latent-optimization script.
        for key, idx in [('original', 0), ('individual_oracle', -1)]:
            old = sum(row['trace'][idx]['mse'] for row in d['oracle']) / len(d['oracle'])
            assert abs(old - report[key]['future_mse']) < 1e-6
        all_reports[kind] = report
        print(json.dumps({kind: report}), flush=True)
        del decoder
        torch.cuda.empty_cache()
    write_json(a.diagnostics / 'oracle-controls.json', dict(complete=True,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), reports=all_reports))


if __name__ == '__main__':
    main()
