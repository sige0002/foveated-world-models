"""Compare fetched LeRobot LAM primitives against pinned RLinf Stage-1 code.

Same initialization seeds, current local checkpoint, and four cached real pairs.
Also execute RLinf's actual optimizer builder to verify all 30k proposed LRs.
"""
import argparse
import ast
from functools import partial
import hashlib
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.train_robotwin_stage1 import write_json


def main():
    import yaml
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--snapshot', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    cfg = yaml.safe_load((a.run / 'resolved_config.yaml').read_text())
    req = yaml.safe_load((a.run / 'requested_config.yaml').read_text())
    os.environ.update(cfg['environment'])
    clean = Path(req['official_source']).parent / 'LaWAM'
    sys.path[:0] = [str(clean), *req['dependency_roots'], str(a.snapshot / 'lam_core')]
    manifest = json.loads((a.snapshot / 'PROVENANCE.json').read_text())
    for item in manifest['files']:
        content = (a.snapshot / item['path']).read_bytes()
        actual = hashlib.sha1(b'blob ' + str(len(content)).encode() + b'\0' + content).hexdigest()
        assert actual == item['git_blob_sha'], item['path']
    import torch
    from latent_action_model.core.utils.lam_encoder import LAMEncoder as UpEncoder
    from latent_action_model.core.utils.lam_decoder import LAMDecoder_v2 as UpDecoder
    from latent_action_model.core.vq import VAEQuantizer as UpVAE
    from core.utils.lam_encoder import LAMEncoder as LeEncoder
    from core.utils.lam_decoder import LAMDecoderV2 as LeDecoder
    from core.vq import VAEQuantizer as LeVAE
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.set_float32_matmul_precision('highest')
    data = torch.load(a.run / 'evaluations/z-mechanism-20261114/features.pt', weights_only=True)
    x, y = data['x'][:4].float().cuda(), data['y'][:4].float().cuda()
    saved = torch.load(a.run / 'checkpoints/last.ckpt', weights_only=True, map_location='cpu', mmap=True)['state_dict']
    mc = cfg['model']
    common = dict(context_dim=mc['dim'], input_dim=x.shape[-1], num_heads=mc['num_heads'],
                  ffn_expansion_factor=mc['ffn_expansion_factor'], dropout=0.,
                  code_dim=mc['code_dim'], num_queries=1, grid_hw=(16, 16))
    factories = [('encoder', UpEncoder, LeEncoder, dict(common, num_layers=24, num_frames=2,
                   modal_mask=True, add_state=False)),
                 ('vq', UpVAE, LeVAE, dict(code_dim=32, layer_norm=True)),
                 ('decoder', UpDecoder, LeDecoder, dict(common, num_layers=12, last_ln=True))]
    reports, intermediates = {}, {}
    for name, original, ported, kwargs in factories:
        torch.manual_seed(42)
        first = original(**kwargs).eval().requires_grad_(False)
        torch.manual_seed(42)
        second = ported(**kwargs).eval().requires_grad_(False)
        left, right = first.state_dict(), second.state_dict()
        assert left.keys() == right.keys()
        init_equal = all(torch.equal(left[k], right[k]) for k in left)
        prefix = 'lam.' + name + '.'
        weights = {k.removeprefix(prefix): v for k, v in saved.items() if k.startswith(prefix)}
        first.load_state_dict(weights, strict=True)
        second.load_state_dict(weights, strict=True)
        first.cuda(); second.cuda()
        row = {'state_tensors': len(weights), 'fresh_seed42_initialization_exact': init_equal}
        if name == 'encoder':
            with torch.no_grad():
                outputs = [first(torch.cat([x, y], 1)), second(torch.cat([x, y], 1))]
            torch.testing.assert_close(*outputs, rtol=0, atol=1e-6)
            row['output_max_abs_difference'] = float((outputs[0] - outputs[1]).abs().max())
            intermediates['nodes'] = outputs[0].detach()
        elif name == 'vq':
            nodes = intermediates['nodes']
            with torch.no_grad():
                p1, p2 = first._encode(nodes), second._encode(nodes)
            for q1, q2 in zip(p1, p2):
                torch.testing.assert_close(q1, q2, rtol=0, atol=1e-6)
            row['posterior_max_abs_difference'] = max(float((q1 - q2).abs().max()) for q1, q2 in zip(p1, p2))
            sampled, grads = [], []
            for module in [first, second]:
                inp = nodes.detach().clone().requires_grad_(True)
                torch.manual_seed(783)
                out = module(inp)
                grad = torch.autograd.grad(out[0].square().mean() + out[4], inp)[0]
                sampled.append((out[0].detach(), out[4].detach()))
                grads.append(grad)
            for q1, q2 in zip(*sampled):
                torch.testing.assert_close(q1, q2, rtol=0, atol=1e-6)
            torch.testing.assert_close(*grads, rtol=0, atol=1e-7)
            row['sampled_z_max_abs_difference'] = float((sampled[0][0] - sampled[1][0]).abs().max())
            row['kl_max_abs_difference'] = float((sampled[0][1] - sampled[1][1]).abs().max())
            row['node_gradient_max_abs_difference'] = float((grads[0] - grads[1]).abs().max())
            intermediates['mu'] = p1[0].detach()
        else:
            with torch.no_grad():
                outputs = [first(x, intermediates['mu']), second(x, intermediates['mu'])]
            torch.testing.assert_close(*outputs, rtol=0, atol=1e-6)
            row['output_max_abs_difference'] = float((outputs[0] - outputs[1]).abs().max())
        reports[name] = row
        print(json.dumps({name: row}), flush=True)
        del first, second, left, right, weights
        torch.cuda.empty_cache()

    # Extract and execute the original method, rather than restating its schedule.
    path = clean / 'latent_action_model/core/lam_lightinng.py'
    tree = ast.parse(path.read_text())
    klass = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'VJEPA_LAM')
    method = next(n for n in klass.body if isinstance(n, ast.FunctionDef) and n.name == 'configure_optimizers')
    module = ast.Module(body=[method], type_ignores=[])
    namespace = {'torch': torch, 'Any': object}
    exec(compile(ast.fix_missing_locations(module), str(path), 'exec'), namespace)
    param = torch.nn.Parameter(torch.ones(1))
    obj = SimpleNamespace(exclude_bias_norm_from_wd=False, warmup_steps=10000,
                          optimizer=partial(torch.optim.AdamW, lr=3e-4, weight_decay=.01),
                          parameters=lambda: [param])
    built = namespace['configure_optimizers'](obj)
    optimizer = built['optimizer']
    scheduler = built['lr_scheduler']['scheduler']
    schedule = []
    for step in range(1, 30001):
        actual = optimizer.param_groups[0]['lr']
        assert abs(actual - 3e-4 * min(step / 10000, 1.)) < 1e-15
        if step in [1, 2, 100, 1000, 5000, 9999, 10000, 10001, 20000, 30000]:
            schedule.append({'update_1_based': step, 'lr_used': actual})
        param.grad = torch.zeros_like(param)
        optimizer.step()
        scheduler.step()
    write_json(a.output, dict(complete=True, lerobot_commit=manifest['commit'],
        upstream_source=str(clean), modules=reports, stage1_schedule=schedule,
        scheduler_source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        schedule_scope='Original RLinf configure_optimizers executed on a dummy parameter, all 30k rates checked; no model training.',
        parity_scope='Four cached real pairs, identical local 10k weights, highest FP32. Primitive encoder/VAE/visual decoder; not the complete LeRobot policy pipeline.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))


if __name__ == '__main__':
    main()
