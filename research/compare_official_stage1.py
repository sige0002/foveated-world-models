"""Compare unmodified upstream and vendored direct Stage-1 in isolated processes.

Run prepare once, then run separately with --source pointing to each source tree,
then compare. Uses identical real images, local checkpoint, config and RNG. No
optimizer step is taken; this tests implementation equivalence, not official
released-weight quality or equivalence of the two dataset loaders.
"""
import argparse
import contextlib
import copy
import difflib
import functools
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  LAM_ENABLE_MANUAL_WANDB='0', CUBLAS_WORKSPACE_CONFIG=':4096:8')


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare(a):
    import numpy as np
    import torch
    import yaml
    sys.path[:0] = [str(ROOT), str(ROOT / 'vendor/lawam')]
    from vendor.dataset.robomind import RoboMINDDataset, robomind_collate
    a.artifacts.mkdir(parents=True, exist_ok=True)
    a.output.mkdir(parents=True, exist_ok=True)
    cfg = yaml.safe_load(a.config.read_text())
    upstream = json.loads((ROOT / 'vendor/lawam/UPSTREAM.json').read_text())
    head = subprocess.check_output(['git', '-C', str(a.source), 'rev-parse', 'HEAD'], text=True).strip()
    assert head == upstream['commit']
    files = {}
    for name, expected in upstream['files_sha256'].items():
        official, local = a.source / name, ROOT / 'vendor/lawam' / name
        assert sha(official) == expected, name
        files[name] = dict(official_sha256=sha(official), local_sha256=sha(local),
                           equal=sha(official) == sha(local))
        if not files[name]['equal']:
            diff = ''.join(difflib.unified_diff(official.read_text().splitlines(True),
                local.read_text().splitlines(True), fromfile='official/' + name, tofile='local/' + name))
            (a.output / (local.name + '.diff')).write_text(diff)
    options = dict(image_hw=cfg['model']['image_hw'], max_state_dim=cfg['model']['max_state_dim'],
                   num_embodiments=cfg['model']['num_embodiments'],
                   horizon_seconds=cfg['data']['horizon_seconds'],
                   tolerance_seconds=cfg['data']['tolerance_seconds'])
    stats, batches = {}, {}
    for split, key in [('train', 'train_manifest'), ('validation', 'val_manifest')]:
        ds = RoboMINDDataset(cfg['data'][key], require_observed_endpoints=split == 'validation', **options)
        missing_pairs = missing_values = tails = 0
        for i, left, right, tail in ds.pairs:
            valid = ds._arrays(i)['original_valid'][[left, right], 1:]
            missing_pairs += int(not valid.all())
            missing_values += int((~valid.astype(bool)).sum())
            tails += int(tail)
        indices = [0, len(ds) // 2]
        samples = [ds[index] for index in indices]
        batches[split] = robomind_collate(samples, max_state_dim=cfg['model']['max_state_dim'])
        stats[split] = dict(pairs=len(ds), episodes=len(ds.episodes), video_groups=len(ds.groups),
            tail_pairs=tails, pairs_with_filled_state_endpoint=missing_pairs,
            filled_raw_state_endpoint_values=missing_values, raw_state_endpoint_values=len(ds)*2*14,
            runtime_indices=indices, runtime_episodes=[s['trajectory_id'] for s in samples],
            manifest_sha256=sha(ds.path))
    torch.save(batches, a.artifacts / 'batches.pt')
    write(a.output / 'inputs.json', dict(official_commit=head, source_files=files, data=stats,
        config=str(a.config), config_sha256=sha(a.config), checkpoint=str(a.checkpoint),
        checkpoint_sha256=sha(a.checkpoint), batch_sha256=sha(a.artifacts / 'batches.pt')))
    print(json.dumps(stats), flush=True)


def run(a):
    import numpy as np
    import torch
    import yaml
    sys.path.insert(0, str(a.source.resolve()))
    import latent_action_model.core.lam_lightinng as implementation
    assert Path(implementation.__file__).resolve().is_relative_to(a.source.resolve())
    # Check all imported Stage-1 modules, not just the entry point.
    for name, module in list(sys.modules.items()):
        if name.startswith('latent_action_model.') and getattr(module, '__file__', None):
            assert Path(module.__file__).resolve().is_relative_to(a.source.resolve()), name
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    cfg = yaml.safe_load(a.config.read_text())
    optimizer = functools.partial(torch.optim.AdamW, **cfg['optimizer'])
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    model = implementation.VJEPA_LAM(**cfg['model'], optimizer=optimizer,
                                    weight_decay=cfg['optimizer']['weight_decay'])
    torch.set_float32_matmul_precision('highest')
    saved = torch.load(a.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(saved['state_dict'], strict=True)
    step = saved['global_step']
    del saved
    model.to(a.device)
    raw = torch.load(a.artifacts / 'batches.pt', weights_only=True)
    captured = {}

    def capture(name, tensor):
        captured[name] = tensor.detach().cpu().clone()

    model.lam.encoder.register_forward_hook(lambda m, args, out: capture('encoder_nodes', out))
    def decoder_input(m, args, kwargs):
        capture('z', kwargs['actions'])
        capture('current_features', kwargs['features'])
    model.lam.decoder.register_forward_pre_hook(decoder_input, with_kwargs=True)
    model.lam.decoder.register_forward_hook(lambda m, args, out: capture('prediction', out))
    results, tensors = {}, {}
    for mode, split in [('eval_fp32', 'validation'), ('train_bf16', 'train')]:
        training = mode.startswith('train')
        model.train(training)
        model.zero_grad(set_to_none=True)
        random.seed(20261113); np.random.seed(20261113); torch.manual_seed(20261113)
        torch.cuda.manual_seed_all(20261113)
        started = time.perf_counter()
        captured.clear()
        batch = model.transfer_batch_to_device(copy.deepcopy(raw[split]), torch.device(a.device), 0)
        capture('videos', batch['videos']); capture('dec_videos', batch['dec_videos'])
        with (contextlib.nullcontext() if training else torch.no_grad()):
            with torch.autocast('cuda', dtype=torch.bfloat16, enabled=training):
                loss, logs = model._compute_step(batch, vq_training=training)
            if training:
                loss.backward()
        capture('loss', loss)
        gradients = {}
        if training:
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                if param.grad is None:
                    gradients[name] = None
                    continue
                grad = param.grad.detach().cpu().contiguous()
                assert torch.isfinite(grad).all(), name
                gradients[name] = dict(shape=list(grad.shape), dtype=str(grad.dtype),
                    sha256=hashlib.sha256(grad.view(torch.uint8).numpy().tobytes()).hexdigest(),
                    norm=float(grad.norm()))
        assert all(torch.isfinite(t).all() for t in captured.values())
        results[mode] = dict(loss=float(loss.detach()),
            logs={key: float(value) for key, value in logs.items()}, gradients=gradients,
            seconds=time.perf_counter()-started)
        tensors[mode] = dict(captured)
        print(a.label, mode, results[mode]['loss'], results[mode]['seconds'], flush=True)
    torch.save(tensors, a.artifacts / f'{a.label}.pt')
    write(a.output / f'{a.label}.json', dict(source=str(a.source), module=implementation.__file__,
        checkpoint_step=step, torch=torch.__version__, device=torch.cuda.get_device_name(a.device),
        config_sha256=sha(a.config), checkpoint_sha256=sha(a.checkpoint),
        batch_sha256=sha(a.artifacts / 'batches.pt'),
        peak_cuda_bytes=torch.cuda.max_memory_allocated(a.device), results=results))


def compare(a):
    import torch
    left = torch.load(a.artifacts / 'official.pt', weights_only=True)
    right = torch.load(a.artifacts / 'local.pt', weights_only=True)
    lj, rj = [json.loads((a.output / f'{name}.json').read_text()) for name in ('official', 'local')]
    for key in ['checkpoint_sha256', 'config_sha256', 'batch_sha256', 'torch', 'device']:
        assert lj[key] == rj[key], key
    results = {}
    for mode in left:
        comparisons = {}
        assert left[mode].keys() == right[mode].keys()
        for key, value in left[mode].items():
            other = right[mode][key]
            comparisons[key] = dict(equal=torch.equal(value, other), shape=list(value.shape),
                max_abs_difference=float((value.float()-other.float()).abs().max()))
        lg, rg = [j['results'][mode]['gradients'] for j in (lj, rj)]
        assert lg.keys() == rg.keys()
        grad_differences = [key for key in lg if lg[key] != rg[key]]
        ll, rl = [j['results'][mode]['logs'] for j in (lj, rj)]
        common_logs_equal = all(ll[key] == rl[key] for key in ll.keys() & rl.keys())
        results[mode] = dict(tensors=comparisons, gradients_checked=sum(v is not None for v in lg.values()),
            gradients_absent=sum(v is None for v in lg.values()), gradient_differences=grad_differences,
            common_logs_equal=common_logs_equal, extra_local_logs=sorted(rl.keys()-ll.keys()))
        assert all(row['equal'] for row in comparisons.values()), mode
        assert not grad_differences and common_logs_equal, mode
    write(a.output / 'comparison.json', results)
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['prepare', 'run', 'compare'])
    p.add_argument('--source', type=Path)
    p.add_argument('--label', choices=['official', 'local'])
    p.add_argument('--config', type=Path, default=Path('/data2/yuki_sadasue/runs/foveated-world-models/stage1-robomind-main/resolved_config.yaml'))
    p.add_argument('--checkpoint', type=Path, default=Path('/data2/yuki_sadasue/runs/foveated-world-models/stage1-robomind-main/checkpoints/final.ckpt'))
    p.add_argument('--output', type=Path, default=ROOT / 'research/results/official-implementation-comparison-20261113')
    p.add_argument('--artifacts', type=Path, default=Path('/data2/yuki_sadasue/runs/foveated-world-models/official-implementation-comparison-20261113'))
    p.add_argument('--device', default='cuda:1')
    args = p.parse_args()
    if args.mode in ('prepare', 'run') and args.source is None:
        p.error('--source is required')
    if args.mode == 'run' and args.label is None:
        p.error('--label is required')
    globals()[args.mode](args)
