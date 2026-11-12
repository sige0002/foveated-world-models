"""Compare teacher-interpolated and generated states; teacher values are oracle diagnostics."""
import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'vendor/lawam')]
import torch
from research.train_cached_decoders import make_decoder, file_hash


def squared_error(a, b):
    return (a - b).square().flatten(1).mean(1)


def local_metrics(current, target, initial, generated, time, teacher_velocity, generated_velocity):
    teacher = (1 - time) * initial + time * target
    velocity_target = target - initial
    teacher_error = squared_error(teacher + (1 - time) * teacher_velocity, target)
    teacher_v_error = squared_error(teacher_velocity, velocity_target)
    # The teacher endpoint identity is algebraic, not evidence of predictive ability.
    torch.testing.assert_close(teacher_error, (1 - time) ** 2 * teacher_v_error,
                               atol=1e-6, rtol=1e-5)
    return dict(teacher_endpoint_mse=teacher_error,
                generated_endpoint_mse=squared_error(generated + (1 - time) * generated_velocity, target),
                teacher_velocity_mse=teacher_v_error,
                generated_reference_velocity_mse=squared_error(generated_velocity, velocity_target),
                state_gap=squared_error(generated, teacher),
                teacher_state_mse=squared_error(teacher, target),
                generated_state_mse=squared_error(generated, target),
                copy_mse=squared_error(current, target))


def code_swap_metrics(target, initial, generated, time, vt, vg, wrong_vt, wrong_vg):
    """Hold state/current fixed: swaps test dependence, not action semantics or conditional MI."""
    teacher = (1 - time) * initial + time * target
    velocity_target = target - initial
    teacher_gap = (squared_error(teacher + (1 - time) * wrong_vt, target)
                   - squared_error(teacher + (1 - time) * vt, target))
    velocity_gap = squared_error(wrong_vt, velocity_target) - squared_error(vt, velocity_target)
    torch.testing.assert_close(teacher_gap, (1 - time) ** 2 * velocity_gap,
                               atol=1e-6, rtol=1e-5)
    return dict(teacher_swap_endpoint_gap=teacher_gap,
                teacher_swap_velocity_gap=velocity_gap,
                teacher_code_velocity_sensitivity=squared_error(wrong_vt, vt),
                generated_swap_endpoint_gap=(
                    squared_error(generated + (1 - time) * wrong_vg, target)
                    - squared_error(generated + (1 - time) * vg, target)),
                generated_code_velocity_sensitivity=squared_error(wrong_vg, vg))


@torch.inference_mode()
def diagnose(model, data, steps, batch_size=8, code_swaps=False):
    model.eval()
    generator = torch.Generator().manual_seed(2026)
    rows = []
    for start in range(0, len(data['current']), batch_size):
        h, y, z = [data[k][start:start + batch_size] for k in ('current', 'future', 'codes')]
        wrong_indices = (torch.arange(start, start + len(h)) + 1) % len(data['codes'])
        wrong_z = data['codes'][wrong_indices]
        initial = h + model.noise_scale * torch.randn(h.shape, generator=generator)
        generated = initial.clone()
        for step in range(steps):
            time = step / steps
            t = torch.full((len(h),), time, dtype=h.dtype)
            teacher = (1 - time) * initial + time * y
            vt = model.velocity(teacher, t, h, z)
            vg = model.velocity(generated, t, h, z)
            metrics = local_metrics(h, y, initial, generated, time, vt, vg)
            if code_swaps:
                metrics.update(code_swap_metrics(
                    y, initial, generated, time, vt, vg,
                    model.velocity(teacher, t, h, wrong_z),
                    model.velocity(generated, t, h, wrong_z)))
            if step == 0:
                torch.testing.assert_close(vt, vg, atol=0, rtol=0)
            for i in range(len(h)):
                rows.append(dict(**data['rows'][start + i], time=time,
                                 **{k: float(v[i]) for k, v in metrics.items()}))
            generated = generated + vg / steps
    keys = list(metrics)  # Cache metadata may also contain older *_mse diagnostics.
    summary = []
    for step in range(steps):
        selected = [r for r in rows if r['time'] == step / steps]
        summary.append(dict(time=step / steps,
                            **{k: sum(r[k] for r in selected) / len(selected) for k in keys}))
    return dict(summary=summary, rows=rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--steps', type=int, default=4)
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--code-swaps', action='store_true',
                   help='Global cyclic code swap at fixed states; descriptive dependence test only')
    args = p.parse_args()
    if min(args.steps, args.threads) < 1 or args.output.exists():
        p.error('Positive steps/threads and a new output path required')
    torch.set_num_threads(args.threads)
    manifest = json.loads((args.cache / 'cache_manifest.json').read_text())
    saved = torch.load(args.checkpoint, weights_only=True)
    if saved['variant'] not in ('flow', 'wide_flow'):
        p.error('Velocity-trained flow or wide_flow checkpoint required')
    result = dict(checkpoint=str(args.checkpoint), checkpoint_sha256=file_hash(args.checkpoint),
                  cache_manifest=manifest, steps=args.steps, sampling_seed=2026,
                  code_swaps=args.code_swaps,
                  note='One noise draw per example. Teacher states contain true future. '
                       'Reference velocity at generated states is not a known optimal field. '
                       'Evaluation batch ordering differs from the four-draw benchmark. '
                       'Code swaps hold states fixed on the correct-code generated path; '
                       'global swaps can be out of distribution and do not identify action utility.', splits={})
    for split in ('train', 'validation'):
        path = args.cache / f'{split}.pt'
        if file_hash(path) != manifest['cache_sha256'][split]:
            raise ValueError('Cache hash mismatch')
        data = torch.load(path, weights_only=True)
        h = data['current']
        model = make_decoder(saved['variant'], saved['dim'], saved['layers'], h.shape[-1],
                             data['codes'].shape[-1], math.isqrt(h.shape[-2]))
        model.load_state_dict(saved['state_dict'], strict=True)
        result['splits'][split] = diagnose(model, data, args.steps, code_swaps=args.code_swaps)
        print(split, json.dumps(result['splits'][split]['summary']), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
