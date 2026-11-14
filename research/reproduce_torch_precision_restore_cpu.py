"""CPU-only repeated reproduction of the continuation precision-context bug.

The evaluate stand-in executes the exact precision API sequence around a CPU
matrix multiply. It does not run the model, CUDA kernels, or optimizer updates.
"""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot():
    state = {}
    queries = {
        'global_matmul_precision': torch.get_float32_matmul_precision,
        'generic_precision': lambda: torch.backends.fp32_precision,
        'cuda_matmul_precision': lambda: torch.backends.cuda.matmul.fp32_precision,
        'mkldnn_matmul_precision': lambda: torch.backends.mkldnn.matmul.fp32_precision,
        'cudnn_precision': lambda: torch.backends.cudnn.fp32_precision,
        'cudnn_conv_precision': lambda: torch.backends.cudnn.conv.fp32_precision,
        'cuda_allow_tf32': lambda: torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': lambda: torch.backends.cudnn.allow_tf32,
    }
    for key, query in queries.items():
        try:
            state[key] = query()
        except RuntimeError as error:
            state[key] = {'error_type': type(error).__name__, 'message': str(error)}
    return state


@contextmanager
def evaluation_precision_context(order):
    precision = torch.get_float32_matmul_precision()
    tf32 = torch.backends.cuda.matmul.allow_tf32
    cudnn_tf32 = torch.backends.cudnn.allow_tf32
    try:
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        assert torch.get_float32_matmul_precision() == 'highest'
        yield
    finally:
        if order == 'old':
            torch.set_float32_matmul_precision(precision)
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
        elif order == 'corrected':
            torch.backends.cuda.matmul.allow_tf32 = tf32
            torch.backends.cudnn.allow_tf32 = cudnn_tf32
            torch.set_float32_matmul_precision(precision)
            assert torch.get_float32_matmul_precision() == precision
            assert torch.backends.cuda.matmul.allow_tf32 == tf32
            assert torch.backends.cudnn.allow_tf32 == cudnn_tf32
        else:
            raise ValueError(order)


def run_case(order, left, right):
    # Reset each case to the same consistent starting state.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('medium')
    initial = snapshot()
    records = []
    for iteration in range(1, 4):
        cpu_rng = torch.get_rng_state().clone()
        record = {'evaluate_call': iteration, 'before': snapshot()}
        try:
            with evaluation_precision_context(order), torch.inference_mode():
                record['inside'] = snapshot()
                output = left @ right
                record['CPU_matmul_executed'] = True
                record['CPU_result_sha256'] = hashlib.sha256(output.contiguous().numpy().tobytes()).hexdigest()
            record['context_completed'] = True
        except RuntimeError as error:
            record.update(context_completed=False, error_type=type(error).__name__, error=str(error))
        record['after'] = snapshot()
        record['cpu_rng_unchanged'] = torch.equal(cpu_rng, torch.get_rng_state())
        record['cuda_initialized'] = torch.cuda.is_initialized()
        records.append(record)
    backend_keys = set(initial) - {'global_matmul_precision'}
    backend_states_match = all(all(record['after'][key] == initial[key] for key in backend_keys) for record in records)
    if order == 'old':
        assert records[0]['context_completed']
        assert isinstance(records[0]['after']['global_matmul_precision'], dict)
        assert not records[1]['context_completed'] and not records[2]['context_completed']
    else:
        assert all(record['context_completed'] and record['after'] == initial for record in records)
    assert all(record['cpu_rng_unchanged'] and not record['cuda_initialized'] for record in records)
    return {'initial': initial, 'records': records,
            'public_backend_settings_match_initial_after_every_call': backend_states_match}


def main():
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(712)
    left = torch.randn(128, 128, generator=generator)
    right = torch.randn(128, 128, generator=generator)
    cases = {order: run_case(order, left, right) for order in ['old', 'corrected']}
    old_hash = cases['old']['records'][0]['CPU_result_sha256']
    assert all(row['CPU_result_sha256'] == old_hash for row in cases['corrected']['records'])
    root = Path(__file__).resolve().parents[1]
    report = {
        'python_executable': sys.executable, 'python_version': sys.version,
        'torch_version': torch.__version__, 'torch_compiled_cuda_version': torch.version.cuda,
        'torch_path': torch.__file__, 'torch_python_init_sha256': digest(torch.__file__),
        'original_training_runner': str(root / 'research/train_robotwin_data_step_continuation.py'),
        'original_training_runner_sha256': digest(root / 'research/train_robotwin_data_step_continuation.py'),
        'script_sha256': digest(__file__), 'calls_per_case': 3,
        'cases': cases, 'cuda_initialized': torch.cuda.is_initialized(),
        'scope': [
            'Exact continuation precision API enter/restore sequence, with CPU matmul replacing GPU model inference.',
            'Old order completes the first evaluation and leaves global getter inconsistent; second and third evaluate calls fail before matrix multiplication.',
            'Corrected order completes all three calls and restores every queried precision flag exactly.',
            'Public per-backend settings after old restore still match the initial medium configuration in this installed build.',
            'Therefore the observed getter failure alone is not evidence that GPU math or learned weights changed after update500.',
            'CPU-only reproduction cannot establish actual GPU kernel choices, arithmetic equality, gradients, optimizer trajectories, or bitwise equivalence for updates501..1000.',
            'Mixed-API global precision state is invalid for reliable introspection and must be fixed before subsequent evaluation.',
            'No model training, CUDA initialization, active-run edits, or optimizer updates.'
        ]}
    output = root / 'research/results/robotwin-precision-restoration-20261114.json'
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'torch': torch.__version__,
                      'old_completed': [r['context_completed'] for r in cases['old']['records']],
                      'corrected_completed': [r['context_completed'] for r in cases['corrected']['records']],
                      'old_public_backend_values_still_match': cases['old']['public_backend_settings_match_initial_after_every_call'],
                      'cuda_initialized': report['cuda_initialized']}, indent=2))


if __name__ == '__main__':
    main()
