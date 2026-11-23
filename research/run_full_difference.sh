#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
audit=/media/sadasue/SSD1/foveated-world-models/research/official-data-contract-audit-20261113
export PYTHONPATH="$PWD/vendor/lawam:$PWD:$audit/LaWAM:$audit/python-deps:$audit/pytorch3d"
export HF_HOME="$audit/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export LAM_ENABLE_MANUAL_WANDB=0 WANDB_MODE=disabled OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 LAM_WORKER_OMP_THREADS=1 TOKENIZERS_PARALLELISM=false
exec .venv-lawam-stage1/bin/python -m research.run_full_difference fit --config configs/stage1_difference_30k.yaml "$@"
