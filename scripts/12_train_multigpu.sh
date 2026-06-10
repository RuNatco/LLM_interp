#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Multi-GPU training launcher (torchrun).
#
# Usage:
#   bash scripts/12_train_multigpu.sh --gpus 4 --config configs/qwen2_5_0_5b_4096f_v1_4gpu.yaml
#   bash scripts/12_train_multigpu.sh --gpus 2 --config configs/qwen2_5_0_5b_4096f_v1_2gpu.yaml
#
# Optional flags:
#   --continue-ckpt <path>   Override init_from_checkpoint in the config.
#   --skip-eval              Skip replacement eval after training.
# ---------------------------------------------------------------------------
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPUS=4
CONFIG=""
CONTINUE_CKPT=""
SKIP_EVAL=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/12_train_multigpu.sh --gpus <N> --config <path> [options]

Options:
  --gpus N              Number of GPUs (default: 4).
  --config <path>       Path to YAML config (required).
  --continue-ckpt <p>   Override init_from_checkpoint path in the config.
  --skip-eval           Skip replacement eval after training.
  -h, --help            Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)       GPUS="$2";          shift 2 ;;
    --config)     CONFIG="$2";        shift 2 ;;
    --continue-ckpt) CONTINUE_CKPT="$2"; shift 2 ;;
    --skip-eval)  SKIP_EVAL=1;        shift ;;
    -h|--help)    usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  echo "Error: --config is required." >&2
  usage >&2
  exit 2
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"

# Faster NCCL on NVLink machines
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1

TRAIN_ARGS=(--config "${CONFIG}")
if [[ -n "${CONTINUE_CKPT}" ]]; then
  # Patch the config on the fly via env var — train_clt reads it if present
  export CLT_OVERRIDE_INIT_CHECKPOINT="${CONTINUE_CKPT}"
fi

echo "Launching training: ${GPUS} GPU(s), config=${CONFIG}"
torchrun \
  --standalone \
  --nproc_per_node="${GPUS}" \
  scripts/01_train_clt.py \
  "${TRAIN_ARGS[@]}"

if [[ "${SKIP_EVAL}" -eq 0 ]]; then
  echo "Running replacement eval (rank 0 only)..."
  python3 scripts/02_eval_replacement_model.py --config "${CONFIG}"
fi

echo "Done."
