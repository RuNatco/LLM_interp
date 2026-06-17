#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

GPUS=1
CONFIG=""
CONTINUE_CKPT=""

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run.sh --gpus <N> --config <path> [options]

Options:
  --gpus N               GPUs to use: 1 (single process) or N>1 (DDP via torchrun).
  --config <path>        Path to YAML config (required).
  --continue-from <ckpt> Checkpoint path to resume training from.
                         Overrides training.init_from_checkpoint in the config.
  -h, --help             Show this help.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)           GPUS="$2";          shift 2 ;;
    --config)         CONFIG="$2";        shift 2 ;;
    --continue-from)  CONTINUE_CKPT="$2"; shift 2 ;;
    -h|--help)        usage; exit 0 ;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 1 ;;
  esac
done

if [[ -z "${CONFIG}" ]]; then
  echo "Error: --config is required." >&2
  usage >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"

if [[ -n "${CONTINUE_CKPT}" ]]; then
  export CLT_OVERRIDE_INIT_CHECKPOINT="${CONTINUE_CKPT}"
  echo "[run.sh] Resuming from: ${CONTINUE_CKPT}"
fi

if ! [[ "${GPUS}" =~ ^[0-9]+$ ]] || [[ "${GPUS}" -lt 1 ]]; then
  echo "Error: --gpus must be a positive integer (got: ${GPUS})." >&2
  exit 1
fi

if [[ "${GPUS}" -eq 1 ]]; then
  echo "[run.sh] Mode: 1-GPU  config: ${CONFIG}"
  exec python3 scripts/01_train_clt.py --config "${CONFIG}"
else
  echo "[run.sh] Mode: ${GPUS}-GPU DDP (torchrun)  config: ${CONFIG}"

  export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
  export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
  export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
  export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
  TORCH_LIB="$(python3 -c 'import torch, os; print(os.path.dirname(torch.__file__))')/lib"
  export LD_LIBRARY_PATH="${TORCH_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

  exec torchrun \
    --standalone \
    --nproc_per_node="${GPUS}" \
    scripts/01_train_clt.py \
    --config "${CONFIG}"
fi
