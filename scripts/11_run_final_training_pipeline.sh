#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

SKIP_TRAIN=0
SKIP_DEEP_TRACE=0
ALLOW_EXISTING=0
GPUS=1

usage() {
  cat <<'EOF'
Usage:
  bash scripts/11_run_final_training_pipeline.sh [options]

Options:
  --gpus N                Number of GPUs for training (default: 1).
                          N=1 runs a single process; N>1 uses DDP via torchrun.
                          Eval / Deep Trace always run on a single process.
  --skip-train            Skip CLT training and run final eval / Deep Trace on existing checkpoint.
  --skip-deep-trace       Skip Deep Trace prompt suite.
  --allow-existing        Allow training into existing output dirs.
  -h, --help              Show this help.

Default behavior refuses to train into existing output dirs, because training
metrics are appended and checkpoints can be overwritten. Use --allow-existing
when that is intentional.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpus)
      GPUS="$2"
      shift 2
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-deep-trace)
      SKIP_DEEP_TRACE=1
      shift
      ;;
    --allow-existing)
      ALLOW_EXISTING=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/.cache/huggingface}"

if ! [[ "${GPUS}" =~ ^[0-9]+$ ]] || [[ "${GPUS}" -lt 1 ]]; then
  echo "Error: --gpus must be a positive integer (got: ${GPUS})." >&2
  exit 2
fi

STAGE_OUTPUT_DIRS=(
  outputs/base_clt_recon_fidelity_v2
  outputs/base_clt_recon_fidelity_v2_continue_v1
  outputs/base_clt_recon_fidelity_v2_continue_v2
)

if [[ "${SKIP_TRAIN}" -eq 0 && "${ALLOW_EXISTING}" -eq 0 ]]; then
  existing_dirs=()
  for dir in "${STAGE_OUTPUT_DIRS[@]}"; do
    if [[ -e "${dir}" ]]; then
      existing_dirs+=("${dir}")
    fi
  done

  if [[ "${#existing_dirs[@]}" -gt 0 ]]; then
    echo "Refusing to train into existing output dirs:" >&2
    printf '  %s\n' "${existing_dirs[@]}" >&2
    echo >&2
    echo "Move or remove the old intermediate dirs first." >&2
    echo >&2
    echo "Or re-run with --allow-existing if overwriting/appending is intentional." >&2
    exit 2
  fi
fi

if [[ "${SKIP_TRAIN}" -eq 0 ]]; then
  echo "Stage 1/3: train 2048-feature reconstruction baseline (${GPUS} GPU)"
  bash scripts/run.sh --gpus "${GPUS}" \
    --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2.yaml

  echo "Stage 2/3: continue from reconstruction baseline (${GPUS} GPU)"
  bash scripts/run.sh --gpus "${GPUS}" \
    --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v1.yaml

  echo "Stage 3/3: final low-LR continuation (${GPUS} GPU)"
  bash scripts/run.sh --gpus "${GPUS}" \
    --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v2.yaml
fi

echo "Evaluate final replacement model"
python3 scripts/02_eval_replacement_model.py \
  --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v2.yaml

if [[ "${SKIP_DEEP_TRACE}" -eq 0 ]]; then
  echo "Build final Deep Trace prompt suite"
  python3 scripts/09_build_deep_trace_prompt_suite.py \
    --checkpoint outputs/base_clt_recon_fidelity_v2_continue_v2/clt_final.pt \
    --output-dir outputs/base_clt_recon_fidelity_v2_continue_v2/deep_trace_suite \
    --summary-output outputs/base_clt_recon_fidelity_v2_continue_v2/deep_trace_suite_summary.json \
    --max-feature-nodes 64 \
    --top-error-nodes 8 \
    --causal-top-k 8

  echo "Render Deep Trace graphs"
  python3 scripts/12_visualize_deep_trace_graph.py \
    outputs/base_clt_recon_fidelity_v2_continue_v2/deep_trace_suite/*.json \
    || echo "Graph rendering skipped (matplotlib missing or no graphs)"
fi

echo
echo "Final checkpoint:"
echo "  outputs/base_clt_recon_fidelity_v2_continue_v2/clt_final.pt"
echo "Final metrics:"
echo "  outputs/base_clt_recon_fidelity_v2_continue_v2/replacement_eval_metrics.json"
echo "Final Deep Trace summary:"
echo "  outputs/base_clt_recon_fidelity_v2_continue_v2/deep_trace_suite_summary.json"
echo "Deep Trace graph images:"
echo "  outputs/base_clt_recon_fidelity_v2_continue_v2/deep_trace_suite/*.svg"
