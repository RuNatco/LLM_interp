#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_BACKUP=0
SKIP_TRAIN=0
SKIP_DEEP_TRACE=0
ALLOW_EXISTING=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/11_run_final_training_pipeline.sh [options]

Options:
  --backup-intermediate   Move intermediate output dirs to outputs/_backup after a successful run.
  --skip-train            Skip CLT training and run final eval / Deep Trace on existing checkpoint.
  --skip-deep-trace       Skip Deep Trace prompt suite.
  --allow-existing        Allow training into existing output dirs.
  -h, --help              Show this help.

Default behavior refuses to train into existing output dirs, because training
metrics are appended and checkpoints can be overwritten. Use the backup script
or --allow-existing when that is intentional.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --backup-intermediate)
      RUN_BACKUP=1
      shift
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
    echo "Move old intermediate dirs first:" >&2
    echo "  bash scripts/10_backup_intermediate_outputs.sh --apply" >&2
    echo >&2
    echo "Or re-run with --allow-existing if overwriting/appending is intentional." >&2
    exit 2
  fi
fi

if [[ "${SKIP_TRAIN}" -eq 0 ]]; then
  echo "Stage 1/3: train 2048-feature reconstruction baseline"
  python3 scripts/01_train_clt.py \
    --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2.yaml

  echo "Stage 2/3: continue from reconstruction baseline"
  python3 scripts/01_train_clt.py \
    --config configs/qwen2_5_0_5b_base_clt_recon_fidelity_v2_continue_v1.yaml

  echo "Stage 3/3: final low-LR continuation"
  python3 scripts/01_train_clt.py \
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
fi

if [[ "${RUN_BACKUP}" -eq 1 ]]; then
  echo "Backup intermediate outputs"
  bash scripts/10_backup_intermediate_outputs.sh --apply
fi

echo
echo "Final checkpoint:"
echo "  outputs/base_clt_recon_fidelity_v2_continue_v2/clt_final.pt"
echo "Final metrics:"
echo "  outputs/base_clt_recon_fidelity_v2_continue_v2/replacement_eval_metrics.json"
echo "Final Deep Trace summary:"
echo "  outputs/base_clt_recon_fidelity_v2_continue_v2/deep_trace_suite_summary.json"
