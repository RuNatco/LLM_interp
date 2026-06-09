#!/usr/bin/env bash
set -euo pipefail

APPLY=0
ALLOW_MISSING_FINAL=0

usage() {
  cat <<'EOF'
Usage:
  bash scripts/10_backup_intermediate_outputs.sh [options]

Options:
  --apply                 Move intermediate output dirs.
  --allow-missing-final   Allow --apply even if the final output dir is absent.
  -h, --help              Show this help.

Without --apply the script only prints a dry run.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      APPLY=1
      shift
      ;;
    --allow-missing-final)
      ALLOW_MISSING_FINAL=1
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

BACKUP_ROOT="${BACKUP_ROOT:-outputs/_backup/$(date +%Y%m%d_%H%M%S)}"

INTERMEDIATE_DIRS=(
  outputs/base_clt_v0
  outputs/instruct_clt_v0
  outputs/base_clt_fidelity_v1
  outputs/base_clt_fidelity_v2
  outputs/base_clt_fidelity_v3
  outputs/base_clt_fidelity_v4
  outputs/instruct_clt_fidelity_v2
  outputs/base_clt_late_loss_v1
  outputs/base_clt_late_target_v1
  outputs/base_clt_late_target_v2
  outputs/base_clt_late_target_v3
  outputs/base_clt_recon_fidelity_v1
  outputs/base_clt_recon_fidelity_v2
  outputs/base_clt_recon_fidelity_v2_continue_v1
)

FINAL_DIR="outputs/base_clt_recon_fidelity_v2_continue_v2"

echo "Final output kept in place: ${FINAL_DIR}"
echo "Backup root: ${BACKUP_ROOT}"

if [[ ! -d "${FINAL_DIR}" ]]; then
  echo "warning: final output dir is missing: ${FINAL_DIR}" >&2

  if [[ "${APPLY}" -eq 1 && "${ALLOW_MISSING_FINAL}" -eq 0 ]]; then
    echo "Refusing to apply backup without the final output dir." >&2
    echo "Use --allow-missing-final only if this is intentional." >&2
    exit 2
  fi
fi

if [[ "${APPLY}" -eq 0 ]]; then
  echo
  echo "Dry run. Re-run with --apply to move existing intermediate output dirs."
fi

for dir in "${INTERMEDIATE_DIRS[@]}"; do
  if [[ ! -d "${dir}" ]]; then
    echo "skip missing: ${dir}"
    continue
  fi

  dest="${BACKUP_ROOT}/${dir#outputs/}"
  echo "move: ${dir} -> ${dest}"

  if [[ "${APPLY}" -eq 1 ]]; then
    mkdir -p "$(dirname "${dest}")"
    mv "${dir}" "${dest}"
  fi
done

if [[ "${APPLY}" -eq 1 ]]; then
  echo
  echo "Backup complete. Intermediate outputs moved to: ${BACKUP_ROOT}"
fi
