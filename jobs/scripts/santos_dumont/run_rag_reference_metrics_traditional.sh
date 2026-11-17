#!/usr/bin/env bash
# Submit traditional reference metrics on ict-h100 (1 GPU requested for QOS; CPU-only eval).
#
# Evaluates BLEU, ROUGE, CHRF, exact match, string presence, and
# non-LLM string similarity between response and reference.
#
# Usage (from repo root):
#   bash jobs/scripts/santos_dumont/run_rag_reference_metrics_traditional.sh --all
#
#   bash jobs/scripts/santos_dumont/run_rag_reference_metrics_traditional.sh \
#     --dataset telco-dpr
#
#   bash jobs/scripts/santos_dumont/run_rag_reference_metrics_traditional.sh \
#     --generation-dir datasets/generated/telco-dpr/generation-b128-vllm-offline-b128
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT="${ROOT}/scripts/evaluation/ragas/run_rag_reference_metrics_traditional.py"

if (("$#" == 0)); then
  echo "Usage: $0 <--generation-dir PATH | --dataset NAME | --all> [extra args...]" >&2
  exit 1
fi

if [[ "$1" != "--generation-dir" && "$1" != "--dataset" && "$1" != "--all" ]]; then
  echo "First argument must be --generation-dir, --dataset, or --all." >&2
  exit 1
fi

MODE="$1"
TARGET="${2:-}"
if [[ "$MODE" != "--all" && -z "$TARGET" ]]; then
  echo "Missing value after ${MODE}." >&2
  exit 1
fi

case "$MODE" in
  --generation-dir) DEFAULT_JOB_NAME="rag-ref-trad" ;;
  --dataset) DEFAULT_JOB_NAME="rag-ref-trad-${TARGET}" ;;
  --all) DEFAULT_JOB_NAME="rag-ref-trad-all" ;;
esac

JOB_NAME="${JOB_NAME:-$DEFAULT_JOB_NAME}"
TIME="${TIME:-02:00:00}"
ACCOUNT="${ACCOUNT:-smartassistant}"
PARTITION="${PARTITION:-ict-h100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-65536M}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/slurm}"
PYTHON_RUNNER="${PYTHON_RUNNER:-uv run python}"

mkdir -p "$LOG_DIR"

EXTRA_CMD="$(printf ' %q' "$@")"

JOB_ID=$(sbatch --parsable \
  --job-name="$JOB_NAME" \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --time="$TIME" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM" \
  --gres=gpu:1 \
  --chdir="$ROOT" \
  --output="${LOG_DIR}/${JOB_NAME}-%j.out" \
  --error="${LOG_DIR}/${JOB_NAME}-%j.err" \
  --wrap="set -euo pipefail
cd '${ROOT}'
echo SLURM_JOB_ID=\$SLURM_JOB_ID
echo Node: \$(hostname)
echo Started: \$(date -Is)
exec ${PYTHON_RUNNER} '${SCRIPT}'${EXTRA_CMD}")

echo "Submitted batch job ${JOB_ID}"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out"
echo "Cancel:   scancel ${JOB_ID}"
