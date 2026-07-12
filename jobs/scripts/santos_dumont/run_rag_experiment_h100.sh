#!/usr/bin/env bash
# Submit RAG experiment(s) on ict-h100 (1 GPU, non-interactive batch job).
#
# Usage (from repo root):
#   bash jobs/scripts/santos_dumont/run_rag_experiment_h100.sh \
#     --experiment telco-dpr-emb-lora-gen-lora-top5
#
#   bash jobs/scripts/santos_dumont/run_rag_experiment_h100.sh --dataset telco-dpr
#
#   bash jobs/scripts/santos_dumont/run_rag_experiment_h100.sh --all
#
#   JOB_NAME=rag-telco TIME=08:00:00 bash jobs/scripts/santos_dumont/run_rag_experiment_h100.sh \
#     --experiment telco-dpr-emb-base-gen-lora-top5 --skip-retrieval
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if (("$#" == 0)); then
  echo "Usage: $0 <--experiment ID | --dataset NAME | --all> [run_rag_experiment.py args...]" >&2
  echo "       $0 --list" >&2
  exit 1
fi

if [[ "$1" == "--list" ]]; then
  exec uv run python "${ROOT}/scripts/generation/run_rag_experiment.py" --list
fi

if [[ "$1" != "--experiment" && "$1" != "--dataset" && "$1" != "--all" ]]; then
  echo "First argument must be --experiment, --dataset, or --all (or --list)." >&2
  exit 1
fi

MODE="$1"
TARGET="${2:-}"
if [[ "$MODE" != "--all" && -z "$TARGET" ]]; then
  echo "Missing value after ${MODE}." >&2
  exit 1
fi

case "$MODE" in
  --experiment) DEFAULT_JOB_NAME="rag-${TARGET}" ;;
  --dataset) DEFAULT_JOB_NAME="rag-${TARGET}" ;;
  --all) DEFAULT_JOB_NAME="rag-all" ;;
esac

JOB_NAME="${JOB_NAME:-$DEFAULT_JOB_NAME}"
TIME="${TIME:-12:00:00}"
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
export CUDA_VISIBLE_DEVICES=0
echo SLURM_JOB_ID=\$SLURM_JOB_ID
echo Node: \$(hostname)
echo Started: \$(date -Is)
exec ${PYTHON_RUNNER} scripts/generation/run_rag_experiment.py${EXTRA_CMD}")

echo "Submitted batch job ${JOB_ID}"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out"
echo "Cancel:   scancel ${JOB_ID}"
