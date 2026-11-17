#!/usr/bin/env bash
# Submit MTEB retrieval experiment(s) on ict-h100 (1 GPU, non-interactive batch job).
#
# Usage (from repo root):
#   bash jobs/scripts/santos_dumont/run_mteb_experiment_h100.sh \
#     --experiment telco-dpr-emb-lora
#
#   bash jobs/scripts/santos_dumont/run_mteb_experiment_h100.sh --dataset telco-dpr
#
#   bash jobs/scripts/santos_dumont/run_mteb_experiment_h100.sh --all
#
#   JOB_NAME=mteb-qasper TIME=04:00:00 bash jobs/scripts/santos_dumont/run_mteb_experiment_h100.sh \
#     --experiment qasper-emb-base
#
# Submit one GPU job per experiment (parallel runs):
#   bash jobs/scripts/santos_dumont/run_mteb_experiment_h100.sh --submit-each --all
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if (("$#" == 0)); then
  echo "Usage: $0 <--experiment ID | --dataset NAME | --all> [run_mteb_experiment.py args...]" >&2
  echo "       $0 --list" >&2
  echo "       $0 --submit-each <--experiment ID | --dataset NAME | --all>" >&2
  exit 1
fi

if [[ "$1" == "--list" ]]; then
  exec uv run python "${ROOT}/scripts/evaluation/mteb/run_mteb_experiment.py" --list
fi

SUBMIT_EACH=false
if [[ "$1" == "--submit-each" ]]; then
  SUBMIT_EACH=true
  shift
  if (("$#" == 0)); then
    echo "Missing target after --submit-each." >&2
    exit 1
  fi
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

JOB_NAME="${JOB_NAME:-}"
TIME="${TIME:-04:00:00}"
ACCOUNT="${ACCOUNT:-smartassistant}"
PARTITION="${PARTITION:-ict-h100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-65536M}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/slurm}"
PYTHON_RUNNER="${PYTHON_RUNNER:-uv run python}"

mkdir -p "$LOG_DIR"

_submit_job() {
  local job_name="$1"
  shift
  local extra_cmd
  extra_cmd="$(printf ' %q' "$@")"

  local job_id
  job_id=$(sbatch --parsable \
    --job-name="$job_name" \
    --account="$ACCOUNT" \
    --partition="$PARTITION" \
    --time="$TIME" \
    --cpus-per-task="$CPUS_PER_TASK" \
    --mem="$MEM" \
    --gres=gpu:1 \
    --chdir="$ROOT" \
    --output="${LOG_DIR}/${job_name}-%j.out" \
    --error="${LOG_DIR}/${job_name}-%j.err" \
    --wrap="set -euo pipefail
cd '${ROOT}'
export CUDA_VISIBLE_DEVICES=0
echo SLURM_JOB_ID=\$SLURM_JOB_ID
echo Node: \$(hostname)
echo Started: \$(date -Is)
exec ${PYTHON_RUNNER} scripts/evaluation/mteb/run_mteb_experiment.py${extra_cmd}")

  echo "Submitted batch job ${job_id}"
  echo "Monitor:  squeue -j ${job_id}"
  echo "Logs:     tail -f ${LOG_DIR}/${job_name}-${job_id}.out"
  echo "Cancel:   scancel ${job_id}"
}

if [[ "$SUBMIT_EACH" == "true" && "$MODE" == "--experiment" ]]; then
  job_name="${JOB_NAME:-mteb-${TARGET}}"
  _submit_job "$job_name" --experiment "$TARGET" "${@:3}"
  exit 0
fi

if [[ "$SUBMIT_EACH" == "true" ]]; then
  list_cmd=(${PYTHON_RUNNER} "${ROOT}/scripts/evaluation/mteb/run_mteb_experiment.py" --list)
  if [[ "$MODE" == "--dataset" ]]; then
    mapfile -t experiment_ids < <("${list_cmd[@]}" | grep -F "dataset=${TARGET}" | cut -f1)
  else
    mapfile -t experiment_ids < <("${list_cmd[@]}" | cut -f1)
  fi

  if (("${#experiment_ids[@]}" == 0)); then
    echo "No experiments matched ${MODE} ${TARGET}." >&2
    exit 1
  fi

  for experiment_id in "${experiment_ids[@]}"; do
    job_name="${JOB_NAME:-mteb-${experiment_id}}"
    _submit_job "$job_name" --experiment "$experiment_id" "${@:3}"
    echo
  done
  exit 0
fi

case "$MODE" in
  --experiment) DEFAULT_JOB_NAME="mteb-${TARGET}" ;;
  --dataset) DEFAULT_JOB_NAME="mteb-${TARGET}" ;;
  --all) DEFAULT_JOB_NAME="mteb-all" ;;
esac

JOB_NAME="${JOB_NAME:-$DEFAULT_JOB_NAME}"
_submit_job "$JOB_NAME" "$@"
