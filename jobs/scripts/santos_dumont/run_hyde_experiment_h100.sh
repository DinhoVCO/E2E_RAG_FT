#!/usr/bin/env bash
# Submit HyDE experiment(s) on ict-h100 (1 GPU, batch job).
#
# Usage (from repo root):
#   bash jobs/scripts/santos_dumont/run_hyde_experiment_h100.sh --experiment bioasq-resplit-hyde
#   bash jobs/scripts/santos_dumont/run_hyde_experiment_h100.sh --all
#   bash jobs/scripts/santos_dumont/run_hyde_experiment_h100.sh --submit-each --all
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
HYDE_SCRIPT="${ROOT}/scripts/query_expansion/hyde/run_hyde_experiment.py"

if (("$#" == 0)); then
  echo "Usage: $0 [--submit-each] <--experiment ID | --dataset NAME | --all> [args...]" >&2
  echo "       $0 --list" >&2
  exit 1
fi

if [[ "$1" == "--list" ]]; then
  exec uv run python "$HYDE_SCRIPT" --list
fi

SUBMIT_EACH=false
if [[ "$1" == "--submit-each" ]]; then
  SUBMIT_EACH=true
  shift
fi

if [[ "$1" != "--experiment" && "$1" != "--dataset" && "$1" != "--all" ]]; then
  echo "First argument must be --experiment, --dataset, or --all." >&2
  exit 1
fi

MODE="$1"
TARGET="${2:-}"
if [[ "$MODE" != "--all" && -z "$TARGET" ]]; then
  echo "Missing value after ${MODE}." >&2
  exit 1
fi

JOB_NAME="${JOB_NAME:-}"
TIME="${TIME:-12:00:00}"
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
exec ${PYTHON_RUNNER} scripts/query_expansion/hyde/run_hyde_experiment.py${extra_cmd}")
  echo "Submitted batch job ${job_id}"
  echo "Logs: tail -f ${LOG_DIR}/${job_name}-${job_id}.out"
}

if [[ "$SUBMIT_EACH" == "true" && "$MODE" == "--experiment" ]]; then
  _submit_job "${JOB_NAME:-hyde-${TARGET}}" --experiment "$TARGET" "${@:3}"
  exit 0
fi

if [[ "$SUBMIT_EACH" == "true" ]]; then
  mapfile -t experiment_ids < <(${PYTHON_RUNNER} "$HYDE_SCRIPT" --list | cut -f1)
  if [[ "$MODE" == "--dataset" ]]; then
    mapfile -t experiment_ids < <(${PYTHON_RUNNER} "$HYDE_SCRIPT" --list | grep -F "dataset=${TARGET}" | cut -f1)
  fi
  for experiment_id in "${experiment_ids[@]}"; do
    _submit_job "${JOB_NAME:-${experiment_id}}" --experiment "$experiment_id" "${@:3}"
    echo
  done
  exit 0
fi

case "$MODE" in
  --experiment) DEFAULT_JOB_NAME="hyde-${TARGET}" ;;
  --dataset) DEFAULT_JOB_NAME="hyde-${TARGET}" ;;
  --all) DEFAULT_JOB_NAME="hyde-all" ;;
esac
_submit_job "${JOB_NAME:-$DEFAULT_JOB_NAME}" "$@"
