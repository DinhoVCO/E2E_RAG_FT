#!/usr/bin/env bash
# Submit RFG experiment(s) on ict-h100 — one GPU job per (experiment, k).
#
# Parallelizes the slow expansion stage by running each k value in a separate job.
# Each job runs expansion + stage-2 MTEB for a single k (default: 1, 3, 5, 7, 10).
#
# Usage (from repo root):
#   bash jobs/scripts/santos_dumont/run_rfg_by_k_h100.sh \
#     --experiment narrativeqa-rfg-emb-base-gen-base --skip-stage1
#
#   bash jobs/scripts/santos_dumont/run_rfg_by_k_h100.sh \
#     --dataset narrativeqa --skip-stage1
#
#   bash jobs/scripts/santos_dumont/run_rfg_by_k_h100.sh --list
#
#   # Only k=10 for one experiment:
#   K_VALUES=10 TIME=04:00:00 bash jobs/scripts/santos_dumont/run_rfg_by_k_h100.sh \
#     --experiment narrativeqa-rfg-emb-base-gen-base --skip-stage1
#
#   # narrativeqa: 4 experiments × 5 k = 20 parallel jobs:
#   TIME=04:00:00 bash jobs/scripts/santos_dumont/run_rfg_by_k_h100.sh \
#     --dataset narrativeqa --skip-stage1
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
RFG_SCRIPT="${ROOT}/scripts/query_expansion/run_rfg_experiment.py"

if (("$#" == 0)); then
  echo "Usage: $0 <--experiment ID | --dataset NAME | --all> [run_rfg_experiment.py args...]" >&2
  echo "       $0 --list" >&2
  echo "Env:   K_VALUES (default: 1 3 5 7 10), TIME (default: 04:00:00 per k job)" >&2
  exit 1
fi

if [[ "$1" == "--list" ]]; then
  exec uv run python "$RFG_SCRIPT" --list
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

K_VALUES="${K_VALUES:-1 3 5 7 10}"
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
exec ${PYTHON_RUNNER} scripts/query_expansion/run_rfg_experiment.py${extra_cmd}")

  echo "Submitted batch job ${job_id} (${job_name})"
  echo "Monitor:  squeue -j ${job_id}"
  echo "Logs:     tail -f ${LOG_DIR}/${job_name}-${job_id}.out"
}

mapfile -t experiment_ids < <(
  list_cmd=(${PYTHON_RUNNER} "$RFG_SCRIPT" --list)
  if [[ "$MODE" == "--experiment" ]]; then
    printf '%s\n' "$TARGET"
  elif [[ "$MODE" == "--dataset" ]]; then
    "${list_cmd[@]}" | grep -F "dataset=${TARGET}" | cut -f1
  else
    "${list_cmd[@]}" | cut -f1
  fi
)

if (("${#experiment_ids[@]}" == 0)); then
  echo "No experiments matched ${MODE} ${TARGET:-}." >&2
  exit 1
fi

read -r -a k_list <<< "$K_VALUES"
extra_args=("${@:3}")
total_jobs=$((${#experiment_ids[@]} * ${#k_list[@]}))

echo "Submitting ${total_jobs} job(s): ${#experiment_ids[@]} experiment(s) × ${#k_list[@]} k value(s)"
echo "k values: ${k_list[*]}"
echo "time limit per job: ${TIME}"
echo

submitted=0
for experiment_id in "${experiment_ids[@]}"; do
  for k in "${k_list[@]}"; do
    job_name="${JOB_NAME:-rfg-${experiment_id}-k${k}}"
    _submit_job "$job_name" \
      --experiment "$experiment_id" \
      --expansion-k "$k" \
      "${extra_args[@]}"
    echo
    submitted=$((submitted + 1))
  done
done

echo "Submitted ${submitted} batch job(s)."
