#!/usr/bin/env bash
# Submit two-stage context MTEB evaluation on ict-h100 (1 GPU, batch job).
#
# Stage 1: retrieve candidate docs with Instruct+Query (no context).
# Stage 2: rebuild queries with ## Context blocks and run MTEB for each --context-k.
#
# Usage (from repo root):
#   bash jobs/scripts/santos_dumont/run_mteb_context_retrieval_h100.sh --dataset qasper
#
#   bash jobs/scripts/santos_dumont/run_mteb_context_retrieval_h100.sh \
#     --dataset qasper --stage1-only
#
#   bash jobs/scripts/santos_dumont/run_mteb_context_retrieval_h100.sh \
#     --dataset qasper \
#     --lora-path DinoStackAI/Qwen3-Emb-4b-lora-ctx-qasper
#
#   bash jobs/scripts/santos_dumont/run_mteb_context_retrieval_h100.sh \
#     --dataset qasper \
#     --lora-path models/qwen3-embedding-4b-lora-ctx/qasper-ctx-b32-e10/final \
#     --run-label qasper-ctx-b32-e10
#
#   bash jobs/scripts/santos_dumont/run_mteb_context_retrieval_h100.sh --all
#
#   bash jobs/scripts/santos_dumont/run_mteb_context_retrieval_h100.sh --submit-each --all
#
#   TIME=08:00:00 JOB_NAME=mteb-ctx-qasper \
#     bash jobs/scripts/santos_dumont/run_mteb_context_retrieval_h100.sh --dataset qasper
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

HUB_ORG="${HF_ORG:-DinoStackAI}"
ALL_DATASETS=(bioasq-resplit narrativeqa qasper telco-dpr)

default_lora_path() {
  echo "${HUB_ORG}/Qwen3-Emb-4b-lora-ctx-${1}"
}

_list_datasets() {
  echo "Supported datasets (default LoRA repo: ${HUB_ORG}/Qwen3-Emb-4b-lora-ctx-<dataset>):"
  for dataset in "${ALL_DATASETS[@]}"; do
    echo "  ${dataset} -> $(default_lora_path "${dataset}")"
  done
  echo
  echo "Defaults: stage1-top-k=10, context-k=1 3 5 7 10"
  echo "Note: qasper uses paper-scoped retrieval by default (stage 1 and stage 2)."
  echo "Use --stage1-only for MTEB with Instruct+Query only (no context docs, no stage 2)."
}

if (("$#" == 0)); then
  echo "Usage: $0 <--dataset NAME [--lora-path PATH] | --all> [run_mteb_context_retrieval.py args...]" >&2
  echo "       $0 --list" >&2
  echo "       $0 --submit-each <--dataset NAME | --all>" >&2
  exit 1
fi

if [[ "$1" == "--list" ]]; then
  _list_datasets
  exit 0
fi

SUBMIT_EACH=false
MODE=""
DATASET=""
LORA_PATH=""
EXTRA_ARGS=()

while (("$#" > 0)); do
  case "$1" in
    --submit-each)
      SUBMIT_EACH=true
      shift
      ;;
    --all)
      MODE="--all"
      shift
      ;;
    --dataset)
      MODE="--dataset"
      DATASET="${2:?Missing value after --dataset}"
      shift 2
      ;;
    --lora-path)
      LORA_PATH="${2:?Missing value after --lora-path}"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -z "$MODE" ]]; then
  echo "First argument must be --dataset or --all (or pass --list)." >&2
  exit 1
fi

if [[ "$MODE" == "--dataset" && -z "$DATASET" ]]; then
  echo "Missing value after --dataset." >&2
  exit 1
fi

if [[ "$MODE" == "--all" && -n "$LORA_PATH" ]]; then
  echo "Warning: --lora-path is ignored with --all; each dataset uses its default Hub adapter." >&2
  LORA_PATH=""
fi

JOB_NAME="${JOB_NAME:-}"
TIME="${TIME:-06:00:00}"
ACCOUNT="${ACCOUNT:-smartassistant}"
PARTITION="${PARTITION:-ict-h100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-65536M}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/slurm}"
PYTHON_RUNNER="${PYTHON_RUNNER:-uv run python}"

mkdir -p "$LOG_DIR"

_build_python_cmd() {
  local dataset="$1"
  local lora_path="$2"
  shift 2
  printf ' %q' \
    --dataset "$dataset" \
    --lora-path "$lora_path" \
    "$@"
}

_submit_job() {
  local job_name="$1"
  shift
  local extra_cmd
  extra_cmd="$(_build_python_cmd "$@")"

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
exec ${PYTHON_RUNNER} scripts/evaluation/mteb/run_mteb_context_retrieval.py${extra_cmd}")

  echo "Submitted batch job ${job_id}"
  echo "Monitor:  squeue -j ${job_id}"
  echo "Logs:     tail -f ${LOG_DIR}/${job_name}-${job_id}.out"
  echo "Cancel:   scancel ${job_id}"
}

_submit_all_in_one_job() {
  local job_name="$1"
  shift

  local body=""
  for dataset in "${ALL_DATASETS[@]}"; do
    local lora_path
    lora_path="$(default_lora_path "${dataset}")"
    local cmd
    cmd="$(_build_python_cmd "$dataset" "$lora_path" "$@")"
    body+="echo '=== ${dataset} ==='; "
    body+="${PYTHON_RUNNER} scripts/evaluation/mteb/run_mteb_context_retrieval.py${cmd} || exit 1; "
  done

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
${body}")

  echo "Submitted batch job ${job_id}"
  echo "Monitor:  squeue -j ${job_id}"
  echo "Logs:     tail -f ${LOG_DIR}/${job_name}-${job_id}.out"
  echo "Cancel:   scancel ${job_id}"
}

if [[ "$SUBMIT_EACH" == "true" ]]; then
  datasets=()
  if [[ "$MODE" == "--dataset" ]]; then
    datasets=("$DATASET")
  else
    datasets=("${ALL_DATASETS[@]}")
  fi

  for dataset in "${datasets[@]}"; do
    lora_path="${LORA_PATH:-$(default_lora_path "${dataset}")}"
    job_name="${JOB_NAME:-mteb-ctx-${dataset}}"
    _submit_job "$job_name" "$dataset" "$lora_path" "${EXTRA_ARGS[@]}"
    echo
  done
  exit 0
fi

if [[ "$MODE" == "--dataset" ]]; then
  lora_path="${LORA_PATH:-$(default_lora_path "${DATASET}")}"
  job_name="${JOB_NAME:-mteb-ctx-${DATASET}}"
  _submit_job "$job_name" "$DATASET" "$lora_path" "${EXTRA_ARGS[@]}"
  exit 0
fi

job_name="${JOB_NAME:-mteb-ctx-all}"
_submit_all_in_one_job "$job_name" "${EXTRA_ARGS[@]}"
