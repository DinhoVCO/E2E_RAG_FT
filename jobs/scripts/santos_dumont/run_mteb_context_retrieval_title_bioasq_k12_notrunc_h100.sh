#!/usr/bin/env bash
# BioASQ-resplit title context MTEB: stage-2 k=1 and k=2, no document truncation.
#
# Output: results/mteb/context_title_notrunc/bioasq-resplit/<run-label>/
# Revisions: ctx-lora-bioasq-resplit-k{k}-title-notrunc
#
# Usage (from repo root):
#   bash jobs/scripts/santos_dumont/run_mteb_context_retrieval_title_bioasq_k12_notrunc_h100.sh
#
#   bash jobs/scripts/santos_dumont/run_mteb_context_retrieval_title_bioasq_k12_notrunc_h100.sh \
#     --lora-path models/qwen3-embedding-4b-lora-ctx/bioasq-resplit-ctx-b32-e10/final \
#     --run-label bioasq-resplit-ctx-b32-e10
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

HUB_ORG="${HF_ORG:-DinoStackAI}"
DATASET="bioasq-resplit"
DEFAULT_LORA_PATH="${HUB_ORG}/Qwen3-Emb-4b-lora-ctx-${DATASET}"

JOB_NAME="${JOB_NAME:-mteb-ctx-title-${DATASET}-k12-notrunc}"
TIME="${TIME:-06:00:00}"
ACCOUNT="${ACCOUNT:-smartassistant}"
PARTITION="${PARTITION:-ict-h100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-65536M}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/slurm}"
PYTHON_RUNNER="${PYTHON_RUNNER:-uv run python}"

LORA_PATH=""
RUN_LABEL=""
EXTRA_ARGS=()

while (("$#" > 0)); do
  case "$1" in
    --lora-path)
      LORA_PATH="${2:?Missing value after --lora-path}"
      shift 2
      ;;
    --run-label)
      RUN_LABEL="${2:?Missing value after --run-label}"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

LORA_PATH="${LORA_PATH:-$DEFAULT_LORA_PATH}"
if [[ -z "$RUN_LABEL" ]]; then
  RUN_LABEL="$(basename "${LORA_PATH%/}")"
fi

mkdir -p "$LOG_DIR"

cmd=(
  --dataset "$DATASET"
  --lora-path "$LORA_PATH"
  --run-label "$RUN_LABEL"
  --context-k 1 2
  --stage1-top-k 2
  --no-truncate-stage2-docs
)
if (("${#EXTRA_ARGS[@]}" > 0)); then
  cmd+=("${EXTRA_ARGS[@]}")
fi

extra_cmd="$(printf ' %q' "${cmd[@]}")"

job_id=$(sbatch --parsable \
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
exec ${PYTHON_RUNNER} scripts/evaluation/mteb/run_mteb_context_retrieval_title.py${extra_cmd}")

echo "Submitted batch job ${job_id}"
echo "Dataset: ${DATASET}"
echo "LoRA: ${LORA_PATH}"
echo "Run label: ${RUN_LABEL}"
echo "Stage 2: k=1,2 (no doc truncation)"
echo "Output: results/mteb/context_title_notrunc/${DATASET}/${RUN_LABEL}/"
echo "Monitor:  squeue -j ${job_id}"
echo "Logs:     tail -f ${LOG_DIR}/${JOB_NAME}-${job_id}.out"
echo "Cancel:   scancel ${job_id}"
