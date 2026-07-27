#!/usr/bin/env bash
# BioASQ-resplit context MTEB with local long-context checkpoint (2048 tokens/doc).
#
# Evaluates checkpoint-550 from the bioasq-resplit 2k-token context fine-tuning run.
# Stage 2 uses k=1, 3, 5 with plain ## Context: blocks (no query/doc titles).
#
# Output: results/mteb/context_bioasq_2k_ckpt550/bioasq-resplit-ctx-b128-e10-2k-ckpt550/
# Revisions: ctx-lora-bioasq-resplit-k{k}-2k-ckpt550
#
# Usage (from repo root):
#   bash jobs/scripts/santos_dumont/run_mteb_context_retrieval_bioasq_2k_ckpt550_h100.sh
#
#   TIME=08:00:00 bash jobs/scripts/santos_dumont/run_mteb_context_retrieval_bioasq_2k_ckpt550_h100.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

DATASET="bioasq-resplit"
LORA_PATH="models/qwen3-embedding-4b-lora-ctx/bioasq-resplit-ctx-b128-e10-2k/checkpoint-550"
RUN_LABEL="bioasq-resplit-ctx-b128-e10-2k-ckpt550"
OUTPUT_DIR="${ROOT}/results/mteb/context_bioasq_2k_ckpt550/${RUN_LABEL}"

JOB_NAME="${JOB_NAME:-mteb-ctx-bioasq-2k-ckpt550}"
TIME="${TIME:-06:00:00}"
ACCOUNT="${ACCOUNT:-smartassistant}"
PARTITION="${PARTITION:-ict-h100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-4}"
MEM="${MEM:-65536M}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/slurm}"
PYTHON_RUNNER="${PYTHON_RUNNER:-uv run python}"

EXTRA_ARGS=()

while (("$#" > 0)); do
  case "$1" in
    --lora-path)
      LORA_PATH="${2:?Missing value after --lora-path}"
      shift 2
      ;;
    --run-label)
      RUN_LABEL="${2:?Missing value after --run-label}"
      OUTPUT_DIR="${ROOT}/results/mteb/context_bioasq_2k_ckpt550/${RUN_LABEL}"
      shift 2
      ;;
    *)
      EXTRA_ARGS+=("$1")
      shift
      ;;
  esac
done

mkdir -p "$LOG_DIR"

cmd=(
  --dataset "$DATASET"
  --lora-path "$LORA_PATH"
  --run-label "$RUN_LABEL"
  --output-dir "$OUTPUT_DIR"
  --context-k 1 3 5
  --stage1-top-k 10
  --max-doc-tokens 2048
  --max-seq-length 14336
  --model-revision-template "ctx-lora-{dataset}-k{k}-2k-ckpt550"
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
exec ${PYTHON_RUNNER} scripts/evaluation/mteb/run_mteb_context_retrieval.py${extra_cmd}")

echo "Submitted batch job ${job_id}"
echo "Dataset: ${DATASET}"
echo "LoRA: ${LORA_PATH}"
echo "Run label: ${RUN_LABEL}"
echo "Stage 2: k=1,3,5; max-doc-tokens=2048; max-seq-length=14336; no titles"
echo "Output: results/mteb/context_bioasq_2k_ckpt550/${RUN_LABEL}/"
echo "Revisions: ctx-lora-bioasq-resplit-k{1,3,5}-2k-ckpt550"
echo "Monitor:  squeue -j ${job_id}"
echo "Logs:     tail -f ${LOG_DIR}/${JOB_NAME}-${job_id}.out"
echo "Cancel:   scancel ${job_id}"
