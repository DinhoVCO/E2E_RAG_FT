#!/usr/bin/env bash
# Submit bioasq-resplit context embedding fine-tuning with 2048-token docs on ict-h100.
#
# Uses 4 GPUs, batch_size=128 per device, max_doc_tokens=2048, max_seq_length=14336.
# Config: scripts/finetuning/embeddings/configs/context/bioasq-resplit-2k.yaml
# Output: models/qwen3-embedding-4b-lora-ctx/bioasq-resplit-ctx-b128-e10-2k/
#
# Usage (from repo root):
#   bash jobs/scripts/santos_dumont/run_finetune_embedding_context_bioasq_2k_h100.sh
#
#   bash jobs/scripts/santos_dumont/run_finetune_embedding_context_bioasq_2k_h100.sh --resume
#
#   TIME=24:00:00 JOB_NAME=ft-emb-ctx-bioasq-2k \
#     bash jobs/scripts/santos_dumont/run_finetune_embedding_context_bioasq_2k_h100.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

CONFIG="${ROOT}/scripts/finetuning/embeddings/configs/context/bioasq-resplit-2k.yaml"
DATASET="bioasq-resplit"

JOB_NAME="${JOB_NAME:-ft-emb-ctx-${DATASET}-2k}"
TIME="${TIME:-24:00:00}"
ACCOUNT="${ACCOUNT:-smartassistant}"
PARTITION="${PARTITION:-ict-h100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEM="${MEM:-262144M}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/slurm}"
PYTHON_RUNNER="${PYTHON_RUNNER:-uv run python}"
GPUS="${GPUS:-4}"

mkdir -p "$LOG_DIR"

EXTRA_CMD=""
if (("$#" > 0)); then
  EXTRA_CMD="$(printf ' %q' "$@")"
fi

JOB_ID=$(sbatch --parsable \
  --job-name="$JOB_NAME" \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --time="$TIME" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM" \
  --gres="gpu:${GPUS}" \
  --chdir="$ROOT" \
  --output="${LOG_DIR}/${JOB_NAME}-%j.out" \
  --error="${LOG_DIR}/${JOB_NAME}-%j.err" \
  --wrap="set -euo pipefail
cd '${ROOT}'
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo SLURM_JOB_ID=\$SLURM_JOB_ID
echo Node: \$(hostname)
echo Dataset: ${DATASET}
echo Config: ${CONFIG}
echo 'Training mode: embedding-with-context (2048 doc tokens)'
echo GPUs: \${CUDA_VISIBLE_DEVICES}
echo Started: \$(date -Is)
exec ${PYTHON_RUNNER} scripts/finetuning/embeddings/finetune_qwen3_embedding_context.py --config '${CONFIG}' --dataset '${DATASET}'${EXTRA_CMD}")

echo "Submitted batch job ${JOB_ID}"
echo "Config: ${CONFIG}"
echo "Output: models/qwen3-embedding-4b-lora-ctx/bioasq-resplit-ctx-b128-e10-2k/"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out"
echo "Cancel:   scancel ${JOB_ID}"
