#!/usr/bin/env bash
# Submit context embedding fine-tuning on ict-h100 (2 GPUs, non-interactive batch job).
#
# Usage (from repo root):
#   bash jobs/scripts/santos_dumont/run_finetune_embedding_context_h100.sh narrativeqa
#   bash jobs/scripts/santos_dumont/run_finetune_embedding_context_h100.sh qasper --epochs 10
#   TIME=24:00:00 JOB_NAME=my-ft-emb-ctx bash jobs/scripts/santos_dumont/run_finetune_embedding_context_h100.sh narrativeqa
#   TIME=12:00:00 bash jobs/scripts/santos_dumont/run_finetune_embedding_context_h100.sh bioasq-resplit --resume
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

DATASET="${1:?Usage: $0 <dataset> [extra finetune args...]}"
shift

JOB_NAME="${JOB_NAME:-ft-emb-ctx-${DATASET}}"
TIME="${TIME:-12:00:00}"
ACCOUNT="${ACCOUNT:-smartassistant}"
PARTITION="${PARTITION:-ict-h100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-131072M}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/slurm}"
PYTHON_RUNNER="${PYTHON_RUNNER:-uv run python}"
GPUS="${GPUS:-2}"

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
export CUDA_VISIBLE_DEVICES=0,1
echo SLURM_JOB_ID=\$SLURM_JOB_ID
echo Node: \$(hostname)
echo Dataset: ${DATASET}
echo Training mode: embedding-with-context
echo GPUs: \${CUDA_VISIBLE_DEVICES}
echo Started: \$(date -Is)
exec ${PYTHON_RUNNER} scripts/finetuning/embeddings/finetune_qwen3_embedding_context.py --dataset '${DATASET}'${EXTRA_CMD}")

echo "Submitted batch job ${JOB_ID}"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out"
echo "Cancel:   scancel ${JOB_ID}"
