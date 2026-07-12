#!/usr/bin/env bash
# Fine-tune Qwen3-8B with LoRA on all four RAG generative datasets sequentially.
#
# Usage (from repo root):
#   bash jobs/scripts/finetuning/finetune_qwen3_generative_all.sh
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASETS=(
  bioasq-resplit
  qasper
  telco-dpr
  narrativeqa
)

for dataset in "${DATASETS[@]}"; do
  echo "=== Fine-tuning generative model on ${dataset} ==="
  python scripts/finetuning/generative/finetune_qwen3_generative.py \
    --dataset "$dataset"
done

echo "All generative fine-tuning runs completed."
