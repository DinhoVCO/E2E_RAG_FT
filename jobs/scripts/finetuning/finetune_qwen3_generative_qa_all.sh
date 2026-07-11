#!/usr/bin/env bash
# Fine-tune Qwen3-8B with LoRA on QA pairs for all four datasets sequentially.
#
# Usage (from repo root):
#   bash jobs/scripts/finetuning/finetune_qwen3_generative_qa_all.sh
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
  echo "=== QA fine-tuning on ${dataset} ==="
  python scripts/finetuning/generative/finetune_qwen3_generative_qa.py \
    --dataset "$dataset"
done

echo "All QA fine-tuning runs completed."
