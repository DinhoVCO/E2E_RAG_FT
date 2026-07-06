#!/usr/bin/env bash
# Fine-tune Qwen3-Embedding-4B with LoRA on all four RAG datasets sequentially.
#
# Usage (from repo root):
#   bash jobs/scripts/finetuning/finetune_qwen3_embedding_all.sh
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
  echo "=== Fine-tuning on ${dataset} ==="
  python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
    --dataset "$dataset" \
    --batch-size 128 \
    --epochs 1 \
    --eval-steps 500 \
    --save-steps 500
done

echo "All fine-tuning runs completed."
