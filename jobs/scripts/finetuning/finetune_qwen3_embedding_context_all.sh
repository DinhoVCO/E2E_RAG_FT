#!/usr/bin/env bash
# Fine-tune Qwen3-Embedding-4B with LoRA (context-augmented) on all four RAG datasets.
#
# Usage (from repo root):
#   bash jobs/scripts/finetuning/finetune_qwen3_embedding_context_all.sh
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
  echo "=== Context fine-tuning on ${dataset} ==="
  python scripts/finetuning/embeddings/finetune_qwen3_embedding_context.py \
    --dataset "$dataset"
done

echo "All context fine-tuning runs completed."
