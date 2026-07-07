#!/usr/bin/env bash
# Fine-tune Qwen3-Embedding-4B with LoRA on one RAG dataset.
#
# Usage (from repo root):
#   bash jobs/scripts/finetuning/finetune_qwen3_embedding.sh qasper
#   CUDA_VISIBLE_DEVICES=0,1 bash jobs/scripts/finetuning/finetune_qwen3_embedding.sh bioasq-resplit
#
set -euo pipefail

DATASET="${1:?Pass dataset name: bioasq-resplit, qasper, telco-dpr, or narrativeqa}"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python scripts/finetuning/embeddings/finetune_qwen3_embedding.py \
  --dataset "$DATASET" \
  "$@"
