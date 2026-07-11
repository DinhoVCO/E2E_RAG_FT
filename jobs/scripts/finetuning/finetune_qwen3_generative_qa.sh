#!/usr/bin/env bash
# Fine-tune Qwen3-8B with LoRA on QA pairs (no context) for one dataset.
#
# Usage (from repo root):
#   bash jobs/scripts/finetuning/finetune_qwen3_generative_qa.sh qasper
#   CUDA_VISIBLE_DEVICES=1 bash jobs/scripts/finetuning/finetune_qwen3_generative_qa.sh narrativeqa
#
set -euo pipefail

DATASET="${1:?Pass dataset name: bioasq-resplit, qasper, telco-dpr, or narrativeqa}"
shift

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python scripts/finetuning/generative/finetune_qwen3_generative_qa.py \
  --dataset "$DATASET" \
  "$@"
