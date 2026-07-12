#!/usr/bin/env bash
# Generate RAG answers for all four datasets using base-model retrieved_docs.
#
# Usage (from repo root):
#   bash scripts/generation/run_all_rag_generation.sh
#
# Optional environment overrides:
#   CUDA_VISIBLE_DEVICES=0
#   MODEL=Qwen/Qwen3-8B
#   TOP_K=5
#   RETRIEVAL_RUN_LABEL=vllm-offline-b128
#   GENERATION_RUN_LABEL=generation-b128
#
# Note: this batch script runs the base generation model only. For LoRA adapters,
# call run_rag_generation.py per dataset with --lora-path (generation adapter).
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

MODEL="${MODEL:-Qwen/Qwen3-8B}"
TOP_K="${TOP_K:-5}"
RETRIEVAL_RUN_LABEL="${RETRIEVAL_RUN_LABEL:-vllm-offline-b128}"
GENERATION_RUN_LABEL="${GENERATION_RUN_LABEL:-generation-b128}"

GENERATE_SCRIPT=(python scripts/generation/run_rag_generation.py)

DATASETS=(
  telco-dpr
  qasper
  narrativeqa
  bioasq-resplit
)

for dataset in "${DATASETS[@]}"; do
  echo "=== Generating answers for ${dataset} ==="
  "${GENERATE_SCRIPT[@]}" \
    --dataset "$dataset" \
    --model "$MODEL" \
    --top-k "$TOP_K" \
    --retrieval-run-label "$RETRIEVAL_RUN_LABEL" \
    --run-label "${GENERATION_RUN_LABEL}-${RETRIEVAL_RUN_LABEL}"
done

echo "All RAG generation runs completed."
