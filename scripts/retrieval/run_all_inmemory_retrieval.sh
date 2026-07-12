#!/usr/bin/env bash
# Run in-memory top-k retrieval for all four RAG datasets with the base model
# and dataset-specific LoRA adapters (offline vLLM).
#
# Usage (from repo root):
#   bash scripts/retrieval/run_all_inmemory_retrieval.sh
#
# Optional environment overrides:
#   CUDA_VISIBLE_DEVICES=0
#   BASE_MODEL=Qwen/Qwen3-Embedding-4B
#   TOP_K=10
#   BATCH_SIZE=128
#   BASE_RUN_LABEL=vllm-offline-b128
#   LORA_RUN_LABEL_PREFIX=vllm-lora
#   SKIP_BASE=1          # skip base-model runs
#   SKIP_LORA=1          # skip LoRA runs
#
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-Embedding-4B}"
TOP_K="${TOP_K:-10}"
BATCH_SIZE="${BATCH_SIZE:-128}"
BASE_RUN_LABEL="${BASE_RUN_LABEL:-vllm-offline-b128}"
LORA_RUN_LABEL_PREFIX="${LORA_RUN_LABEL_PREFIX:-vllm-lora}"

RETRIEVE_SCRIPT=(python scripts/retrieval/retrieve_rag_top_k_inmemory.py)

DATASETS=(
  telco-dpr
  qasper
  narrativeqa
  bioasq-resplit
)

LORA_REPOS=(
  DinoStackAI/Qwen3-Emb-4b-lora-telco-dpr
  DinoStackAI/Qwen3-Emb-4b-lora-qasper
  DinoStackAI/Qwen3-Emb-4b-lora-narrativeqa
  DinoStackAI/Qwen3-Emb-4b-lora-bioasq-resplit
)

run_base() {
  local dataset="$1"
  echo "=== [base] ${dataset} (run-label: ${BASE_RUN_LABEL}) ==="
  "${RETRIEVE_SCRIPT[@]}" \
    --dataset "$dataset" \
    --mode offline \
    --model "$BASE_MODEL" \
    --top-k "$TOP_K" \
    --batch-size "$BATCH_SIZE" \
    --run-label "$BASE_RUN_LABEL"
}

run_lora() {
  local dataset="$1"
  local lora_path="$2"
  local run_label="${LORA_RUN_LABEL_PREFIX}-${dataset}-b128"
  echo "=== [lora] ${dataset} (run-label: ${run_label}) ==="
  "${RETRIEVE_SCRIPT[@]}" \
    --dataset "$dataset" \
    --mode offline \
    --model "$BASE_MODEL" \
    --lora-path "$lora_path" \
    --top-k "$TOP_K" \
    --batch-size "$BATCH_SIZE" \
    --run-label "$run_label"
}

if [[ "${SKIP_BASE:-0}" != "1" ]]; then
  echo ">>> Base model retrieval (${BASE_RUN_LABEL})"
  for dataset in "${DATASETS[@]}"; do
    run_base "$dataset"
  done
fi

if [[ "${SKIP_LORA:-0}" != "1" ]]; then
  echo ">>> LoRA adapter retrieval (${LORA_RUN_LABEL_PREFIX}-<dataset>-b128)"
  for i in "${!DATASETS[@]}"; do
    run_lora "${DATASETS[$i]}" "${LORA_REPOS[$i]}"
  done
fi

echo "All in-memory retrieval runs completed."
