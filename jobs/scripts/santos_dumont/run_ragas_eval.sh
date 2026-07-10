#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT"

GENERATION_DIR="${1:?Usage: $0 <generation-dir> [extra args...]}"
shift

# RAGAS talks to vLLM OpenAI-compatible servers — no local GPU required here.
# Point these at your running vLLM instances (same node or remote).
export RAGAS_JUDGE_BASE_URL="${RAGAS_JUDGE_BASE_URL:-http://127.0.0.1:8000/v1}"
export RAGAS_EMBEDDING_BASE_URL="${RAGAS_EMBEDDING_BASE_URL:-http://127.0.0.1:8001/v1}"
export RAGAS_OPENAI_API_KEY="${RAGAS_OPENAI_API_KEY:-EMPTY}"
export RAGAS_JUDGE_MODEL="${RAGAS_JUDGE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-70B}"
export RAGAS_EMBEDDING_MODEL="${RAGAS_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
export RAGAS_MAX_WORKERS="${RAGAS_MAX_WORKERS:-4}"

uv run python scripts/evaluation/ragas/run_rag_ragas_evaluation.py \
  --generation-dir "$GENERATION_DIR" \
  "$@"
