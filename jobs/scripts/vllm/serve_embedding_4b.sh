#!/usr/bin/env bash
# Servidor vLLM para Qwen3-Embedding-4B (texto, API /v1/embeddings).
#
# Uso (desde la raíz del repo):
#   bash jobs/scripts/vllm/serve_embedding_4b.sh
#
# Opcional — sobrescribir modelo, host o puerto:
#   VLLM_MODEL=Qwen/Qwen3-Embedding-8B VLLM_PORT=8002 bash jobs/scripts/vllm/serve_embedding_4b.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
VENV="${ROOT}/.venv"

if [[ ! -x "${VENV}/bin/vllm" ]]; then
  echo "vllm no encontrado en ${VENV}/bin/vllm" >&2
  echo "Instala dependencias con: cd ${ROOT} && uv sync" >&2
  exit 1
fi

MODEL="${VLLM_MODEL:-Qwen/Qwen3-Embedding-4B}"
HOST="${VLLM_HOST:-127.0.0.1}"
PORT="${VLLM_PORT:-8000}"

exec "${VENV}/bin/vllm" serve "${MODEL}" \
  --runner pooling \
  --host "${HOST}" \
  --port "${PORT}"
