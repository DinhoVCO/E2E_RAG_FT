#!/usr/bin/env bash
set -euo pipefail

# Reserve 1 node on ict-h100 and start vLLM OpenAI embedding server only:
#   - 1 GPU, port 8001 by default
#   - Default model: Qwen/Qwen3-Embedding-8B
#
# Usage (interactive — keeps server running until Ctrl+D):
#   bash jobs/scripts/santos_dumont/run_vllm_embedding_server_h100.sh
#
# After the server is ready, from any node that reaches it:
#   export RAGAS_EMBEDDING_BASE_URL=http://<embed-node>:8001/v1
#   export RAGAS_EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B
#   bash jobs/scripts/santos_dumont/run_rag_reference_metrics_semantic_h100.sh --all
#
# Override defaults:
#   TIME=08:00:00 EMBEDDING_MODEL=Qwen/Qwen3-Embedding-8B \
#     bash jobs/scripts/santos_dumont/run_vllm_embedding_server_h100.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT="$ROOT/jobs/scripts/santos_dumont/run_vllm_embedding_server_h100.sh"

GPUS_PER_NODE=1
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
MEM_PER_GPU="${MEM_PER_GPU:-32G}"
TIME="${TIME:-04:00:00}"
ACCOUNT="${ACCOUNT:-smartassistant}"
PARTITION="${PARTITION:-ict-h100}"
JOB_NAME="${JOB_NAME:-vllm_embed}"

EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8001}"
EMBEDDING_GPUS="${EMBEDDING_GPUS:-1}"
EMBEDDING_MAX_MODEL_LEN="${EMBEDDING_MAX_MODEL_LEN:-8192}"
EMBEDDING_MAX_NUM_SEQS="${EMBEDDING_MAX_NUM_SEQS:-256}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"

export ROOT SCRIPT \
  EMBEDDING_MODEL EMBEDDING_PORT EMBEDDING_GPUS \
  EMBEDDING_MAX_MODEL_LEN EMBEDDING_MAX_NUM_SEQS SERVER_READY_TIMEOUT

_wait_for_server() {
  local base_url="$1"
  local label="$2"
  local elapsed=0
  local interval=10

  echo "Waiting for ${label} at ${base_url} ..."
  while (( elapsed < SERVER_READY_TIMEOUT )); do
    if curl -sf "${base_url}/models" -H "Authorization: Bearer EMPTY" >/dev/null 2>&1; then
      echo "${label} is ready (${elapsed}s)."
      return 0
    fi
    sleep "$interval"
    elapsed=$((elapsed + interval))
  done

  echo "Error: timed out after ${SERVER_READY_TIMEOUT}s waiting for ${label}." >&2
  echo "Check logs under ${ROOT}/logs/vllm/${SLURM_JOB_ID:-local}/" >&2
  return 1
}

_start_embedding_server() {
  local host="${1:-0.0.0.0}"
  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
  exec uv run vllm serve "${EMBEDDING_MODEL}" \
    --host "${host}" \
    --port "${EMBEDDING_PORT}" \
    --task embed \
    --tensor-parallel-size 1 \
    --max-model-len "${EMBEDDING_MAX_MODEL_LEN}" \
    --max-num-seqs "${EMBEDDING_MAX_NUM_SEQS}" \
    --served-model-name "${EMBEDDING_MODEL}"
}

_deploy_server() {
  mapfile -t NODES_LIST < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  local embed_node="${NODES_LIST[0]}"
  local log_dir="${ROOT}/logs/vllm/${SLURM_JOB_ID}"
  mkdir -p "$log_dir"

  echo "Embed node:  ${embed_node}  (${EMBEDDING_GPUS} GPU, port ${EMBEDDING_PORT})"
  echo "Model:       ${EMBEDDING_MODEL}"
  echo "Logs:        ${log_dir}/embedding.log"

  echo "Starting embedding server on ${embed_node}..."
  srun --jobid="${SLURM_JOB_ID}" \
    --nodes=1 --nodelist="${embed_node}" --ntasks=1 \
    --gpus-per-node="${EMBEDDING_GPUS}" \
    --output="${log_dir}/embedding.log" \
    --error="${log_dir}/embedding.log" \
    bash -lc "
      cd '${ROOT}'
      $(declare -f _start_embedding_server)
      _start_embedding_server 0.0.0.0
    " &
  EMBED_SRUN_PID=$!

  _wait_for_server "http://${embed_node}:${EMBEDDING_PORT}/v1" "Embedding server"

  echo ""
  echo "=== vLLM embedding server ready ==="
  echo "export RAGAS_EMBEDDING_BASE_URL=http://${embed_node}:${EMBEDDING_PORT}/v1"
  echo "export RAGAS_EMBEDDING_MODEL=${EMBEDDING_MODEL}"
  echo "export RAGAS_OPENAI_API_KEY=EMPTY"
  echo ""
  echo "Tail logs:"
  echo "  tail -f ${log_dir}/embedding.log"
  echo ""
  echo "Ctrl+D exits this shell and stops the server."

  _cleanup() {
    echo "Stopping embedding server..."
    kill "${EMBED_SRUN_PID}" 2>/dev/null || true
    wait "${EMBED_SRUN_PID}" 2>/dev/null || true
  }
  trap _cleanup EXIT INT TERM

  srun --jobid="${SLURM_JOB_ID}" \
    --nodes=1 --nodelist="${embed_node}" --ntasks=1 \
    --gpus-per-node="${EMBEDDING_GPUS}" \
    --pty bash
}

if [[ "${1:-}" == "--deploy" ]]; then
  cd "$ROOT"
  _deploy_server
  exit 0
fi

if [[ "${1:-}" == "--start-local" ]]; then
  cd "$ROOT"
  _start_embedding_server "${2:-127.0.0.1}"
  exit 0
fi

echo "Allocating 1 node x ${GPUS_PER_NODE} GPU for vLLM embedding server."

salloc \
  --job-name="$JOB_NAME" \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --time="$TIME" \
  --nodes=1 \
  --gpus-per-node="$GPUS_PER_NODE" \
  --cpus-per-gpu="$CPUS_PER_GPU" \
  --mem-per-gpu="$MEM_PER_GPU" \
  --chdir="$ROOT" \
  bash -c "exec bash '${SCRIPT}' --deploy"
