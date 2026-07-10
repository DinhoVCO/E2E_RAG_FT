#!/usr/bin/env bash
set -euo pipefail

# Reserve 2 nodes on ict-h100 and start vLLM OpenAI servers:
#   - Node 1: judge LLM (4 GPUs, TP=4) on port 8000
#   - Node 2: embedding model (1 GPU) on port 8001
#
# Usage:
#   bash jobs/scripts/santos_dumont/run_vllm_servers_h100.sh
#
# After servers are ready, run RAGAS from any node that reaches those URLs:
#   RAGAS_JUDGE_BASE_URL=http://<judge-node>:8000/v1 \
#   RAGAS_EMBEDDING_BASE_URL=http://<embed-node>:8001/v1 \
#   bash jobs/scripts/santos_dumont/run_ragas_eval.sh <generation-dir>
#
# Override defaults:
#   TIME=08:00:00 JUDGE_MODEL=... bash jobs/scripts/santos_dumont/run_vllm_servers_h100.sh

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT="$ROOT/jobs/scripts/santos_dumont/run_vllm_servers_h100.sh"

NODES=2
GPUS_PER_NODE=4
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
MEM_PER_GPU="${MEM_PER_GPU:-32G}"
TIME="${TIME:-04:00:00}"
ACCOUNT="${ACCOUNT:-smartassistant}"
PARTITION="${PARTITION:-ict-h100}"
JOB_NAME="${JOB_NAME:-vllm_servers}"

JUDGE_MODEL="${JUDGE_MODEL:-deepseek-ai/DeepSeek-R1-Distill-Llama-70B}"
EMBEDDING_MODEL="${EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
JUDGE_PORT="${JUDGE_PORT:-8000}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8001}"
JUDGE_TP="${JUDGE_TP:-4}"
JUDGE_MAX_MODEL_LEN="${JUDGE_MAX_MODEL_LEN:-8192}"
EMBEDDING_GPUS="${EMBEDDING_GPUS:-1}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"

export ROOT SCRIPT \
  JUDGE_MODEL EMBEDDING_MODEL JUDGE_PORT EMBEDDING_PORT \
  JUDGE_TP JUDGE_MAX_MODEL_LEN EMBEDDING_GPUS SERVER_READY_TIMEOUT

_deploy_servers() {
  mapfile -t NODES_LIST < <(scontrol show hostnames "${SLURM_JOB_NODELIST}")
  if ((${#NODES_LIST[@]} < 2)); then
    echo "Error: need 2 nodes in SLURM_JOB_NODELIST (got ${#NODES_LIST[@]})." >&2
    exit 1
  fi

  local judge_node="${NODES_LIST[0]}"
  local embed_node="${NODES_LIST[1]}"
  local log_dir="${ROOT}/logs/vllm/${SLURM_JOB_ID}"
  mkdir -p "$log_dir"

  echo "Judge node:  ${judge_node}  (${JUDGE_TP} GPUs, port ${JUDGE_PORT})"
  echo "Embed node:  ${embed_node}  (${EMBEDDING_GPUS} GPU, port ${EMBEDDING_PORT})"
  echo "Logs:        ${log_dir}/"

  export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"

  echo "Starting embedding server on ${embed_node}..."
  srun --jobid="${SLURM_JOB_ID}" \
    --nodes=1 --nodelist="${embed_node}" --ntasks=1 \
    --gpus-per-node="${EMBEDDING_GPUS}" \
    --output="${log_dir}/embedding.log" \
    --error="${log_dir}/embedding.log" \
    bash -lc "
      cd '${ROOT}'
      exec uv run vllm serve '${EMBEDDING_MODEL}' \
        --host 0.0.0.0 --port '${EMBEDDING_PORT}' \
        --task embed \
        --tensor-parallel-size 1 \
        --served-model-name '${EMBEDDING_MODEL}'
    " &
  EMBED_SRUN_PID=$!

  echo "Starting judge server on ${judge_node}..."
  srun --jobid="${SLURM_JOB_ID}" \
    --nodes=1 --nodelist="${judge_node}" --ntasks=1 \
    --gpus-per-node="${JUDGE_TP}" \
    --output="${log_dir}/judge.log" \
    --error="${log_dir}/judge.log" \
    bash -lc "
      cd '${ROOT}'
      exec uv run vllm serve '${JUDGE_MODEL}' \
        --host 0.0.0.0 --port '${JUDGE_PORT}' \
        --tensor-parallel-size '${JUDGE_TP}' \
        --max-model-len '${JUDGE_MAX_MODEL_LEN}' \
        --served-model-name '${JUDGE_MODEL}'
    " &
  JUDGE_SRUN_PID=$!

  _wait_for_server "http://${embed_node}:${EMBEDDING_PORT}/v1" "Embedding server" &
  local embed_wait_pid=$!
  _wait_for_server "http://${judge_node}:${JUDGE_PORT}/v1" "Judge server" &
  local judge_wait_pid=$!

  wait "$embed_wait_pid"
  wait "$judge_wait_pid"

  echo ""
  echo "=== vLLM servers ready ==="
  echo "export RAGAS_JUDGE_BASE_URL=http://${judge_node}:${JUDGE_PORT}/v1"
  echo "export RAGAS_EMBEDDING_BASE_URL=http://${embed_node}:${EMBEDDING_PORT}/v1"
  echo "export RAGAS_OPENAI_API_KEY=EMPTY"
  echo ""
  echo "Tail logs:"
  echo "  tail -f ${log_dir}/judge.log"
  echo "  tail -f ${log_dir}/embedding.log"
  echo ""
  echo "Ctrl+D exits this shell and stops both servers."

  _cleanup() {
    echo "Stopping vLLM servers..."
    kill "${JUDGE_SRUN_PID}" "${EMBED_SRUN_PID}" 2>/dev/null || true
    wait "${JUDGE_SRUN_PID}" "${EMBED_SRUN_PID}" 2>/dev/null || true
  }
  trap _cleanup EXIT INT TERM

  srun --jobid="${SLURM_JOB_ID}" \
    --nodes=1 --nodelist="${judge_node}" --ntasks=1 \
    --gpus-per-node="${JUDGE_TP}" \
    --pty bash
}

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
  echo "Check logs under ${ROOT}/logs/vllm/${SLURM_JOB_ID}/" >&2
  return 1
}

if [[ "${1:-}" == "--deploy" ]]; then
  cd "$ROOT"
  _deploy_servers
  exit 0
fi

echo "Allocating ${NODES} nodes x ${GPUS_PER_NODE} GPUs for vLLM judge + embedding servers."

salloc \
  --job-name="$JOB_NAME" \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --time="$TIME" \
  --nodes="$NODES" \
  --gpus-per-node="$GPUS_PER_NODE" \
  --cpus-per-gpu="$CPUS_PER_GPU" \
  --mem-per-gpu="$MEM_PER_GPU" \
  --chdir="$ROOT" \
  bash -c "exec bash '${SCRIPT}' --deploy"
