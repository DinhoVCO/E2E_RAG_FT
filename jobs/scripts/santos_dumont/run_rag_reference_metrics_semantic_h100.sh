#!/usr/bin/env bash
# Submit semantic-similarity reference metrics on ict-h100 (1 GPU).
#
# Starts vLLM embedding server (Qwen3-Embedding-8B) on the same node, then
# evaluates response vs reference via RAGAS OpenAI-compatible embeddings API.
#
# Usage (from repo root):
#   bash jobs/scripts/santos_dumont/run_rag_reference_metrics_semantic_h100.sh --all
#
#   bash jobs/scripts/santos_dumont/run_rag_reference_metrics_semantic_h100.sh \
#     --dataset bioasq-resplit
#
# If you already have a remote embedding server running:
#   RAGAS_EMBEDDING_BASE_URL=http://<host>:8001/v1 \
#     bash jobs/scripts/santos_dumont/run_rag_reference_metrics_semantic_h100.sh --all
#
# To start only the embedding server (interactive):
#   bash jobs/scripts/santos_dumont/run_vllm_embedding_server_h100.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
SCRIPT="${ROOT}/scripts/evaluation/ragas/run_rag_reference_metrics_semantic.py"
EMBED_SCRIPT="${ROOT}/jobs/scripts/santos_dumont/run_vllm_embedding_server_h100.sh"

if (("$#" == 0)); then
  echo "Usage: $0 <--generation-dir PATH | --dataset NAME | --all> [extra args...]" >&2
  exit 1
fi

if [[ "$1" != "--generation-dir" && "$1" != "--dataset" && "$1" != "--all" ]]; then
  echo "First argument must be --generation-dir, --dataset, or --all." >&2
  exit 1
fi

MODE="$1"
TARGET="${2:-}"
if [[ "$MODE" != "--all" && -z "$TARGET" ]]; then
  echo "Missing value after ${MODE}." >&2
  exit 1
fi

case "$MODE" in
  --generation-dir) DEFAULT_JOB_NAME="rag-ref-sem" ;;
  --dataset) DEFAULT_JOB_NAME="rag-ref-sem-${TARGET}" ;;
  --all) DEFAULT_JOB_NAME="rag-ref-sem-all" ;;
esac

JOB_NAME="${JOB_NAME:-$DEFAULT_JOB_NAME}"
TIME="${TIME:-04:00:00}"
ACCOUNT="${ACCOUNT:-smartassistant}"
PARTITION="${PARTITION:-ict-h100}"
CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
MEM="${MEM:-65536M}"
LOG_DIR="${LOG_DIR:-${ROOT}/logs/slurm}"
PYTHON_RUNNER="${PYTHON_RUNNER:-uv run python}"
EMBEDDING_MODEL="${RAGAS_EMBEDDING_MODEL:-Qwen/Qwen3-Embedding-8B}"
EMBEDDING_PORT="${EMBEDDING_PORT:-8001}"
EMBEDDING_MAX_MODEL_LEN="${EMBEDDING_MAX_MODEL_LEN:-8192}"
EMBEDDING_MAX_NUM_SEQS="${EMBEDDING_MAX_NUM_SEQS:-256}"
SERVER_READY_TIMEOUT="${SERVER_READY_TIMEOUT:-600}"
USE_LOCAL_EMBED_SERVER="${USE_LOCAL_EMBED_SERVER:-1}"
EMBEDDING_BASE_URL="${RAGAS_EMBEDDING_BASE_URL:-}"

mkdir -p "$LOG_DIR"

EXTRA_CMD="$(printf ' %q' "$@")"

JOB_ID=$(sbatch --parsable \
  --job-name="$JOB_NAME" \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --time="$TIME" \
  --cpus-per-task="$CPUS_PER_TASK" \
  --mem="$MEM" \
  --gres=gpu:1 \
  --chdir="$ROOT" \
  --output="${LOG_DIR}/${JOB_NAME}-%j.out" \
  --error="${LOG_DIR}/${JOB_NAME}-%j.err" \
  --wrap="set -euo pipefail
cd '${ROOT}'
export CUDA_VISIBLE_DEVICES=0
export RAGAS_EMBEDDING_MODEL='${EMBEDDING_MODEL}'
export RAGAS_OPENAI_API_KEY='${RAGAS_OPENAI_API_KEY:-EMPTY}'
export EMBEDDING_MODEL='${EMBEDDING_MODEL}'
export EMBEDDING_PORT='${EMBEDDING_PORT}'
export EMBEDDING_MAX_MODEL_LEN='${EMBEDDING_MAX_MODEL_LEN}'
export EMBEDDING_MAX_NUM_SEQS='${EMBEDDING_MAX_NUM_SEQS}'
export SERVER_READY_TIMEOUT='${SERVER_READY_TIMEOUT}'
export USE_LOCAL_EMBED_SERVER='${USE_LOCAL_EMBED_SERVER}'
export EMBEDDING_BASE_URL='${EMBEDDING_BASE_URL}'

echo SLURM_JOB_ID=\$SLURM_JOB_ID
echo Node: \$(hostname)
echo Started: \$(date -Is)

_wait_for_server() {
  local base_url=\"\$1\"
  local label=\"\$2\"
  local elapsed=0
  local interval=10
  echo \"Waiting for \${label} at \${base_url} ...\"
  while (( elapsed < SERVER_READY_TIMEOUT )); do
    if curl -sf \"\${base_url}/models\" -H \"Authorization: Bearer EMPTY\" >/dev/null 2>&1; then
      echo \"\${label} is ready (\${elapsed}s).\"
      return 0
    fi
    sleep \"\$interval\"
    elapsed=\$((elapsed + interval))
  done
  echo \"Error: timed out after \${SERVER_READY_TIMEOUT}s waiting for \${label}.\" >&2
  return 1
}

EMBED_PID=\"\"
_cleanup() {
  if [[ -n \"\${EMBED_PID}\" ]]; then
    echo \"Stopping local embedding server (pid \${EMBED_PID})...\"
    kill \"\${EMBED_PID}\" 2>/dev/null || true
    wait \"\${EMBED_PID}\" 2>/dev/null || true
  fi
}
trap _cleanup EXIT INT TERM

if [[ -z \"\${EMBEDDING_BASE_URL}\" && \"\${USE_LOCAL_EMBED_SERVER}\" == \"1\" ]]; then
  EMBEDDING_BASE_URL=\"http://127.0.0.1:\${EMBEDDING_PORT}/v1\"
  echo \"Starting local vLLM embedding server at \${EMBEDDING_BASE_URL}\"
  bash '${EMBED_SCRIPT}' --start-local 127.0.0.1 &
  EMBED_PID=\$!
  _wait_for_server \"\${EMBEDDING_BASE_URL}\" \"Embedding server\"
elif [[ -z \"\${EMBEDDING_BASE_URL}\" ]]; then
  echo \"Set RAGAS_EMBEDDING_BASE_URL or USE_LOCAL_EMBED_SERVER=1\" >&2
  exit 1
fi

export RAGAS_EMBEDDING_BASE_URL=\"\${EMBEDDING_BASE_URL}\"
echo \"RAGAS_EMBEDDING_BASE_URL=\${RAGAS_EMBEDDING_BASE_URL}\"
echo \"RAGAS_EMBEDDING_MODEL=\${RAGAS_EMBEDDING_MODEL}\"

exec ${PYTHON_RUNNER} '${SCRIPT}' \
  --embedding-model '${EMBEDDING_MODEL}' \
  --embedding-base-url \"\${EMBEDDING_BASE_URL}\"${EXTRA_CMD}")

echo "Submitted batch job ${JOB_ID}"
echo "Monitor:  squeue -j ${JOB_ID}"
echo "Logs:     tail -f ${LOG_DIR}/${JOB_NAME}-${JOB_ID}.out"
echo "Cancel:   scancel ${JOB_ID}"
