#!/usr/bin/env bash
set -euo pipefail

# Request an interactive session on ict-h100 for RAGAS evaluation.
#
# ict-h100 (PETROBRAS / ICT): shared queue, max 4 GPUs per node, max 24 h wallclock.
# You must pass --account and --time. For GPU queues you must request GPUs explicitly
# (--gpus, --gpus-per-node, or --gres=gpu:N).
#
# Use this allocation to run vLLM OpenAI servers (judge / embeddings).
# RAGAS evaluation itself only needs HTTP access — run run_ragas_eval.sh
# from any node that can reach those server URLs.
#
# Examples:
#   # 1 node x 4 GPUs — vllm serve judge (TP=4)
#   bash jobs/scripts/santos_dumont/run_ragas_h100.sh
#
#   # 2 nodes — e.g. judge on node 1, embeddings on node 2
#   NODES=2 bash jobs/scripts/santos_dumont/run_ragas_h100.sh
#
#   # After servers are up:
#   RAGAS_JUDGE_BASE_URL=http://<node1>:8000/v1 \
#   RAGAS_EMBEDDING_BASE_URL=http://<node2>:8001/v1 \
#   bash jobs/scripts/santos_dumont/run_ragas_eval.sh \
#     datasets/generated/telco-dpr/generation-b128-vllm-offline-b128
#
# Override resources:
#   NODES=1 GPUS_PER_NODE=2 TIME=02:00:00 bash jobs/scripts/santos_dumont/run_ragas_h100.sh

NODES="${NODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-4}"
CPUS_PER_GPU="${CPUS_PER_GPU:-8}"
MEM_PER_GPU="${MEM_PER_GPU:-32G}"
TIME="${TIME:-04:00:00}"
ACCOUNT="${ACCOUNT:-smartassistant}"
PARTITION="${PARTITION:-ict-h100}"
JOB_NAME="${JOB_NAME:-ragas_eval}"

TOTAL_GPUS=$((NODES * GPUS_PER_NODE))

if (( GPUS_PER_NODE > 4 )); then
  echo "Error: H100 nodes on SDumont2nd expose at most 4 GPUs per node." >&2
  echo "Use GPUS_PER_NODE<=4 (got ${GPUS_PER_NODE})." >&2
  exit 1
fi

if (( NODES == 2 && GPUS_PER_NODE == 4 )); then
  echo "Allocating 2 nodes x ${GPUS_PER_NODE} GPUs (${TOTAL_GPUS} GPUs total)."
  echo "Typical layout: judge vLLM on node 1, embedding vLLM on node 2."
fi

salloc \
  --job-name="$JOB_NAME" \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --time="$TIME" \
  --nodes="$NODES" \
  --gpus-per-node="$GPUS_PER_NODE" \
  --cpus-per-gpu="$CPUS_PER_GPU" \
  --mem-per-gpu="$MEM_PER_GPU" \
  --chdir="$PWD" \
  bash -c '
    echo "SLURM_JOB_ID=$SLURM_JOB_ID"
    echo "SLURM_JOB_NODELIST=$SLURM_JOB_NODELIST"
    nodeset -e "$SLURM_JOB_NODELIST" 2>/dev/null || true
    exec srun --jobid="$SLURM_JOB_ID" --nodes=1 --ntasks=1 --gpus-per-node='"${GPUS_PER_NODE}"' --pty bash
  '
