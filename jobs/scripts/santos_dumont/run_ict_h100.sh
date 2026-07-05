#!/usr/bin/env bash
set -euo pipefail

salloc \
  --job-name="${JOB_NAME:-my_session}" \
  --account=smartassistant \
  --partition=ict-h100 \
  --time=02:00:00 \
  --cpus-per-task=4 \
  --mem=8192M \
  --gres=gpu:3 \
  --chdir="$PWD" \
  bash -c 'echo "SLURM_JOB_ID=$SLURM_JOB_ID"; exec srun --jobid="$SLURM_JOB_ID" --pty bash'