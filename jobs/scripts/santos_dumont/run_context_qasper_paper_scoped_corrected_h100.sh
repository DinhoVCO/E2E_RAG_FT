#!/usr/bin/env bash
# Evaluate context-aware embeddings on QASPER with paper-scoped retrieval
# in both stages, saving corrected results under a separate output root.
#
# Each stage-2 context document is truncated independently to 512 tokens.
# The total prompt is not truncated, so every requested context document is
# retained for k=1,3,5,7,10.
#
# Usage:
#   bash jobs/scripts/santos_dumont/run_context_qasper_paper_scoped_corrected_h100.sh

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_RUNNER="${ROOT}/jobs/scripts/santos_dumont/run_mteb_context_retrieval_h100.sh"
OUTPUT_DIR="${ROOT}/results/mteb/context_qasper_paper_scoped_corrected/qasper/Qwen3-Emb-4b-lora-ctx-qasper"

exec env JOB_NAME="${JOB_NAME:-ctx-qasper-paper-scoped}" \
  TIME="${TIME:-06:00:00}" \
  bash "$BASE_RUNNER" \
    --dataset qasper \
    --paper-scoped \
    --context-k 1 3 5 7 10 \
    --stage1-top-k 10 \
    --max-doc-tokens 512 \
    --no-truncate-stage2-total \
    --model-revision-template 'ctx-lora-qasper-k{k}-512-notrunc-paper-scoped-corrected' \
    --output-dir "$OUTPUT_DIR"
