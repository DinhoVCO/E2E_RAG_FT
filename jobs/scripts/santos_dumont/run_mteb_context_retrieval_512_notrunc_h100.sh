#!/usr/bin/env bash
# Submit context-aware retrieval evaluation without total stage-2 truncation.
#
# Each retrieved document is truncated independently to 512 tokens. The complete
# stage-2 prompt is not shortened, so k=1,3,5,7,10 keeps every available document.
# Results are written under results/mteb/context_512_notrunc/.
#
# Usage:
#   bash jobs/scripts/santos_dumont/run_mteb_context_retrieval_512_notrunc_h100.sh \
#     --dataset bioasq-resplit
#
#   bash jobs/scripts/santos_dumont/run_mteb_context_retrieval_512_notrunc_h100.sh \
#     --submit-each --all

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BASE_RUNNER="${ROOT}/jobs/scripts/santos_dumont/run_mteb_context_retrieval_h100.sh"

if (("$#" == 0)); then
  echo "Usage: $0 <--dataset NAME | --all> [submission options]" >&2
  exit 1
fi

exec bash "$BASE_RUNNER" \
  "$@" \
  --context-k 1 3 5 7 10 \
  --stage1-top-k 10 \
  --max-doc-tokens 512 \
  --no-truncate-stage2-total \
  --model-revision-template 'ctx-lora-{dataset}-k{k}-512-notrunc'
