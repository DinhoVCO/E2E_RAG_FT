#!/usr/bin/env bash
set -euo pipefail

# Pre-download tiktoken vocab files required by openai/gpt-oss-* on vLLM.
# Run on the login node (with network) before starting vLLM on compute nodes.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
OUT_DIR="${1:-${ROOT}/data/tiktoken_encodings}"

mkdir -p "$OUT_DIR"

BASE_URL="https://openaipublic.blob.core.windows.net/encodings"
for file in o200k_base.tiktoken cl100k_base.tiktoken; do
  dest="${OUT_DIR}/${file}"
  if [[ -f "$dest" ]]; then
    echo "OK ${dest} (already exists)"
    continue
  fi
  echo "Downloading ${file} -> ${dest}"
  curl -fL "${BASE_URL}/${file}" -o "$dest"
done

echo ""
echo "Done. Use this before gpt-oss vLLM:"
echo "  export TIKTOKEN_ENCODINGS_BASE=${OUT_DIR}"
