#!/usr/bin/env bash
set -euo pipefail

# Run from inside an interactive session (run_ict_h100.sh)
QDRANT_STORAGE="$(pwd)/qdrant_storage"
QDRANT_IMAGE="$(pwd)/images/qdrant.sif"

mkdir -p "$QDRANT_STORAGE"

singularity run \
  --pwd /qdrant \
  --writable-tmpfs \
  --bind "$QDRANT_STORAGE:/qdrant/storage" \
  "$QDRANT_IMAGE" /qdrant/qdrant &

echo "Qdrant started in background (PID: $!)"
echo "API REST: http://localhost:6333"
echo "gRPC:     localhost:6334"

#bash jobs/utils/run_qdrant.sh