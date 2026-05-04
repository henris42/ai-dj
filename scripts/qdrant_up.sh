#!/usr/bin/env bash
# Start the Qdrant container (podman). Idempotent: re-running starts a stopped container.
set -euo pipefail

NAME=ai-dj-qdrant
IMAGE=docker.io/qdrant/qdrant:v1.17.1
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORAGE="$ROOT/data/qdrant"

mkdir -p "$STORAGE"

if podman container exists "$NAME"; then
    podman start "$NAME" >/dev/null
else
    podman run -d \
        --name "$NAME" \
        --restart unless-stopped \
        -p 127.0.0.1:6333:6333 \
        -p 127.0.0.1:6334:6334 \
        -v "$STORAGE:/qdrant/storage:z" \
        -e QDRANT__LOG_LEVEL=INFO \
        "$IMAGE" >/dev/null
fi

echo "qdrant running at http://127.0.0.1:6333 (gRPC :6334)"
echo "  storage: $STORAGE"
