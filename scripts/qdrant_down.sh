#!/usr/bin/env bash
set -euo pipefail
podman stop ai-dj-qdrant >/dev/null 2>&1 || true
echo "qdrant stopped"
