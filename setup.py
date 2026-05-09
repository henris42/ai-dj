#!/usr/bin/env python3
"""One-shot environment setup for AI DJ.

Most platforms (Linux+NVIDIA, Windows+NVIDIA, macOS) work fine with whatever
`uv sync` installs — pip's torch wheel resolver does the right thing. The
only host that needs special handling is AMD ROCm on Linux, whose torch
wheels live on a separate index. This script runs `uv sync` and, if it
detects a ROCm-capable GPU on the host, installs the ROCm-built torch on
top.

Usage:
    python3 setup.py

Run it once after `git clone`. Re-runnable; it's idempotent.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


def has_rocm_gpu() -> bool:
    """True iff the host has an AMD GPU exposed to userspace via ROCm."""
    if sys.platform != "linux":
        return False
    # `/dev/kfd` is present whenever the AMD compute kernel driver loaded an
    # exposed GPU (native Linux, or WSL2 once the libhsa shim is in place).
    if not Path("/dev/kfd").exists():
        return False
    return shutil.which("rocminfo") is not None


def has_nvidia_gpu() -> bool:
    return shutil.which("nvidia-smi") is not None


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> int:
    if shutil.which("uv") is None:
        print("uv not found. Install it first:\n"
              "  curl -LsSf https://astral.sh/uv/install.sh | sh\n"
              "  (or see https://github.com/astral-sh/uv)")
        return 2

    print("==> uv sync")
    run(["uv", "sync"])

    if has_rocm_gpu():
        print("\n==> AMD ROCm GPU detected — installing ROCm-built torch on top")
        run([
            "uv", "pip", "install",
            "--index-url", "https://download.pytorch.org/whl/rocm6.4",
            "torch==2.9.1+rocm6.4",
            "torchaudio==2.9.1+rocm6.4",
            "pytorch-triton-rocm",
        ])
        print("\nROCm torch installed. If you're on WSL and the embed step "
              "hangs, see the ROCm-on-WSL note in README.md.")
    elif has_nvidia_gpu():
        print("\n==> NVIDIA GPU detected — default torch wheel already "
              "includes CUDA support, nothing extra to install.")
    elif sys.platform == "darwin":
        print("\n==> macOS — torch's MPS backend is in the default wheel.")
    else:
        print("\n==> No discrete GPU detected — using CPU torch. "
              "Indexing will work but will be slow on a large library.")

    print("\nReady. Next:")
    print("  ./scripts/qdrant_up.sh                  # start Qdrant")
    print("  uv run python scripts/build_index.py --library /path/to/your/music")
    print("  uv run python scripts/tag_library_discogs.py")
    print("  uv run python -m ai_dj.gui.app          # run the app")
    return 0


if __name__ == "__main__":
    sys.exit(main())
