#!/usr/bin/env python3
"""Single entry point for AI DJ.

Usage:
    python3 run.py                       # set up if needed, then launch the GUI
    python3 run.py --library PATH        # also index the library at PATH (first time)

The script is idempotent: each step checks state and skips when nothing's
to do. The first run on a new machine analyzes the host, installs the right
torch wheel (CPU / CUDA / ROCm / MPS), downloads the genre-tagger models,
brings up Qdrant, optionally indexes a music library, then launches the
GUI. Subsequent runs verify deps + Qdrant and go straight to the GUI.

Stdlib only — has to work before `uv sync` runs.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS_DIR = ROOT / "data" / "models"
QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "tracks"

MODEL_FILES = {
    "discogs-effnet-bs64-1.pb":
        "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb",
    "genre_discogs400-discogs-effnet-1.pb":
        "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb",
    "genre_discogs400-discogs-effnet-1.json":
        "https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json",
}


def step(msg: str) -> None:
    print(f"\n==> {msg}")


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print(f"   $ {' '.join(cmd)}")
    return subprocess.run(cmd, check=True, **kw)


def host_has_rocm() -> bool:
    if sys.platform != "linux":
        return False
    return Path("/dev/kfd").exists() and shutil.which("rocminfo") is not None


def host_has_nvidia() -> bool:
    return shutil.which("nvidia-smi") is not None


# ---- 1. Python deps ----------------------------------------------------------

def venv_exists() -> bool:
    return (ROOT / ".venv").exists()


def torch_is_rocm() -> bool:
    """True if the venv's torch was built against ROCm."""
    try:
        out = subprocess.run(
            ["uv", "run", "python", "-c",
             "import torch, sys; sys.stdout.write(torch.__version__)"],
            cwd=ROOT, capture_output=True, text=True, check=True,
        ).stdout
        return "+rocm" in out
    except Exception:  # noqa: BLE001
        return False


def ensure_python_deps() -> None:
    if shutil.which("uv") is None:
        print("uv is not installed. Install it first:")
        print("  curl -LsSf https://astral.sh/uv/install.sh | sh")
        print("  (or see https://github.com/astral-sh/uv)")
        sys.exit(2)

    if not venv_exists():
        step("First run — installing Python deps with uv sync")
        run(["uv", "sync"])
    else:
        # Lightweight idempotent re-sync to pick up pyproject changes.
        step("Verifying Python deps")
        run(["uv", "sync"])

    # ROCm needs a torch wheel from a different index. Detect and override only
    # if it isn't already in place.
    if host_has_rocm() and not torch_is_rocm():
        step("AMD ROCm GPU detected — installing ROCm-built torch")
        run([
            "uv", "pip", "install",
            "--index-url", "https://download.pytorch.org/whl/rocm6.4",
            "torch==2.9.1+rocm6.4",
            "torchaudio==2.9.1+rocm6.4",
            "pytorch-triton-rocm",
        ])
    elif host_has_nvidia():
        print("   NVIDIA GPU detected — default torch wheel includes CUDA.")
    elif sys.platform == "darwin":
        print("   macOS — torch's MPS backend is in the default wheel.")


# ---- 2. Genre-tagger models --------------------------------------------------

def ensure_models() -> None:
    step("Verifying genre-tagger models")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in MODEL_FILES.items():
        dst = MODELS_DIR / name
        if dst.exists() and dst.stat().st_size > 0:
            print(f"   ok  {name}")
            continue
        print(f"   download {name}")
        try:
            urllib.request.urlretrieve(url, dst)
        except urllib.error.URLError as e:
            print(f"   FAILED to download {url}: {e}")
            sys.exit(3)


# ---- 3. Qdrant ---------------------------------------------------------------

def qdrant_up() -> bool:
    try:
        with urllib.request.urlopen(QDRANT_URL + "/", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def collection_count() -> int:
    try:
        with urllib.request.urlopen(QDRANT_URL + f"/collections/{COLLECTION}", timeout=2) as r:
            data = json.loads(r.read())
        return int(data.get("result", {}).get("points_count", 0))
    except Exception:  # noqa: BLE001
        return 0


def ensure_qdrant() -> None:
    step("Verifying Qdrant")
    if qdrant_up():
        print("   already running")
        return
    qdrant_sh = ROOT / "scripts" / "qdrant_up.sh"
    if not qdrant_sh.exists():
        print("   ERR scripts/qdrant_up.sh missing")
        sys.exit(4)
    run(["bash", str(qdrant_sh)])
    deadline = time.time() + 20
    while time.time() < deadline:
        if qdrant_up():
            print("   Qdrant is up")
            return
        time.sleep(0.5)
    print("   ERR Qdrant didn't come up within 20 s")
    sys.exit(5)


# ---- 4. Library indexing -----------------------------------------------------

def maybe_index_library(library: str | None) -> None:
    if library is None:
        if collection_count() == 0:
            print("\n!! Qdrant collection is empty. Re-run with --library /path/to/music")
            print("   to index your library. (Or drop tracks in via the GUI later.)")
        return
    lib_path = Path(library).expanduser().resolve()
    if not lib_path.exists():
        print(f"   ERR --library path does not exist: {lib_path}")
        sys.exit(6)
    step(f"Indexing library: {lib_path}")
    print("   (this is the long part — re-runnable, skips already-done tracks)")
    run(["uv", "run", "python", "scripts/build_index.py", "--library", str(lib_path)])
    step("Tagging genres with Discogs-Effnet")
    run(["uv", "run", "python", "scripts/tag_library_discogs.py"])


# ---- 5. Launch ---------------------------------------------------------------

def launch_gui() -> int:
    step("Launching AI DJ")
    return subprocess.call(["uv", "run", "python", "-m", "ai_dj.gui.app"])


def main() -> int:
    ap = argparse.ArgumentParser(description="AI DJ — set up if needed, then launch.")
    ap.add_argument("--library", default=None,
                    help="path to a music library to scan + embed (only needed once)")
    ap.add_argument("--no-launch", action="store_true",
                    help="set everything up but don't launch the GUI")
    args = ap.parse_args()

    os.chdir(ROOT)

    ensure_python_deps()
    ensure_models()
    ensure_qdrant()
    maybe_index_library(args.library)
    if args.no_launch:
        return 0
    return launch_gui()


if __name__ == "__main__":
    sys.exit(main())
