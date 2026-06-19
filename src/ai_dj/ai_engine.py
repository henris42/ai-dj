"""First-run installation of the AI engine.

The base AI DJ install (PySide6 + libmpv + qdrant-client + the app) is
small and ships in every package. The heavy embedding stack — torch,
transformers, librosa — is **not** bundled by default (keeps the
installer light and avoids 1–3 GB of GPU-specific wheels in the box).
This module handles the gap: detect what's missing, install it via
`uv pip` in a subprocess, stream output so the GUI can show progress.

This is what makes the customer's first Teach actually run on a fresh
install. Without it, Teach surfaces "ML stack not installed" — which is
honest but not friendly.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable, Iterable

REQUIRED = ("torch", "transformers", "librosa")


def is_available() -> bool:
    """All three required imports succeed?"""
    for mod in REQUIRED:
        try:
            __import__(mod)
        except ImportError:
            return False
    return True


def missing() -> list[str]:
    out = []
    for mod in REQUIRED:
        try:
            __import__(mod)
        except ImportError:
            out.append(mod)
    return out


def _find_uv() -> str | None:
    """Locate uv.exe / uv on the host. We invoke it as a subprocess rather
    than a library so the GUI's own venv stays untouched until pip lands
    the wheels."""
    if sys.platform == "win32":
        for cand in (Path(__file__).resolve().parents[2] / "uv.exe",
                     Path.home() / ".local" / "bin" / "uv.exe"):
            if cand.is_file():
                return str(cand)
    found = shutil.which("uv") or shutil.which("uv.exe")
    return found


def install(
    project_root: Path | None = None,
    extra_index_url: str | None = None,
    on_line: Callable[[str], None] | None = None,
) -> int:
    """Run `uv pip install torch transformers librosa`, streaming each
    output line to `on_line`. Returns the process return code (0 = OK).

    The default wheels on PyPI handle CPU + CUDA automatically. On Apple
    Silicon torch's stock wheel gives you MPS for free. For AMD on
    Windows pass extra_index_url='https://download.pytorch.org/whl/...'
    when we add a GPU picker; for v1 we install the defaults.
    """
    uv = _find_uv()
    if uv is None:
        if on_line:
            on_line("ERROR: uv not found on PATH. Install uv first.")
        return 127

    cmd: list[str] = [uv, "pip", "install"]
    if extra_index_url:
        cmd += ["--extra-index-url", extra_index_url]
    cmd += list(REQUIRED)

    if on_line:
        on_line(f"$ {' '.join(cmd)}")
    try:
        proc = subprocess.Popen(
            cmd,
            cwd=str(project_root) if project_root else None,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
    except FileNotFoundError as e:
        if on_line:
            on_line(f"ERROR: {e}")
        return 127

    assert proc.stdout is not None
    for raw in proc.stdout:
        line = raw.rstrip()
        if on_line and line:
            on_line(line)
    proc.wait()
    return int(proc.returncode or 0)
