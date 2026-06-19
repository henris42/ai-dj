#!/usr/bin/env python3
"""Silent launcher for the native-Windows AI DJ GUI.

The Start Menu shortcut targets this via `pythonw.exe`, so:

  - no console window ever flashes — `pythonw` is windowless;
  - the GUI is spawned from the project's venv with CREATE_NO_WINDOW and
    DETACHED_PROCESS so the launcher exits immediately and the GUI keeps
    running as its own process (no hung "launcher running" indicator in
    the Start Menu).

Embedded Qdrant means there's no server to bring up — the GUI opens the
file-backed store at `config.db_root()` itself. Failures pop a Windows
message box since there's no console to print to.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# Windows-only: hide child consoles. No-op on other platforms.
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_DETACHED = getattr(subprocess, "DETACHED_PROCESS", 0)


def _msgbox(text: str, title: str = "AI DJ") -> None:
    try:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, text, title, 0x10)
    except Exception:  # noqa: BLE001
        pass


def venv_pythonw() -> Path:
    return ROOT / ".venv" / "Scripts" / "pythonw.exe"


def main() -> int:
    pyw = venv_pythonw()
    if not pyw.exists():
        _msgbox(
            f".venv\\Scripts\\pythonw.exe not found at:\n  {pyw}\n\n"
            f"Run once on Windows:\n  uv sync --project \"{ROOT}\""
        )
        return 1
    try:
        subprocess.Popen(
            [str(pyw), "-m", "ai_dj.gui.app"], cwd=str(ROOT),
            creationflags=_NO_WINDOW | _DETACHED, close_fds=True,
        )
    except Exception as e:  # noqa: BLE001
        _msgbox(f"Couldn't start AI DJ:\n\n{type(e).__name__}: {e}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
