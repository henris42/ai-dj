#!/usr/bin/env python3
"""Create the Windows Start Menu entry for AI DJ.

One-time, idempotent. Drops "AI DJ.lnk" into your personal Start Menu
Programs folder (no admin, no system-wide install) pointing at `start.cmd`
in the project root, with the project icon. After this, Start → "AI DJ"
launches the app; right-click → Pin to taskbar/start as usual.

Run once on Windows:

    uv run python scripts/install_shortcut.py

Replaces any existing shortcut at the same path (so it overwrites the old
launch.ps1-based one cleanly).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def main() -> int:
    if sys.platform != "win32":
        sys.exit("This installer only runs on Windows. (You're on "
                 f"{sys.platform!r}.)")
    try:
        from win32com.client import Dispatch  # type: ignore[import-not-found]
    except ImportError:
        sys.exit("pywin32 not installed. Run:  uv pip install pywin32")

    project = Path(__file__).resolve().parents[1]
    pyw = project / ".venv" / "Scripts" / "pythonw.exe"
    launcher = project / "start_windows.py"
    icon = project / "assets" / "icon.ico"
    if not pyw.exists():
        sys.exit(f".venv pythonw.exe not found at {pyw}. "
                 f"Run once: uv sync --project \"{project}\"")
    if not launcher.exists():
        sys.exit(f"start_windows.py not found at {launcher}")
    if not icon.exists():
        sys.exit(f"icon not found at {icon}")

    start_menu = Path(os.environ["APPDATA"]) / "Microsoft" / "Windows" / "Start Menu" / "Programs"
    lnk_path = start_menu / "AI DJ.lnk"
    start_menu.mkdir(parents=True, exist_ok=True)

    shell = Dispatch("WScript.Shell")
    lnk = shell.CreateShortCut(str(lnk_path))
    lnk.TargetPath = str(pyw)               # pythonw is windowless — no console flash
    lnk.Arguments = f'"{launcher}"'
    lnk.WorkingDirectory = str(project)
    lnk.IconLocation = f"{icon},0"
    lnk.Description = "AI DJ"
    lnk.WindowStyle = 1                     # normal — pythonw shows no window anyway
    lnk.Save()

    print(f"Start Menu entry created: {lnk_path}")
    print("Open Start, type 'AI DJ' — right-click to pin to taskbar.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
