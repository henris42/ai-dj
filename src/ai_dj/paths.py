"""Path translation between WSL (/mnt/e/...) and Windows (E:\\...).

Tracks are embedded from the WSL side, so Qdrant stores POSIX paths like
`/mnt/e/Music/...`. When the player runs natively on Windows we need to convert
back to drive-letter form before handing paths to libmpv."""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

_WSL_MNT_RE = re.compile(r"^/mnt/([a-zA-Z])/(.*)$")


def configure_libmpv_dll() -> None:
    """On Windows, python-mpv's ctypes load looks for `libmpv-2.dll` on %PATH%.
    Ensure the project-bundled copy is findable. Call this before importing
    `mpv` (directly or transitively via ai_dj.player)."""
    if sys.platform != "win32":
        return
    # gui/app.py -> ai_dj/gui -> ai_dj -> src -> project root
    project_root = Path(__file__).resolve().parents[2]
    dll = project_root / "libmpv-2.dll"
    if not dll.exists():
        return
    dll_dir = str(project_root)
    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
    add = getattr(os, "add_dll_directory", None)
    if add is not None:
        try:
            add(dll_dir)
        except OSError:
            pass


def resolve_for_player(path: str) -> str:
    """Return a path the local libmpv/OS can open for the current platform.

    On Windows: translate `/mnt/<drive>/rest` → `<DRIVE>:\\rest`.
    On Linux (including WSL): return as-is.
    """
    if sys.platform != "win32":
        return path
    m = _WSL_MNT_RE.match(path)
    if not m:
        return path
    drive, rest = m.group(1).upper(), m.group(2)
    return f"{drive}:\\" + rest.replace("/", "\\")


def is_windows() -> bool:
    return sys.platform == "win32"
