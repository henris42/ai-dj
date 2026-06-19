"""Resolve track rel-paths to local file paths the OS / libmpv can open.

The index now stores **paths relative to the music library root**, so
resolution = `music_root / rel_path` then OS-native normalisation. The old
WSL-`/mnt/<drive>` ↔ Windows-`<DRIVE>:\\` regex is no longer needed: the
music root itself is platform-native (configured via `AIDJ_MUSIC` or the
bundle default), so the same `rel_path` produces a valid Windows or POSIX
path just by joining with that machine's root.

A small legacy fallback remains for the migration window: anything that
still looks like an absolute path is returned (Windows-translated if
needed) so the player can play a track whose payload hasn't been re-keyed
yet.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

from . import config

_WSL_MNT_RE = re.compile(r"^/mnt/([a-zA-Z])/(.*)$")


def configure_libmpv_dll() -> None:
    """On Windows, python-mpv's ctypes load looks for `libmpv-2.dll` on %PATH%.
    Ensure the project-bundled copy is findable. Call this before importing
    `mpv` (directly or transitively via ai_dj.player)."""
    if sys.platform != "win32":
        return
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


def _looks_absolute(s: str) -> bool:
    return s.startswith(("/", "\\")) or (len(s) >= 2 and s[1] == ":")


def resolve_for_player(rel_or_abs: str, music_root: "Path | None" = None) -> str:
    """Return an OS-native absolute path libmpv can open.

    Primary case: `rel_or_abs` is a POSIX path relative to `music_root`
    (the new format). We join + normalise. `music_root` defaults to
    `config.music_root()` so call sites that don't know about the bundle
    keep working.

    Legacy fallback: if `rel_or_abs` is already absolute (e.g. an unmigrated
    payload still carrying `/mnt/e/Music/...`), we return it as-is on POSIX,
    and translate `/mnt/<drive>/...` → `<DRIVE>:\\...` on Windows so playback
    keeps working until migration runs."""
    if _looks_absolute(rel_or_abs):
        if sys.platform == "win32":
            m = _WSL_MNT_RE.match(rel_or_abs)
            if m:
                drive, rest = m.group(1).upper(), m.group(2)
                return f"{drive}:\\" + rest.replace("/", "\\")
        return rel_or_abs
    root = music_root if music_root is not None else config.music_root()
    return str((root / rel_or_abs).resolve())


def is_windows() -> bool:
    return sys.platform == "win32"
