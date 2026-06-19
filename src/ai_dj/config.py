"""Bundle / library path configuration.

The "bundle" is a single directory that holds the working data for AI DJ:

    <bundle>/
      db/            embedded Qdrant storage (no server)
      Music/         the music library (iTunes-style: Artist/Album/file.ext)
      Photos/        fetched artist photos (Artist/artist_NN.jpg)

The bundle root comes from `AIDJ_BUNDLE`, else a per-OS default. Sub-paths
can be overridden individually (`AIDJ_DB`, `AIDJ_MUSIC`, `AIDJ_PHOTOS`) so
the music library can live outside the bundle (typical iTunes setup) while
the small db/photos stay under the bundle.

No third-party dep — per-OS defaults hand-rolled rather than pulling in
`platformdirs` just for two constants.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_APP = "AI DJ"


def bundle_root() -> Path:
    p = os.environ.get("AIDJ_BUNDLE")
    if p:
        return Path(p)
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser(r"~\AppData\Local")
        return Path(base) / _APP
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / _APP
    # Linux + other POSIX: XDG.
    xdg = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(xdg) / "ai-dj"


def db_root() -> Path:
    p = os.environ.get("AIDJ_DB")
    return Path(p) if p else bundle_root() / "db"


def music_root() -> Path:
    p = os.environ.get("AIDJ_MUSIC")
    return Path(p) if p else bundle_root() / "Music"


def photos_root() -> Path:
    p = os.environ.get("AIDJ_PHOTOS") or os.environ.get("PHOTO_ROOT")
    return Path(p) if p else bundle_root() / "Photos"


def ensure_dirs() -> None:
    """Create the bundle layout if it doesn't exist yet (idempotent)."""
    for d in (db_root(), music_root(), photos_root()):
        d.mkdir(parents=True, exist_ok=True)
