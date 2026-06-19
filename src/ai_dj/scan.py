"""Walk a music library, read audio tags via mutagen, yield per-track metadata."""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import mutagen

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".mp3", ".m4a", ".flac", ".wav", ".aac", ".ogg", ".opus"}
DRM_EXTS = {".m4p"}  # FairPlay DRM — cannot decode
VIDEO_EXTS = {".m4v", ".mp4", ".mov", ".mkv", ".avi", ".wmv", ".webm"}
_NS = uuid.UUID("6f9619ff-8b86-d011-b42d-00c04fc964ff")  # stable namespace for track IDs


@dataclass(frozen=True)
class TrackMeta:
    """Per-track scan result.

    Identity is `uuid5(_NS, rel_path)` where `rel_path` is the file's POSIX
    path **relative to its library root** — so a track keeps the same id
    when the library directory itself moves (different machine, different
    drive letter, WSL vs Windows), and a prebuilt index ships portably.
    The absolute path is recoverable on demand via the library's root."""

    track_id: str
    rel_path: str               # POSIX, relative to the library root
    title: str | None
    artist: str | None
    album: str | None
    duration_s: float | None
    genre: str | None


def _first(v):
    if v is None:
        return None
    if isinstance(v, list):
        return v[0] if v else None
    return v


def rel_track_id(rel_path: str) -> str:
    """uuid5 over the POSIX rel-path. Same id regardless of where the
    library is mounted, as long as the in-library structure is unchanged."""
    return str(uuid.uuid5(_NS, rel_path))


def _parse(path: Path, root: Path) -> TrackMeta | None:
    try:
        tags = mutagen.File(path, easy=True)
    except Exception as e:
        logger.warning("mutagen failed on %s: %s", path, e)
        return None
    if tags is None:
        return None

    duration = getattr(tags.info, "length", None) if getattr(tags, "info", None) else None
    meta = dict(tags) if tags else {}
    rel = path.relative_to(root).as_posix()
    return TrackMeta(
        track_id=rel_track_id(rel),
        rel_path=rel,
        title=_first(meta.get("title")),
        artist=_first(meta.get("artist")) or _first(meta.get("albumartist")),
        album=_first(meta.get("album")),
        duration_s=float(duration) if duration else None,
        genre=_first(meta.get("genre")),
    )


def iter_library(root: Path) -> Iterator[TrackMeta]:
    """Yield TrackMeta for every playable audio file under `root`. Skips DRM
    + unreadable files. Identity is derived from the path *relative to root*
    so call sites can iterate multiple roots and the ids stay stable across
    moves of the library."""
    root = root.resolve()
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in DRM_EXTS:
            logger.info("skip DRM: %s", p)
            continue
        if ext in VIDEO_EXTS:
            logger.info("skip video: %s", p)
            continue
        if ext not in AUDIO_EXTS:
            continue
        meta = _parse(p, root)
        if meta is not None:
            yield meta
