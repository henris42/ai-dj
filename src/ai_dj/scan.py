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
    track_id: str
    path: str
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


def _parse(path: Path) -> TrackMeta | None:
    try:
        tags = mutagen.File(path, easy=True)
    except Exception as e:
        logger.warning("mutagen failed on %s: %s", path, e)
        return None
    if tags is None:
        return None

    duration = getattr(tags.info, "length", None) if getattr(tags, "info", None) else None
    meta = dict(tags) if tags else {}
    return TrackMeta(
        track_id=str(uuid.uuid5(_NS, str(path))),
        path=str(path),
        title=_first(meta.get("title")),
        artist=_first(meta.get("artist")) or _first(meta.get("albumartist")),
        album=_first(meta.get("album")),
        duration_s=float(duration) if duration else None,
        genre=_first(meta.get("genre")),
    )


def iter_library(root: Path) -> Iterator[TrackMeta]:
    """Yield TrackMeta for every playable audio file under root. Skips DRM + unreadable files."""
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
        meta = _parse(p)
        if meta is not None:
            yield meta
