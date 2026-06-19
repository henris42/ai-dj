"""Sets — captured paths through the library.

A *set* is the AI DJ unit of session: an ordered list of track ids walked
in a specific order, with a name and a timestamp. Different from a
playlist in spirit — playlists are curated orderings the user assembles;
sets are *journeys the planner walked*, captured so they can be relived
or branched from.

On disk: `<bundle>/sets/<id>.json`, one file per set. Plain JSON, no
schema migration ceremony for the first pass.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from . import config


@dataclass
class SetRecord:
    id: str
    name: str
    created_at: float          # unix seconds
    track_ids: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "created_at": self.created_at,
            "track_ids": list(self.track_ids),
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "SetRecord":
        return cls(
            id=str(d.get("id") or uuid.uuid4()),
            name=str(d.get("name") or "untitled"),
            created_at=float(d.get("created_at") or time.time()),
            track_ids=[str(t) for t in (d.get("track_ids") or [])],
            notes=str(d.get("notes") or ""),
        )


def _sets_dir() -> Path:
    p = config.bundle_root() / "sets"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_set(name: str, track_ids: list[str], notes: str = "") -> SetRecord:
    rec = SetRecord(
        id=str(uuid.uuid4()),
        name=name or _auto_name(),
        created_at=time.time(),
        track_ids=list(track_ids),
        notes=notes,
    )
    path = _sets_dir() / f"{rec.id}.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(rec.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(path)
    return rec


def load_set(set_id: str) -> SetRecord | None:
    path = _sets_dir() / f"{set_id}.json"
    if not path.exists():
        return None
    try:
        return SetRecord.from_dict(json.loads(path.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return None


def list_sets() -> list[SetRecord]:
    """Most-recent first."""
    out: list[SetRecord] = []
    for path in _sets_dir().glob("*.json"):
        try:
            out.append(SetRecord.from_dict(
                json.loads(path.read_text(encoding="utf-8"))))
        except (json.JSONDecodeError, OSError):
            continue
    out.sort(key=lambda r: r.created_at, reverse=True)
    return out


def delete_set(set_id: str) -> bool:
    path = _sets_dir() / f"{set_id}.json"
    try:
        path.unlink()
        return True
    except FileNotFoundError:
        return False


def _auto_name() -> str:
    t = time.localtime()
    return f"Set — {time.strftime('%Y-%m-%d %H:%M', t)}"
