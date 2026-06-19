"""User-editable settings, persisted under the bundle root.

Tiny JSON store — no UI for editing yet beyond the in-app SetupWindow.

Layout in `<bundle>/settings.json`:

    {
      "version": 1,
      "library_sources": [
        {"name": "iTunes Library", "path": "D:\\\\ai-dj\\\\Music"},
        ...
      ]
    }

Per "show your mp3 databases, select them all, click Teach" — sources are
the multi-select unit. Each source has a display name (whatever the user
wants to call it) and an absolute path. The indexer iterates each source.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from . import config

SETTINGS_VERSION = 1


@dataclass
class LibrarySource:
    name: str
    path: str


@dataclass
class Settings:
    version: int = SETTINGS_VERSION
    library_sources: list[LibrarySource] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        return cls(
            version=int(d.get("version", SETTINGS_VERSION)),
            library_sources=[
                LibrarySource(name=str(s.get("name", "")),
                              path=str(s.get("path", "")))
                for s in (d.get("library_sources") or [])
                if s.get("path")
            ],
        )

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "library_sources": [
                {"name": s.name, "path": s.path} for s in self.library_sources
            ],
        }


def settings_path() -> Path:
    return config.bundle_root() / "settings.json"


def load() -> Settings:
    p = settings_path()
    if not p.exists():
        return Settings()
    try:
        return Settings.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except (json.JSONDecodeError, OSError):
        return Settings()


def save(s: Settings) -> None:
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    # Atomic-ish: write to a temp then rename.
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(s.to_dict(), indent=2), encoding="utf-8")
    tmp.replace(p)


def add_source(name: str, path: str) -> Settings:
    s = load()
    abs_path = str(Path(path).resolve())
    # No duplicates by path.
    for src in s.library_sources:
        if str(Path(src.path).resolve()) == abs_path:
            return s
    s.library_sources.append(LibrarySource(name=name, path=abs_path))
    save(s)
    return s


def remove_source(path: str) -> Settings:
    s = load()
    abs_target = str(Path(path).resolve())
    s.library_sources = [
        src for src in s.library_sources
        if str(Path(src.path).resolve()) != abs_target
    ]
    save(s)
    return s
