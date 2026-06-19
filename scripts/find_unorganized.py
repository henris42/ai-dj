#!/usr/bin/env python3
"""Report audio files iTunes/Apple Music cannot cleanly organize.

iTunes builds its Media-folder tree as `Artist/Album/NN Title.ext`, where
`Artist` is the album-artist or artist tag. A file is left unorganized (or
dumped in "Unknown Artist"/"Unknown Album") when the tags it needs are
missing, when it has no readable tags at all, or when it is DRM'd.

This script only *reports* — it never moves, renames, or edits anything. It
reuses AI DJ's own tag reader (ai_dj.scan / mutagen) so its view of the
library matches what the app itself sees.

Usage (from the project, via uv so the env is active):

    uv run python scripts/find_unorganized.py [MUSIC_ROOT] [--out FILE] [--layout]

MUSIC_ROOT defaults to $AIDJ_MUSIC, else $AIDJ_BUNDLE/Music. `--layout` also
flags files that are physically not in a two-level Artist/Album/ tree (a
weaker, on-disk signal, separate from the tag problem). `--out` writes the
full path list (one per line, tab-separated reasons) for a later fix step.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

# Reuse the app's extension policy + tag reader so results stay consistent.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_dj.scan import AUDIO_EXTS, DRM_EXTS, VIDEO_EXTS, _first  # noqa: E402

import mutagen  # noqa: E402


def _resolve_root(arg: str | None) -> Path:
    if arg:
        return Path(arg)
    env = os.environ.get("AIDJ_MUSIC")
    if env:
        return Path(env)
    bundle = os.environ.get("AIDJ_BUNDLE")
    if bundle:
        return Path(bundle) / "Music"
    sys.exit("No MUSIC_ROOT given and neither $AIDJ_MUSIC nor $AIDJ_BUNDLE set.")


def reasons_for(path: Path, root: Path, check_layout: bool) -> list[str]:
    """Return the reasons iTunes can't cleanly organize this file ([] = fine)."""
    ext = path.suffix.lower()
    if ext in DRM_EXTS:
        return ["drm (.m4p — cannot decode/manage)"]

    try:
        tags = mutagen.File(path, easy=True)
    except Exception as e:  # noqa: BLE001
        return [f"unreadable ({type(e).__name__})"]
    if tags is None:
        return ["no readable tags"]

    meta = dict(tags)
    artist = _first(meta.get("artist")) or _first(meta.get("albumartist"))
    album = _first(meta.get("album"))
    title = _first(meta.get("title"))

    out: list[str] = []
    if not (artist and str(artist).strip()):
        out.append("missing artist/albumartist")
    if not (album and str(album).strip()):
        out.append("missing album")
    if not (title and str(title).strip()):
        out.append("missing title")

    if check_layout and not out:
        # Two-level tree expected: <root>/Artist/Album/file.ext
        rel_parts = path.relative_to(root).parts
        if len(rel_parts) != 3:
            out.append(f"off-tree (depth {len(rel_parts) - 1}, expected 2)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Report files iTunes can't organize.")
    ap.add_argument("root", nargs="?", help="Music root (else $AIDJ_MUSIC / $AIDJ_BUNDLE/Music)")
    ap.add_argument("--out", help="write 'path<TAB>reasons' lines here")
    ap.add_argument("--layout", action="store_true",
                    help="also flag files not in a 2-level Artist/Album/ tree")
    args = ap.parse_args()

    root = _resolve_root(args.root).resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    total = 0
    flagged: list[tuple[Path, list[str]]] = []
    reason_counts: Counter[str] = Counter()

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in VIDEO_EXTS or (ext not in AUDIO_EXTS and ext not in DRM_EXTS):
            continue
        total += 1
        rs = reasons_for(p, root, args.layout)
        if rs:
            flagged.append((p, rs))
            for r in rs:
                reason_counts[r] += 1

    print(f"\nScanned {total} audio files under {root}")
    print(f"iTunes-unorganizable: {len(flagged)} "
          f"({(len(flagged) / total * 100 if total else 0):.1f}%)\n")
    for reason, n in reason_counts.most_common():
        print(f"  {n:5d}  {reason}")

    if flagged and not args.out:
        print("\nFirst 25:")
        for p, rs in flagged[:25]:
            print(f"  {p}  —  {', '.join(rs)}")
        if len(flagged) > 25:
            print(f"  … {len(flagged) - 25} more (use --out to dump all)")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            for p, rs in flagged:
                fh.write(f"{p}\t{'; '.join(rs)}\n")
        print(f"\nFull list written to {args.out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
