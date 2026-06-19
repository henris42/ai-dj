#!/usr/bin/env python3
"""Rename audio files whose *filename* is nonsensical, using ID3 tags.

Principle: **ID3 tags are definitive.** A file is renamed only when the tags
give a real title and the current filename is junk (download gibberish, hash
strings, "track01", all-digits, etc.). The new name is the tag-derived,
iTunes-style `NN Title.ext` (track number prefix when present).

Older files with **no ID3 at all** have no definitive source — this script
**never fabricates** a name from a guess. Such files are reported under
"skipped: no tags" for manual handling, not renamed. Files whose name is
already sensible are left untouched.

Renames happen *in place* (same folder — this is a renamer, not an
organizer). Collisions get a " (2)" suffix. An undo log (old<TAB>new) is
written so every change is reversible.

SAFE BY DEFAULT: dry-run unless --apply is given.

    uv run python scripts/rename_nonsensical.py [MUSIC_ROOT] [--apply]
                                                [--undo-log FILE]

MUSIC_ROOT defaults to $AIDJ_MUSIC, else $AIDJ_BUNDLE/Music.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_dj.scan import AUDIO_EXTS, DRM_EXTS, VIDEO_EXTS, _first  # noqa: E402

import mutagen  # noqa: E402

# Filesystem-illegal / troublesome characters (Windows superset, so the
# result is valid on the iTunes-compatible NTFS tree too).
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Stem patterns that scream "no human named this".
_JUNK_PATTERNS = [
    re.compile(r"^\s*$"),
    re.compile(r"^[\d\W_]+$"),                       # only digits/punctuation
    re.compile(r"^(track|audio|audiotrack|title)\s*[\d_\-]*$", re.I),
    re.compile(r"^(untitled|unknown|unnamed|new recording|temp|tmp|output|file|download|recording)\b", re.I),
    re.compile(r"^[0-9a-f]{16,}$", re.I),            # hex / hash blob
    re.compile(r"^[a-z0-9]{20,}$", re.I),            # long random token, no spaces
    re.compile(r".*\(\d+\)\s*$"),                    # "something(1)" download dupes
    re.compile(r"^\d{1,3}[\s\-_.]*$"),               # bare track number
]


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def is_nonsensical(stem: str, title: str | None) -> bool:
    """True if the filename is junk. Conservative: if it already reflects the
    title, it's fine regardless of pattern."""
    if title and _norm(title) and _norm(title) in _norm(stem):
        return False
    return any(p.match(stem) for p in _JUNK_PATTERNS)


def sanitize(name: str) -> str:
    name = _ILLEGAL.sub("", name)
    name = re.sub(r"\s+", " ", name).strip(" .")
    return name[:180]  # keep well under path limits


def proposed_stem(meta: dict) -> str | None:
    """iTunes-style stem from tags, or None if tags can't yield a real name."""
    title = _first(meta.get("title"))
    if not (title and str(title).strip()):
        return None
    title = str(title).strip()
    track = _first(meta.get("tracknumber"))
    prefix = ""
    if track:
        m = re.match(r"\s*(\d+)", str(track))
        if m:
            prefix = f"{int(m.group(1)):02d} "
    stem = sanitize(f"{prefix}{title}")
    return stem or None


def unique_target(path: Path, new_stem: str) -> Path:
    target = path.with_name(new_stem + path.suffix)
    if not target.exists() or target == path:
        return target
    i = 2
    while True:
        cand = path.with_name(f"{new_stem} ({i}){path.suffix}")
        if not cand.exists():
            return cand
        i += 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Rename nonsensical audio filenames from ID3 tags.")
    ap.add_argument("root", nargs="?", help="Music root (else $AIDJ_MUSIC / $AIDJ_BUNDLE/Music)")
    ap.add_argument("--apply", action="store_true", help="actually rename (default: dry-run)")
    ap.add_argument("--undo-log", default="data/rename_undo.tsv",
                    help="where to write old<TAB>new (for reverting)")
    args = ap.parse_args()

    if args.root:
        root = Path(args.root)
    elif os.environ.get("AIDJ_MUSIC"):
        root = Path(os.environ["AIDJ_MUSIC"])
    elif os.environ.get("AIDJ_BUNDLE"):
        root = Path(os.environ["AIDJ_BUNDLE"]) / "Music"
    else:
        sys.exit("No MUSIC_ROOT and neither $AIDJ_MUSIC nor $AIDJ_BUNDLE set.")
    root = root.resolve()
    if not root.is_dir():
        sys.exit(f"Not a directory: {root}")

    total = renamed = skip_sensible = skip_no_tags = skip_no_title = 0
    plan: list[tuple[Path, Path]] = []
    no_tag_files: list[Path] = []

    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in VIDEO_EXTS or ext in DRM_EXTS or ext not in AUDIO_EXTS:
            continue
        total += 1

        try:
            tags = mutagen.File(p, easy=True)
        except Exception:  # noqa: BLE001
            tags = None
        meta = dict(tags) if tags else {}
        title = _first(meta.get("title"))

        if not is_nonsensical(p.stem, title):
            skip_sensible += 1
            continue
        # Name is junk. Tags are the only sanctioned source.
        if not meta:
            skip_no_tags += 1            # the older, ID3-less files
            no_tag_files.append(p)
            continue
        new_stem = proposed_stem(meta)
        if not new_stem:
            skip_no_title += 1           # has some tags but no usable title
            continue
        target = unique_target(p, new_stem)
        if target == p:
            skip_sensible += 1
            continue
        plan.append((p, target))
        renamed += 1

    print(f"\nScanned {total} audio files under {root}")
    print(f"  to rename (junk name + usable tags): {renamed}")
    print(f"  skipped, name already sensible:      {skip_sensible}")
    print(f"  skipped, NO ID3 tags (older files):  {skip_no_tags}")
    print(f"  skipped, tags present but no title:  {skip_no_title}")

    for old, new in plan[:30]:
        print(f"  {old.name}  ->  {new.name}")
    if len(plan) > 30:
        print(f"  … {len(plan) - 30} more")

    if no_tag_files:
        nl = Path("data/no_id3.txt")
        nl.parent.mkdir(parents=True, exist_ok=True)
        nl.write_text("\n".join(str(x) for x in no_tag_files), encoding="utf-8")
        print(f"\nNo-ID3 junk-named files (need manual handling): {len(no_tag_files)}"
              f"\n  list: {nl}")

    if not args.apply:
        print("\nDRY-RUN. Nothing changed. Re-run with --apply to perform these renames.")
        return 0

    log = Path(args.undo_log)
    log.parent.mkdir(parents=True, exist_ok=True)
    done = 0
    with open(log, "w", encoding="utf-8") as fh:
        for old, new in plan:
            try:
                old.rename(new)
                fh.write(f"{new}\t{old}\n")  # new<TAB>old: replay to undo
                done += 1
            except OSError as e:
                print(f"  FAILED {old}: {e}", file=sys.stderr)
    print(f"\nRenamed {done}/{len(plan)} files. Undo log: {log}")
    print("To revert: read the log, rename column-1 back to column-2.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
