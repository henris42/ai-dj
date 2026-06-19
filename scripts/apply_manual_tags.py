#!/usr/bin/env python3
"""Push manually-added tags from D:\\ai-dj\\unmatched into the app (Qdrant).

The unmatched files were moved out to a review folder and hand-tagged in a
GUI tagger. Their ID3 is now the authoritative source ("ID3 is definitive",
and human > fingerprint). This reads those tags and writes them onto the
matching Qdrant points, stamped meta_source=manual so they outrank any
acoustid guess and survive a re-run of the recognition pipeline.

Identity: the index keys track_id = uuid5(_NS, <original E: path>). Files
were only moved on the D: copy, so we reconstruct the E: path from the
review-folder-relative path and key off that.

SAFE BY DEFAULT: dry-run unless --apply.

    uv run python scripts/apply_manual_tags.py [--apply]
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_dj.index import COLLECTION, TrackIndex  # noqa: E402
from ai_dj.scan import _NS  # noqa: E402

REVIEW_ROOT = Path("/mnt/d/ai-dj/unmatched")
# What the indexer originally walked (str(Path) of this + rel == the key).
E_MUSIC_ROOT = "/mnt/e/Music/iTunes/iTunes Media/Music"
AUDIO = {".mp3", ".m4a", ".flac", ".wav", ".aac", ".ogg", ".opus"}


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply hand-added tags to the index.")
    ap.add_argument("--apply", action="store_true", help="write to Qdrant (default: dry-run)")
    args = ap.parse_args()

    import mutagen
    updates: list[tuple[str, str, str, str]] = []  # (tid, artist, title, relname)
    skipped_untitled = 0
    for p in REVIEW_ROOT.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in AUDIO:
            continue
        try:
            t = mutagen.File(p, easy=True)
        except Exception:  # noqa: BLE001
            t = None
        meta = dict(t) if t else {}
        title = (meta.get("title") or [None])[0]
        artist = (meta.get("artist") or meta.get("albumartist") or [None])[0]
        if not (title and str(title).strip()):
            skipped_untitled += 1
            continue
        rel = p.relative_to(REVIEW_ROOT).as_posix()
        e_path = f"{E_MUSIC_ROOT}/{rel}"          # original indexed path
        tid = str(uuid.uuid5(_NS, e_path))
        updates.append((tid, str(artist or "").strip(), str(title).strip(), rel))

    print(f"hand-tagged files found: {len(updates)}  (skipped {skipped_untitled} still untitled)")

    idx = TrackIndex()
    present = idx.existing_ids([u[0] for u in updates]) if updates else set()
    landing = [u for u in updates if u[0] in present]
    print(f"  match a point in the index: {len(landing)}")
    print(f"  no matching point (skip):   {len(updates) - len(landing)}")
    for _tid, a, ti, rel in landing[:30]:
        print(f"    {a or '?'} - {ti}   [{rel}]")

    if not args.apply:
        print("\nDRY-RUN - Qdrant unchanged. Re-run with --apply to write.")
        return 0

    for tid, artist, title, _rel in landing:
        payload = {"title": title, "meta_source": "manual"}
        if artist:
            payload["artist"] = artist
        idx.client.set_payload(COLLECTION, payload=payload, points=[tid], wait=False)
    print(f"\nUpdated {len(landing)} points (meta_source=manual, outranks acoustid).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
