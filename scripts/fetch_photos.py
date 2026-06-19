#!/usr/bin/env python3
"""Batch fetcher: public artist photos for the library, tiered by track count.

The reusable fetch logic lives in `ai_dj.photos` so the GUI can call the
same code from a worker thread (auto-fetch while playing). This script is
the CLI wrapper that walks the Qdrant index, decides per-artist targets,
and reports progress / writes a manifest.

Sources: Wikipedia REST (page summary + media-list) + Wikimedia Commons
search. No API key. Polite throttle. Resumable.

SAFE BY DEFAULT: dry-run unless --apply.

    PHOTO_ROOT=D:\\ai-dj\\Photos uv run python scripts/fetch_photos.py [--apply]
        [--limit N] [--max-edge 1920] [--per-artist 0]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_dj.index import COLLECTION, TrackIndex  # noqa: E402
from ai_dj.photos import (  # noqa: E402
    already_have, find_image_urls, normalise_and_save, photo_root,
    safe, target_photos, _get,
)
import time

MANIFEST = Path("data/photos_fetched.tsv")


def artist_track_counts() -> dict[str, int]:
    """Count tracks per artist across the whole index."""
    idx = TrackIndex()
    counts: dict[str, int] = {}
    offset = None
    while True:
        recs, offset = idx.client.scroll(
            COLLECTION, limit=1024, offset=offset,
            with_payload=True, with_vectors=False,
        )
        for r in recs:
            a = (r.payload or {}).get("artist")
            if a and str(a).strip():
                counts[str(a).strip()] = counts.get(str(a).strip(), 0) + 1
        if offset is None:
            break
    return counts


def main() -> int:
    ap = argparse.ArgumentParser(description="Fetch public artist photos.")
    ap.add_argument("--apply", action="store_true", help="actually download (default: dry-run)")
    ap.add_argument("--limit", type=int, default=0, help="cap at N artists this run")
    ap.add_argument("--max-edge", type=int, default=1920, help="resize: max long edge in px")
    ap.add_argument("--per-artist", type=int, default=0,
                    help="override cap on photos per artist (0 = use tiered target)")
    args = ap.parse_args()

    root = photo_root()
    counts = artist_track_counts()
    print(f"library has {len(counts)} unique artists. photo root: {root}")

    pending: list[tuple[str, int]] = []
    covered = 0
    for a, n in sorted(counts.items()):
        target = target_photos(n)
        if args.per_artist > 0:
            target = min(target, args.per_artist)
        if already_have(root / safe(a), target):
            covered += 1
        else:
            pending.append((a, target))
    plan = pending[: args.limit] if args.limit else pending
    cap = (f", capped at {len(plan)} this run"
           if args.limit and len(plan) < len(pending) else "")
    print(f"to fetch: {len(pending)} artists pending"
          f"  ({covered} already covered{cap})")

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    saved = nophoto = err = 0
    with open(MANIFEST, "a", encoding="utf-8") as man:
        for i, (artist, want) in enumerate(plan, 1):
            try:
                urls = find_image_urls(artist, want)
            except Exception as e:  # noqa: BLE001
                err += 1
                print(f"  [{i}/{len(plan)}] ERR {artist}: {type(e).__name__} {e}", file=sys.stderr)
                man.write(f"{artist}\tERR\t{e}\n")
                continue
            if not urls:
                nophoto += 1
                print(f"  [{i}/{len(plan)}] -- no photo for {artist}")
                man.write(f"{artist}\tNONE\t\n")
                continue
            print(f"  [{i}/{len(plan)}] OK  {artist} ({counts[artist]} tracks) — "
                  f"found {len(urls)}/{want}")
            if not args.apply:
                continue
            artist_dir = root / safe(artist)
            for j, url in enumerate(urls, 1):
                dest = artist_dir / f"artist_{j:02d}.jpg"
                if dest.exists():
                    continue
                try:
                    blob = _get(url, timeout=30)
                    time.sleep(0.12)
                    normalise_and_save(blob, dest, args.max_edge)
                    man.write(f"{artist}\tOK\t{dest}\n")
                    saved += 1
                except Exception as e:  # noqa: BLE001
                    err += 1
                    print(f"      [{j}/{len(urls)}] {url[:80]}  {type(e).__name__} {e}",
                          file=sys.stderr)
                    man.write(f"{artist}\tDLFAIL\t{e}\n")

    print(f"\nsaved: {saved}   no-photo: {nophoto}   errors: {err}")
    if not args.apply:
        print("DRY-RUN. Re-run with --apply to actually download + resize.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
