#!/usr/bin/env python3
"""Push AcoustID-recovered metadata into the app (the Qdrant index).

Reads identify_untagged.py's results log and, for every confident match,
updates that track's payload in Qdrant (title + artist + provenance) so AI
DJ immediately shows the right metadata for the previously-untagged files.

Track identity is the same `uuid5(_NS, path)` the indexer uses, so updates
land on the existing points (no re-embedding, no new points). Only tracks
already present in the collection are touched; missing ones are reported.

SAFE BY DEFAULT: dry-run unless --apply.

    uv run python scripts/apply_recovered.py [--results data/identify_results.tsv]
        [--min-score 0.90] [--apply]
"""
from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_dj.index import TrackIndex  # noqa: E402
from ai_dj.scan import _NS  # noqa: E402  (same namespace the indexer keys with)


def main() -> int:
    ap = argparse.ArgumentParser(description="Apply recovered metadata to the Qdrant index.")
    ap.add_argument("--results", default="data/identify_results.tsv")
    ap.add_argument("--min-score", type=float, default=0.90,
                    help="only apply matches at/above this AcoustID score")
    ap.add_argument("--apply", action="store_true", help="write to Qdrant (default: dry-run)")
    args = ap.parse_args()

    res = Path(args.results)
    if not res.exists():
        sys.exit(f"No results file: {res} (run identify_untagged.py first)")

    # Parse OK rows: path \t OK \t score \t artist \t title
    updates: list[tuple[str, str, str, float]] = []
    skipped_low = 0
    for line in res.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 5 or parts[1] != "OK":
            continue
        path, _, score_s, artist, title = parts[0], parts[1], parts[2], parts[3], parts[4]
        try:
            score = float(score_s)
        except ValueError:
            continue
        if score < args.min_score or not (artist and title):
            skipped_low += 1
            continue
        tid = str(uuid.uuid5(_NS, path))
        updates.append((tid, artist, title, score))

    if not updates:
        print(f"No confident matches at >= {args.min_score}. (skipped {skipped_low})")
        return 0

    idx = TrackIndex()
    # Only touch points that actually exist in the collection.
    present = idx.existing_ids([u[0] for u in updates])
    landing = [u for u in updates if u[0] in present]
    missing = len(updates) - len(landing)

    print(f"{len(updates)} confident matches  "
          f"(>= {args.min_score}; {skipped_low} below threshold)")
    print(f"  in collection, will update: {len(landing)}")
    print(f"  not in collection (skip):   {missing}")
    for tid, artist, title, score in landing[:20]:
        print(f"    {score:.2f}  {artist} — {title}")
    if len(landing) > 20:
        print(f"    … {len(landing) - 20} more")

    if not args.apply:
        print("\nDRY-RUN — Qdrant unchanged. Re-run with --apply to write.")
        return 0

    idx.set_recovered_meta(landing)
    print(f"\nUpdated {len(landing)} track payloads in Qdrant "
          f"(title/artist + meta_source=acoustid).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
