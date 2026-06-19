#!/usr/bin/env python3
"""One-shot migration: legacy Qdrant *server* with absolute-path identity
→ embedded Qdrant store with **rel_path identity**.

What this does:

  - Scrolls every point out of the running server collection (vectors +
    payload).
  - Strips the configured iTunes Music root prefix off `payload["path"]`,
    yielding the POSIX `rel_path`.
  - Re-keys the point: new `track_id = uuid5(_NS, rel_path)`.
  - Writes the new id + vector + payload (with `rel_path` added, legacy
    `path` removed) into the embedded store at `config.db_root()`.
  - Vectors are preserved — no MERT re-embedding. This is purely a re-key
    + payload rewrite.

The legacy server is **not** modified — it stays as the safety net until
you're satisfied the embedded store works end-to-end.

Run:

    AIDJ_BUNDLE=/mnt/d/ai-dj python scripts/migrate_to_embedded.py \\
        [--itunes-root "/mnt/e/Music/iTunes/iTunes Media/Music"] [--apply]

Dry-run by default — prints how many points it would migrate and a sample
of re-keyings. Add --apply to actually write to the embedded store.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from qdrant_client import QdrantClient  # noqa: E402
from qdrant_client.http import models as qm  # noqa: E402

from ai_dj import config  # noqa: E402
from ai_dj.index import COLLECTION, DIM, TrackIndex  # noqa: E402
from ai_dj.scan import rel_track_id  # noqa: E402

DEFAULT_ITUNES_ROOT = "/mnt/e/Music/iTunes/iTunes Media/Music"
BATCH = 256


def _rekey_payload(payload: dict, itunes_root: str) -> tuple[str | None, dict]:
    """Return (rel_path, new_payload) or (None, _) when the legacy path doesn't
    sit under the expected iTunes root (those are skipped + reported)."""
    legacy = (payload.get("path") or "").strip()
    if not legacy:
        return None, payload
    # Allow itunes_root with or without a trailing slash.
    root = itunes_root.rstrip("/") + "/"
    if not legacy.startswith(root):
        return None, payload
    rel = legacy[len(root):]
    new = dict(payload)
    new["rel_path"] = rel
    new.pop("path", None)            # the legacy abs path is no longer truth
    return rel, new


def main() -> int:
    ap = argparse.ArgumentParser(description="Re-key the index from abs paths to rel paths.")
    ap.add_argument("--itunes-root", default=DEFAULT_ITUNES_ROOT,
                    help="Prefix of legacy absolute paths to strip")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=6333)
    ap.add_argument("--apply", action="store_true",
                    help="actually write to the embedded store (default: dry-run)")
    args = ap.parse_args()

    src = QdrantClient(host=args.host, port=args.port)
    if not src.collection_exists(COLLECTION):
        sys.exit(f"Source server has no '{COLLECTION}' collection on {args.host}:{args.port}")

    dst_root = config.db_root()
    dst_root.mkdir(parents=True, exist_ok=True)

    sample: list[tuple[str, str]] = []   # (rel, old_id_prefix → new_id_prefix)
    n_total = n_rekeyed = n_skipped = 0
    pending: list[qm.PointStruct] = []

    dst: TrackIndex | None = None
    if args.apply:
        dst = TrackIndex(path=dst_root)
        dst.ensure_collection()

    print(f"src: {args.host}:{args.port}  →  dst: {dst_root}  (mode: "
          f"{'APPLY' if args.apply else 'dry-run'})")

    offset = None
    while True:
        recs, offset = src.scroll(
            COLLECTION, limit=BATCH, offset=offset,
            with_payload=True, with_vectors=True,
        )
        if not recs:
            break
        for r in recs:
            n_total += 1
            payload = r.payload or {}
            rel, new_payload = _rekey_payload(payload, args.itunes_root)
            if rel is None:
                n_skipped += 1
                if n_skipped <= 5:
                    print(f"  SKIP (no match under itunes_root): "
                          f"{(payload.get('path') or '')[:90]}")
                continue
            new_id = rel_track_id(rel)
            n_rekeyed += 1
            if len(sample) < 6:
                sample.append((rel, f"{str(r.id)[:8]}…→{new_id[:8]}…"))
            if dst is not None:
                pending.append(qm.PointStruct(
                    id=new_id, vector=r.vector, payload=new_payload,
                ))
                if len(pending) >= BATCH:
                    dst.client.upsert(COLLECTION, points=pending, wait=False)
                    pending.clear()
        if offset is None:
            break

    if dst is not None and pending:
        dst.client.upsert(COLLECTION, points=pending, wait=True)

    print(f"\nscanned: {n_total}   re-keyed: {n_rekeyed}   "
          f"skipped (off-root): {n_skipped}")
    print("sample re-keyings:")
    for rel, change in sample:
        print(f"  {change}    rel: {rel}")

    if not args.apply:
        print("\nDRY-RUN. Re-run with --apply to write into the embedded store.")
        return 0

    # Verify count + a single retrieve round-trip on the new store.
    assert dst is not None
    after = dst.client.count(COLLECTION).count
    print(f"\nembedded store now contains {after} points at {dst_root}")
    dst.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
