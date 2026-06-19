"""Embedding-driven discovery: surfaces only the vector index can produce.

Implemented:
  - outliers(): tracks that sit in sparse regions of the embedding space —
    their nearest neighbour is far away. The "hidden gems" of the library
    that random shuffle would never surface in a useful way.
  - twins(): pairs of tracks that are each other's nearest neighbour
    (mutual-NN). Natural pairings the planner would draw to anyway, but
    catalogued so the user can hand-pick interesting ones.

Both run on demand against the embedded Qdrant store; results are cached
to `<bundle>/db/discover_<kind>.json` so a re-open is instant. Pass
`refresh=True` to recompute (e.g. after Teach added new tracks).
"""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from . import config
from .index import COLLECTION, TrackIndex

logger = logging.getLogger(__name__)


def _cache_path(kind: str) -> Path:
    return config.db_root() / f"discover_{kind}.json"


def _load_cache(kind: str) -> dict | None:
    p = _cache_path(kind)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _save_cache(kind: str, data: dict) -> None:
    p = _cache_path(kind)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data), encoding="utf-8")
    tmp.replace(p)


def _sample_ids(idx: TrackIndex, k: int) -> list[str]:
    """Random sample of `k` track ids. We can't tell Qdrant 'random' so we
    pull a single scroll and pick from it."""
    ids: list[str] = []
    offset = None
    while True:
        recs, offset = idx.client.scroll(
            COLLECTION, limit=1024, offset=offset,
            with_payload=False, with_vectors=False,
        )
        ids.extend(str(r.id) for r in recs)
        if offset is None:
            break
    if len(ids) <= k:
        return ids
    return random.sample(ids, k)


def outliers(idx: TrackIndex, *, sample: int = 500, n: int = 40,
             refresh: bool = False) -> list[str]:
    """Return up to `n` track ids whose nearest non-self neighbour has the
    lowest cosine similarity (the most isolated tracks in the sample).

    `sample`: how many tracks to evaluate (one kNN query each — cost scales
    here, not with library size). 500 covers ~5% of a 10k library and
    surfaces the most isolated reliably.
    """
    if not refresh:
        cache = _load_cache("outliers")
        if cache and cache.get("ids"):
            return cache["ids"][:n]

    ids = _sample_ids(idx, sample)
    if not ids:
        return []

    scored: list[tuple[str, float]] = []
    for tid in ids:
        try:
            hits = idx.client.query_points(
                COLLECTION, query=tid, limit=2,
                with_payload=False, with_vectors=False,
            ).points
        except Exception:  # noqa: BLE001
            continue
        nbrs = [h for h in hits if str(h.id) != tid]
        if not nbrs:
            continue
        scored.append((tid, float(nbrs[0].score)))

    # Lowest score = most isolated.
    scored.sort(key=lambda x: x[1])
    out_ids = [tid for tid, _ in scored[:n]]
    _save_cache("outliers", {"ids": out_ids,
                              "scores": dict(scored[:n])})
    logger.info("outliers computed from %d samples → %d kept", len(ids), len(out_ids))
    return out_ids


def twins(idx: TrackIndex, *, sample: int = 500, n: int = 30,
          refresh: bool = False) -> list[tuple[str, str]]:
    """Mutual-nearest-neighbour pairs from a sample. Each returned pair
    (a, b) satisfies: a's nearest non-self ≈ b, AND b's nearest non-self
    ≈ a. Returns up to `n` such pairs, highest-affinity first."""
    if not refresh:
        cache = _load_cache("twins")
        if cache and cache.get("pairs"):
            return [tuple(p) for p in cache["pairs"][:n]]

    ids = _sample_ids(idx, sample)
    if not ids:
        return []

    nearest: dict[str, tuple[str, float]] = {}
    for tid in ids:
        try:
            hits = idx.client.query_points(
                COLLECTION, query=tid, limit=2,
                with_payload=False, with_vectors=False,
            ).points
        except Exception:  # noqa: BLE001
            continue
        nbrs = [h for h in hits if str(h.id) != tid]
        if nbrs:
            nearest[tid] = (str(nbrs[0].id), float(nbrs[0].score))

    seen: set[tuple[str, str]] = set()
    pairs: list[tuple[str, str, float]] = []
    for a, (b, score) in nearest.items():
        if b in nearest and nearest[b][0] == a:
            key = tuple(sorted((a, b)))
            if key in seen:
                continue
            seen.add(key)
            pairs.append((a, b, score))
    pairs.sort(key=lambda x: -x[2])      # highest affinity first

    out_pairs = [(a, b) for a, b, _ in pairs[:n]]
    _save_cache("twins", {"pairs": [list(p) for p in out_pairs]})
    logger.info("twins computed from %d samples → %d pairs", len(ids), len(out_pairs))
    return out_pairs
