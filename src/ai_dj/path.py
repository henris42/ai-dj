"""Path planning through the track embedding space.

Streaming planner for the live mix: `plan_next` returns one next track given
the current pivot, active styles, and an exclusion set (already-played + already-
queued). Fetches more neighbours than needed when styles are active, filters by
style, and samples with a temperature over cosine similarity so the same pivot
doesn't always pick the same successor."""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

import numpy as np
from qdrant_client.http import models as qm

from . import styles as stylesmod
from .index import COLLECTION, TrackIndex

logger = logging.getLogger(__name__)


@dataclass
class PlannedTrack:
    track_id: str
    path: str
    title: str | None
    artist: str | None
    album: str | None
    duration_s: float | None
    genre: str | None = None
    ai_genre: str | None = None


def planned_from_payload(track_id: str, payload: dict) -> PlannedTrack:
    return PlannedTrack(
        track_id=track_id,
        path=payload["path"],
        title=payload.get("title"),
        artist=payload.get("artist"),
        album=payload.get("album"),
        duration_s=payload.get("duration_s"),
        genre=payload.get("genre"),
        ai_genre=payload.get("ai_genre"),
    )


class Planner:
    def __init__(
        self,
        idx: TrackIndex,
        # Path planning is supposed to find *clever* relations through the
        # embedding — not the most obvious match each step. The closest
        # cosine-neighbour to a track is usually the same artist's next
        # song, a remix, an alternate mix. Skipping the top few and
        # sampling with moderate temperature picks the second/third order
        # of relatedness — tracks that are related but not the obvious
        # continuation, which is where the interesting drift comes from.
        temperature: float = 0.6,
        base_k: int = 18,
        expand_k: int = 100,
        skip_top: int = 3,
        seed: int | None = None,
    ):
        self.idx = idx
        self.client = idx.client
        self.temperature = temperature
        self.base_k = base_k
        self.expand_k = expand_k
        self.skip_top = skip_top
        # Default to a fresh time-based seed each instance so "New mix" gives
        # a different track on every launch even if the user's first action
        # is to immediately click it.
        self.rng = random.Random(seed if seed is not None else time.time_ns())
        self.active_styles: set[str] = set()
        self.restrict_artists: set[str] = set()

    def set_styles(self, active: set[str]) -> None:
        self.active_styles = set(active)

    def set_artists(self, artists: set[str]) -> None:
        """When non-empty, only tracks by these artists are considered."""
        self.restrict_artists = set(artists)

    def _query_filter(self) -> "qm.Filter | None":
        """Combined Qdrant filter for active styles + pinned artists. Both are
        hard constraints — when set, only matching tracks reach the candidate
        pool. Returns None when nothing is constrained."""
        must = []
        if self.restrict_artists:
            must.append(qm.FieldCondition(
                key="artist", match=qm.MatchAny(any=list(self.restrict_artists))
            ))
        if self.active_styles:
            must.append(qm.FieldCondition(
                key="ai_genre", match=qm.MatchAny(any=list(self.active_styles))
            ))
        if not must:
            return None
        return qm.Filter(must=must)

    def random_seed_track(self, prefer_styles: bool = True) -> PlannedTrack:  # noqa: ARG002
        """Pick a random seed track that matches the current style + artist
        filters.

        Implementation note: Qdrant's `scroll(offset=X)` treats X as a *point
        id* cursor, not a numeric offset. Passing a random integer to it (the
        previous strategy) didn't sample uniformly — it returned no records
        for any offset that wasn't a real point id, the empty-result fallback
        then re-rolled to "first record in scan order", and so the same seed
        track was returned every time. Now we collect the id list (filtered
        when relevant) and pick one in Python."""
        flt = self._query_filter()
        if flt is None:
            ids = self._all_ids()
        else:
            ids = self._ids_matching(flt)
        if not ids:
            raise RuntimeError("no tracks match current filters")
        track_id = self.rng.choice(ids)
        rec = self.client.retrieve(COLLECTION, [track_id], with_payload=True, with_vectors=False)
        if not rec:
            raise RuntimeError(f"could not retrieve track {track_id}")
        return planned_from_payload(track_id, rec[0].payload or {})

    def _all_ids(self) -> list[str]:
        cache = getattr(self, "_all_ids_cache", None)
        if cache is None:
            cache = self._scroll_ids(scroll_filter=None)
            self._all_ids_cache = cache
        return cache

    def _ids_matching(self, flt) -> list[str]:
        return self._scroll_ids(scroll_filter=flt)

    def _scroll_ids(self, scroll_filter) -> list[str]:
        ids: list[str] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                COLLECTION, limit=1024, offset=offset,
                scroll_filter=scroll_filter,
                with_payload=False, with_vectors=False,
            )
            ids.extend(str(r.id) for r in records)
            if offset is None:
                break
        return ids

    def _get_vector(self, track_id: str) -> np.ndarray | None:
        """Cached fetch of a single track's embedding."""
        cache = getattr(self, "_vec_cache", None)
        if cache is None:
            cache = {}
            self._vec_cache = cache
        if track_id in cache:
            return cache[track_id]
        rec = self.client.retrieve(COLLECTION, [track_id], with_payload=False, with_vectors=True)
        if not rec or rec[0].vector is None:
            return None
        v = np.asarray(rec[0].vector, dtype=np.float32)
        cache[track_id] = v
        return v

    def _build_query_vector(
        self, current_id: str, context_ids: tuple[str, ...]
    ) -> "str | list[float]":
        """Centroid of the current pivot + recent context, normalised. Falls
        back to the bare track id (Qdrant's own lookup) when there's no
        context — that path is the same as the original cosine query."""
        if not context_ids:
            return current_id
        cur_vec = self._get_vector(current_id)
        if cur_vec is None:
            return current_id
        # Most-recent context first; weights decay so the very recent history
        # nudges the centroid more than older tracks. `current` keeps a 1.0
        # weight that dominates the sum.
        recent = list(context_ids)[-3:][::-1]
        ctx_vecs: list[np.ndarray] = []
        ctx_w: list[float] = []
        decay = [0.4, 0.25, 0.15]
        for w, tid in zip(decay, recent):
            v = self._get_vector(tid)
            if v is not None:
                ctx_vecs.append(v)
                ctx_w.append(w)
        if not ctx_vecs:
            return current_id
        stacked = np.stack([cur_vec, *ctx_vecs])
        weights = np.asarray([1.0, *ctx_w], dtype=np.float32)
        centroid = (stacked * weights[:, None]).sum(axis=0) / weights.sum()
        n = float(np.linalg.norm(centroid))
        if n > 1e-9:
            centroid = centroid / n
        return centroid.astype(np.float32).tolist()

    def plan_next(
        self,
        current: PlannedTrack,
        exclude: set[str],
        avoid_artists: set[str] | None = None,
        context_ids: tuple[str, ...] = (),
    ) -> PlannedTrack | None:
        """Pick one next track. Returns None if no candidates remain.

        `avoid_artists` is a soft constraint: candidates with an artist in the
        set are filtered out, but if that empties the pool we fall back to the
        unfiltered candidates so the path never stalls.

        `context_ids` are the most recent tracks played *before* `current`
        (most-recent-last is fine; we weight by position). When supplied, the
        Qdrant query uses a centroid that's mostly `current` plus a soft tail
        from the recent history — so the next pick fits the recent vibe, not
        just the immediate pivot. `current` is weighted heavily so the path
        still drifts and doesn't get glued to the cluster the recent history
        sits in."""
        k = self.expand_k if (self.active_styles or self.restrict_artists) else self.base_k
        query_arg = self._build_query_vector(current.track_id, context_ids)
        hits = self.client.query_points(
            COLLECTION,
            query=query_arg,
            limit=k + len(exclude) + 1,
            with_payload=True,
            with_vectors=False,
            query_filter=self._query_filter(),
        ).points
        # Filter out excluded and the query point itself. Style + artist filters
        # were already applied at the Qdrant level, so every hit here is a hard
        # match — we don't need a Python-side genre check anymore.
        candidates = [h for h in hits if str(h.id) not in exclude and str(h.id) != current.track_id]
        if not candidates:
            return None

        # Artist-diversity filter remains a SOFT preference — we don't want it
        # to ever stall the path completely (only 1-2 artists pinned + diversity
        # avoidance can otherwise empty the pool).
        if avoid_artists:
            filtered = [h for h in candidates if (h.payload or {}).get("artist") not in avoid_artists]
            if filtered:
                candidates = filtered

        # Skip the most-obvious top matches (same artist's other songs,
        # remixes, etc.) — but only when there's enough pool left to keep
        # the path moving under tight filters.
        if len(candidates) > self.skip_top + 4:
            candidates = candidates[self.skip_top : self.skip_top + self.base_k]
        else:
            candidates = candidates[: self.base_k]
        scores = np.array([h.score for h in candidates], dtype=np.float32)
        logits = scores / max(self.temperature, 1e-6)
        logits -= logits.max()
        probs = np.exp(logits)
        probs = probs / probs.sum()
        pick = self.rng.choices(range(len(candidates)), weights=probs.tolist(), k=1)[0]
        chosen = candidates[pick]
        return planned_from_payload(str(chosen.id), chosen.payload or {})
