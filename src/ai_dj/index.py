"""Qdrant-backed track index for AI DJ. Collection: tracks, vector: MERT (768-d cosine)."""
from __future__ import annotations

import logging

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

logger = logging.getLogger(__name__)

COLLECTION = "tracks"
DIM = 768


class TrackIndex:
    def __init__(self, host: str = "127.0.0.1", port: int = 6333):
        self.client = QdrantClient(host=host, port=port)

    def ensure_collection(self) -> None:
        if self.client.collection_exists(COLLECTION):
            return
        self.client.create_collection(
            collection_name=COLLECTION,
            vectors_config=qm.VectorParams(size=DIM, distance=qm.Distance.COSINE),
        )
        for field in ("artist", "album", "genre"):
            self.client.create_payload_index(
                COLLECTION, field_name=field, field_schema=qm.PayloadSchemaType.KEYWORD
            )
        logger.info("created collection %s", COLLECTION)

    def contains(self, track_id: str) -> bool:
        return bool(self.client.retrieve(COLLECTION, [track_id], with_payload=False, with_vectors=False))

    def existing_ids(self, track_ids: list[str], batch: int = 500) -> set[str]:
        """Return the subset of track_ids that are already in the collection."""
        found: set[str] = set()
        for i in range(0, len(track_ids), batch):
            chunk = track_ids[i : i + batch]
            records = self.client.retrieve(COLLECTION, chunk, with_payload=False, with_vectors=False)
            found.update(str(r.id) for r in records)
        return found

    def upsert(self, track_id: str, vector: np.ndarray, payload: dict) -> None:
        self.client.upsert(
            COLLECTION,
            points=[qm.PointStruct(id=track_id, vector=vector.tolist(), payload=payload)],
            wait=False,
        )

    def upsert_many(self, points: list[tuple[str, np.ndarray, dict]]) -> None:
        """Bulk upsert a list of (id, vector, payload) tuples in a single RPC."""
        if not points:
            return
        self.client.upsert(
            COLLECTION,
            points=[
                qm.PointStruct(id=pid, vector=v.tolist(), payload=pl)
                for pid, v, pl in points
            ],
            wait=False,
        )

    def count(self) -> int:
        return self.client.count(COLLECTION).count

    def set_ai_genres(
        self,
        updates: list[tuple[str, str, float, list[tuple[str, float]]]],
        tagger: str | None = None,
    ) -> None:
        """Attach AI-derived genre tags to existing points. Each update is
        (track_id, top_style, top_score, top3_ranked).

        `tagger` (e.g. "discogs400") stamps which model produced the tag, so
        re-runs after a tagger upgrade can identify stale entries."""
        if not updates:
            return
        for track_id, top_style, top_score, top3 in updates:
            payload = {
                "ai_genre": top_style,
                "ai_genre_score": float(top_score),
                "ai_genre_top3": [(s, float(sc)) for s, sc in top3],
            }
            if tagger:
                payload["ai_tagger"] = tagger
            self.client.set_payload(COLLECTION, payload=payload, points=[track_id], wait=False)

    def all_ids(self) -> list[str]:
        """Every track id in the collection, for tagger re-runs."""
        ids: list[str] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                COLLECTION, limit=1024, offset=offset,
                with_payload=False, with_vectors=False,
            )
            ids.extend(str(r.id) for r in records)
            if offset is None:
                break
        return ids

    def ids_not_tagged_by(self, tagger: str) -> list[str]:
        """Track ids whose `ai_tagger` payload entry isn't `tagger` (or is
        absent). Used to skip already-processed tracks on tagger re-runs."""
        ids: list[str] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                COLLECTION, limit=1024, offset=offset,
                with_payload=["ai_tagger"], with_vectors=False,
            )
            for r in records:
                if (r.payload or {}).get("ai_tagger") != tagger:
                    ids.append(str(r.id))
            if offset is None:
                break
        return ids

    def ids_missing_ai_genre(self) -> list[str]:
        """Return track_ids that don't yet have an `ai_genre` payload entry."""
        ids: list[str] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                COLLECTION, limit=1024, offset=offset,
                with_payload=["ai_genre"], with_vectors=False,
            )
            for r in records:
                if not (r.payload or {}).get("ai_genre"):
                    ids.append(str(r.id))
            if offset is None:
                break
        return ids

    def all_track_summaries(self) -> list[tuple[str, str, str, str, str]]:
        """Return (track_id, artist, title, album, path) for every indexed track.
        Used for autocomplete + library browser; ~11k entries is ~1.5 MB."""
        out: list[tuple[str, str, str, str, str]] = []
        offset = None
        while True:
            records, offset = self.client.scroll(
                COLLECTION, limit=512, offset=offset,
                with_payload=["artist", "title", "album", "path"], with_vectors=False,
            )
            for r in records:
                p = r.payload or {}
                out.append((
                    str(r.id),
                    p.get("artist") or "",
                    p.get("title") or "",
                    p.get("album") or "",
                    p.get("path") or "",
                ))
            if offset is None:
                break
        return out
