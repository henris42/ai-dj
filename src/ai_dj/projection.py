"""2D projection of the embedding space for visualisation.

UMAP on cosine distance — produces genuine clusters where PCA just flattens the
variance. Also stashes each point's `ai_genre` alongside its xy so the GUI can
color dots by genre without a second Qdrant round-trip."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from .index import COLLECTION, TrackIndex

logger = logging.getLogger(__name__)

CACHE_PATH = Path("data/projection.npz")


@dataclass
class Projection:
    track_ids: np.ndarray  # (N,) str
    xyz: np.ndarray         # (N, 3) float32 — 3D UMAP coordinates
    genres: np.ndarray      # (N,) str — ai_genre or '' when untagged
    scores: np.ndarray      # (N,) float32 — ai_genre_score (CLAP confidence), 0 when missing

    @property
    def xy(self) -> np.ndarray:
        """Backwards-compat 2D view (drops the z axis) for any legacy 2D code."""
        return self.xyz[:, :2]

    def save(self, path: Path = CACHE_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, track_ids=self.track_ids, xyz=self.xyz,
                 genres=self.genres, scores=self.scores)

    @classmethod
    def load(cls, path: Path = CACHE_PATH) -> "Projection":
        d = np.load(path, allow_pickle=False)
        if "xyz" in d.files:
            xyz = d["xyz"]
        else:
            xy = d["xy"]
            xyz = np.concatenate([xy, np.zeros((len(xy), 1), dtype=xy.dtype)], axis=1)
        scores = d["scores"] if "scores" in d.files else np.zeros(len(xyz), dtype=np.float32)
        return cls(track_ids=d["track_ids"], xyz=xyz, genres=d["genres"], scores=scores)


def _iter_all_points(idx: TrackIndex, batch: int = 512) -> Iterator[tuple[str, list[float], str, float]]:
    offset = None
    while True:
        records, offset = idx.client.scroll(
            COLLECTION,
            limit=batch,
            offset=offset,
            with_payload=["ai_genre", "ai_genre_score"],
            with_vectors=True,
        )
        for r in records:
            payload = r.payload or {}
            yield (
                str(r.id),
                r.vector,
                payload.get("ai_genre") or "",
                float(payload.get("ai_genre_score") or 0.0),
            )
        if offset is None:
            break


def fit(idx: TrackIndex, n_neighbors: int = 15, min_dist: float = 0.7, seed: int = 42,
        n_components: int = 3) -> Projection:
    """Fit a 3D UMAP over every indexed track (or 2D if requested)."""
    ids: list[str] = []
    vecs: list[list[float]] = []
    genres: list[str] = []
    scores: list[float] = []
    for tid, v, g, s in _iter_all_points(idx):
        ids.append(tid)
        vecs.append(v)
        genres.append(g)
        scores.append(s)
    if not ids:
        raise RuntimeError("no points in collection")

    X = np.asarray(vecs, dtype=np.float32)
    logger.info("fitting %dD UMAP on %d points, dim=%d ...",
                n_components, X.shape[0], X.shape[1])
    from umap import UMAP
    reducer = UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        verbose=False,
    )
    out = reducer.fit_transform(X).astype(np.float32)
    if out.shape[1] == 2:
        out = np.concatenate([out, np.zeros((len(out), 1), dtype=np.float32)], axis=1)
    # Centre the cloud at origin and scale so the largest extent is roughly 12
    # units — gives the camera more room to manoeuvre and makes the terrain
    # heightmap variations actually visible.
    out -= out.mean(axis=0, keepdims=True)
    span = float(np.max(np.linalg.norm(out, axis=1)))
    if span > 0:
        out *= (12.0 / span)
    logger.info("UMAP done: extent per axis = %s",
                [(float(out[:, i].min()), float(out[:, i].max())) for i in range(out.shape[1])])

    return Projection(
        track_ids=np.asarray(ids),
        xyz=out,
        genres=np.asarray(genres),
        scores=np.asarray(scores, dtype=np.float32),
    )
