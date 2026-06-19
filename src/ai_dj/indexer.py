"""In-app indexing pipeline: scan + embed + upsert, callable from the GUI.

The GUI's first-launch "Teach" button drives this. Pure-Python — no Qt
dependency — so it can be wrapped in any background worker (we wrap it in
a QRunnable in the SetupWindow).

Pipeline per source folder:

    1. Scan        — walk the tree, read tags, build TrackMeta (rel_path
                     identity).
    2. Resume      — drop ids already in the index. Re-runs are cheap.
    3. Embed       — load audio (librosa, mono 24 kHz, centred window) →
                     MERT forward pass on the chosen device (cuda / mps /
                     directml / cpu autodetect, CPU is the always-works
                     fallback) → 768-d vector.
    4. Upsert      — bulk-write (id, vector, payload) into Qdrant in
                     batches of `batch_size`. Vectors land first; the
                     planner picks them up immediately.

Heavy imports (torch, transformers, librosa) are deferred until embed time
so this module is import-cheap and a missing ML stack downgrades to a
clear error instead of crashing the GUI.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from . import scan
from .index import COLLECTION, TrackIndex

logger = logging.getLogger(__name__)


@dataclass
class IndexProgress:
    """Per-tick status handed to the GUI."""
    phase: str            # "scanning" | "embedding" | "done" | "error"
    current: int          # items completed in this phase
    total: int            # total for this phase (may be 0 during initial scan)
    file: str = ""        # rel_path of the most recently handled file
    message: str = ""     # extra detail (errors, phase notes)


@dataclass
class IndexResult:
    scanned: int = 0
    needs_embed: int = 0
    embedded: int = 0
    failed: int = 0
    skipped_existing: int = 0
    errors: list[str] = field(default_factory=list)


ProgressCallback = Callable[[IndexProgress], None]


def _payload_for(meta) -> dict:
    return {
        "rel_path": meta.rel_path,
        "title": meta.title,
        "artist": meta.artist,
        "album": meta.album,
        "duration_s": meta.duration_s,
        "genre": meta.genre,
    }


def _pick_device() -> str:
    """CPU is the always-works fallback. Try MPS (Apple Silicon UMA) → CUDA
    (NVIDIA) → DirectML (AMD on Windows) → CPU."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    try:
        import torch_directml  # noqa: F401
        return "directml"
    except ImportError:
        pass
    return "cpu"


def index_sources(
    sources: Iterable,
    idx: TrackIndex,
    *,
    progress: ProgressCallback | None = None,
    batch_size: int = 16,
) -> IndexResult:
    """Full pipeline driven by `sources` (each has `.path` + `.name`).

    Embedding cost dominates wall-clock; the scan phase is fast. We report
    `phase` so the GUI can swap labels (scan-progress is bytes-cheap; the
    user wants to know "we're learning" once embed starts).
    """
    out = IndexResult()

    # --- 1. Scan + resume filter ---
    metas = []
    src_roots: dict[str, Path] = {}        # rel_path → its source root, for embed-time absolute resolution
    for src in sources:
        root = Path(src.path)
        if not root.is_dir():
            out.errors.append(f"source missing: {root}")
            continue
        for meta in scan.iter_library(root):
            metas.append(meta)
            src_roots[meta.rel_path] = root
            if progress and len(metas) % 200 == 0:
                progress(IndexProgress("scanning", len(metas), 0, meta.rel_path))
    out.scanned = len(metas)
    if progress:
        progress(IndexProgress("scanning", out.scanned, out.scanned, ""))

    if not metas:
        if progress:
            progress(IndexProgress("done", 1, 1, "", "no audio files found"))
        return out

    existing = idx.existing_ids([m.track_id for m in metas])
    todo = [m for m in metas if m.track_id not in existing]
    out.skipped_existing = len(metas) - len(todo)
    out.needs_embed = len(todo)
    if not todo:
        if progress:
            progress(IndexProgress("done", 1, 1, "",
                                   f"library already indexed ({out.skipped_existing} tracks)"))
        return out

    # --- 2. Heavy imports for embedding ---
    try:
        from . import embed as embed_mod
        import numpy as np
        from qdrant_client.http import models as qm
    except ImportError as e:
        msg = f"ML stack not installed (need torch + transformers + librosa): {e}"
        out.errors.append(msg)
        out.failed = len(todo)
        if progress:
            progress(IndexProgress("error", 0, len(todo), "", msg))
        return out

    device = _pick_device()
    try:
        embedder = embed_mod.Embedder(device=device)
    except Exception as e:  # noqa: BLE001
        msg = f"failed to load MERT model on {device}: {type(e).__name__}: {e}"
        out.errors.append(msg)
        out.failed = len(todo)
        if progress:
            progress(IndexProgress("error", 0, len(todo), "", msg))
        return out
    logger.info("embedder loaded on %s", device)

    # --- 3. Embed + upsert in batches ---
    pending_audios = []
    pending_metas = []

    def flush():
        nonlocal pending_audios, pending_metas
        if not pending_metas:
            return
        try:
            audios = np.stack(pending_audios)
            vectors = embedder.embed_batch(audios)
            points = [
                qm.PointStruct(
                    id=m.track_id,
                    vector=(v.tolist() if hasattr(v, "tolist") else list(v)),
                    payload=_payload_for(m),
                )
                for m, v in zip(pending_metas, vectors)
            ]
            idx.client.upsert(COLLECTION, points=points, wait=False)
            out.embedded += len(pending_metas)
        except Exception as e:  # noqa: BLE001
            out.failed += len(pending_metas)
            out.errors.append(f"batch failed: {type(e).__name__}: {e}")
            logger.exception("embed batch failed")
        pending_audios = []
        pending_metas = []

    n = len(todo)
    for i, meta in enumerate(todo, 1):
        try:
            root = src_roots[meta.rel_path]
            audio = embed_mod.load_audio(str(root / meta.rel_path), meta.duration_s)
            pending_audios.append(audio)
            pending_metas.append(meta)
        except Exception as e:  # noqa: BLE001
            out.failed += 1
            out.errors.append(f"{meta.rel_path}: {type(e).__name__}: {e}")
        if len(pending_metas) >= batch_size or i == n:
            flush()
        if progress:
            progress(IndexProgress("embedding", i, n, meta.rel_path))

    if progress:
        progress(IndexProgress("done", 1, 1, "",
                               f"{out.embedded} embedded, {out.failed} failed, "
                               f"{out.skipped_existing} skipped"))
    return out
