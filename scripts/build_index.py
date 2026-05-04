"""Full scan + embed of the music library into Qdrant.

Pipeline: N worker threads prefetch audio from disk (IO + ffmpeg decode release the
GIL) and return fixed-length arrays. The main thread collects them into a batch of
`--batch-size` tracks, runs a single GPU forward pass, and bulk-upserts to Qdrant.
Resumable via `TrackIndex.existing_ids` pre-filter."""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

# Enable AOTriton Flash/Mem-efficient attention on AMD. Without it, SDPA falls
# back to a path that hangs indefinitely on gfx1101 + long sequences + batched input.
os.environ.setdefault("TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL", "1")

from tqdm import tqdm

from ai_dj import embed, index, scan

DEFAULT_LIBRARY = Path("/mnt/e/Music/iTunes/iTunes Media/Music")
ERRORS_LOG = Path("data/embed_errors.log")


def _payload(meta):
    return {
        "path": meta.path,
        "title": meta.title,
        "artist": meta.artist,
        "album": meta.album,
        "duration_s": meta.duration_s,
        "genre": meta.genre,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--library", type=Path, default=DEFAULT_LIBRARY)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sample-seconds", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=6, help="audio prefetch worker threads")
    ap.add_argument("--batch-size", type=int, default=16, help="tracks per GPU forward pass")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("build_index")

    idx = index.TrackIndex()
    idx.ensure_collection()
    log.info("existing points in Qdrant: %d", idx.count())

    log.info("scanning %s ...", args.library)
    t0 = time.time()
    tracks = list(scan.iter_library(args.library))
    log.info("found %d audio files in %.1fs", len(tracks), time.time() - t0)

    if args.limit:
        tracks = tracks[: args.limit]
        log.info("--limit: processing first %d", len(tracks))

    t0 = time.time()
    existing = idx.existing_ids([m.track_id for m in tracks])
    todo = [m for m in tracks if m.track_id not in existing]
    log.info(
        "resume filter in %.1fs: todo=%d skipping=%d",
        time.time() - t0,
        len(todo),
        len(existing),
    )
    if not todo:
        return

    emb = embed.Embedder()
    ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
    errors_f = ERRORS_LOG.open("a")

    done = failed = 0
    pbar = tqdm(total=len(todo), desc="embed", unit="track")

    pending_audio: list = []
    pending_meta: list = []

    def flush_batch():
        nonlocal done, failed
        if not pending_audio:
            return
        try:
            vecs = emb.embed_batch(pending_audio)
        except Exception as e:  # noqa: BLE001
            for meta in pending_meta:
                errors_f.write(f"{meta.path}\tEMBED:{type(e).__name__}\t{e}\n")
            errors_f.flush()
            failed += len(pending_meta)
            pbar.update(len(pending_meta))
            pending_audio.clear()
            pending_meta.clear()
            return

        idx.upsert_many(
            [(m.track_id, v, _payload(m)) for m, v in zip(pending_meta, vecs)]
        )
        done += len(pending_meta)
        pbar.update(len(pending_meta))
        pending_audio.clear()
        pending_meta.clear()

    max_inflight = max(args.workers * 2, args.batch_size + 4)
    todo_iter = iter(todo)
    inflight: dict = {}

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="audio") as pool:
        def _submit_next():
            try:
                meta = next(todo_iter)
            except StopIteration:
                return False
            fut = pool.submit(embed.load_audio, meta.path, meta.duration_s, args.sample_seconds)
            inflight[fut] = meta
            return True

        for _ in range(max_inflight):
            if not _submit_next():
                break

        while inflight:
            finished, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
            for fut in finished:
                meta = inflight.pop(fut)
                _submit_next()
                try:
                    y = fut.result()
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    errors_f.write(f"{meta.path}\tLOAD:{type(e).__name__}\t{e}\n")
                    errors_f.flush()
                    pbar.update(1)
                    continue
                pending_audio.append(y)
                pending_meta.append(meta)
                if len(pending_audio) >= args.batch_size:
                    flush_batch()

        flush_batch()

    pbar.close()
    errors_f.close()
    log.info("done=%d failed=%d  (errors -> %s)", done, failed, ERRORS_LOG)
    log.info("total points now: %d", idx.count())


if __name__ == "__main__":
    main()
