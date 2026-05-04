"""Discogs-Effnet genre-tagging pass over the embedded library.

Replaces the CLAP zero-shot tagger (`tag_library.py`). Iterates Qdrant points
that haven't been Discogs-tagged yet, loads 30 s of 16 kHz mono from the stored
audio path, runs the EffnetDiscogs backbone + 400-class genre head, aggregates
to the high-level steering styles, and writes payload back. Re-runnable; uses
`ai_tagger == "discogs400"` to skip already-done tracks unless `--all` forces."""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path

from tqdm import tqdm

from ai_dj import discogs_tag
from ai_dj.index import COLLECTION, TrackIndex

ERRORS_LOG = Path("data/tag_errors_discogs.log")
TAGGER_NAME = "discogs400"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4,
                    help="audio prefetch worker threads (CPU-bound TF inference, "
                         "so the bottleneck isn't audio loading)")
    ap.add_argument("--batch-size", type=int, default=4,
                    help="Essentia processes one clip at a time internally; "
                         "we still batch on the prefetch side for parity with "
                         "tag_library.py")
    ap.add_argument("--all", action="store_true",
                    help="re-tag every track even if it's already tagged by "
                         "this tagger (for re-runs after a label-mapping change)")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        stream=sys.stderr,
    )
    log = logging.getLogger("tag_library_discogs")

    idx = TrackIndex()
    idx.ensure_collection()
    log.info("points in Qdrant: %d", idx.count())

    t0 = time.time()
    if args.all:
        todo_ids = idx.all_ids()
    else:
        todo_ids = idx.ids_not_tagged_by(TAGGER_NAME)
    log.info("tracks to (re)tag: %d (scanned in %.1fs)", len(todo_ids), time.time() - t0)
    if args.limit:
        todo_ids = todo_ids[: args.limit]
        log.info("--limit: processing first %d", len(todo_ids))
    if not todo_ids:
        return

    id_to_meta: dict[str, tuple[str, float | None]] = {}
    for i in range(0, len(todo_ids), 500):
        chunk = todo_ids[i : i + 500]
        recs = idx.client.retrieve(COLLECTION, chunk, with_payload=True, with_vectors=False)
        for r in recs:
            pl = r.payload or {}
            id_to_meta[str(r.id)] = (pl.get("path"), pl.get("duration_s"))

    tagger = discogs_tag.DiscogsTagger()
    ERRORS_LOG.parent.mkdir(parents=True, exist_ok=True)
    errors_f = ERRORS_LOG.open("a")

    done = failed = 0
    pbar = tqdm(total=len(todo_ids), desc="tag", unit="track")

    pending_audio: list = []
    pending_ids: list[str] = []

    def flush_batch():
        nonlocal done, failed
        if not pending_audio:
            return
        try:
            results = tagger.tag_batch(pending_audio)
        except Exception as e:  # noqa: BLE001
            for tid in pending_ids:
                errors_f.write(f"{tid}\tDISCOGS:{type(e).__name__}\t{e}\n")
            errors_f.flush()
            failed += len(pending_ids)
            pbar.update(len(pending_ids))
            pending_audio.clear()
            pending_ids.clear()
            return
        updates = [(tid, top, score, top3) for tid, (top, score, top3) in zip(pending_ids, results)]
        idx.set_ai_genres(updates, tagger=TAGGER_NAME)
        done += len(updates)
        pbar.update(len(updates))
        pending_audio.clear()
        pending_ids.clear()

    max_inflight = max(args.workers * 2, args.batch_size + 4)
    ids_iter = iter(todo_ids)
    inflight: dict = {}

    with ThreadPoolExecutor(max_workers=args.workers, thread_name_prefix="audio") as pool:
        def _submit_next():
            while True:
                try:
                    tid = next(ids_iter)
                except StopIteration:
                    return False
                meta = id_to_meta.get(tid)
                if not meta or not meta[0]:
                    failed_skip(tid, "no path")
                    continue
                path, dur = meta
                fut = pool.submit(discogs_tag.load_audio_16k, path, dur)
                inflight[fut] = tid
                return True

        def failed_skip(tid: str, reason: str) -> None:
            nonlocal failed
            failed += 1
            errors_f.write(f"{tid}\tSKIP\t{reason}\n")
            errors_f.flush()
            pbar.update(1)

        for _ in range(max_inflight):
            if not _submit_next():
                break

        while inflight:
            finished, _ = wait(list(inflight.keys()), return_when=FIRST_COMPLETED)
            for fut in finished:
                tid = inflight.pop(fut)
                _submit_next()
                try:
                    y = fut.result()
                except Exception as e:  # noqa: BLE001
                    errors_f.write(f"{tid}\tLOAD:{type(e).__name__}\t{e}\n")
                    errors_f.flush()
                    failed += 1
                    pbar.update(1)
                    continue
                pending_audio.append(y)
                pending_ids.append(tid)
                if len(pending_audio) >= args.batch_size:
                    flush_batch()

        flush_batch()

    pbar.close()
    errors_f.close()
    log.info("done=%d failed=%d  (errors -> %s)", done, failed, ERRORS_LOG)


if __name__ == "__main__":
    main()
