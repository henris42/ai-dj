#!/usr/bin/env python3
"""Recover metadata for untagged audio via AcoustID acoustic fingerprinting.

For the older files with no usable ID3, "tags are definitive" can't help —
there are none. This identifies them by their *sound*: Chromaprint fingerprint
(`fpcalc`) → AcoustID web service → MusicBrainz recording (artist/title/album
+ a match score). It only accepts confident matches; uncertain ones are
listed for manual handling, never guessed.

What it does with a match is a runtime choice, not a code fork:

  --mode sidecar  (default, SAFE)  write recovered fields to a JSON sidecar
                                   (data/recovered_tags.json). The 99%-full
                                   E: drive is never written. ID3 stays the
                                   in-file source; the sidecar is an overlay
                                   the indexer can read.
  --mode id3      (opt-in, RISKY)  write the tags into the MP3s themselves.
                                   Rewrites files → needs transient free
                                   space. Refuses unless --force given.

SAFE BY DEFAULT: dry-run unless --apply. Resumable: already-resolved files
(present in the results TSV) are skipped on re-run. Throttled to AcoustID's
~3 req/s policy.

    AIDJ_ACOUSTID_KEY=... uv run python scripts/identify_untagged.py \\
        [MUSIC_ROOT] [--from-list data/no_id3.txt] [--apply]
        [--mode sidecar|id3] [--min-score 0.85]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from ai_dj.scan import AUDIO_EXTS, DRM_EXTS, VIDEO_EXTS, _first  # noqa: E402

import mutagen  # noqa: E402

try:
    import acoustid
except ImportError:
    sys.exit("pyacoustid not installed (uv pip install pyacoustid).")

# This WSL Python's system CA bundle is broken (raw TLS verify fails). Point
# every HTTP path — requests *and* urllib — at certifi's bundle so the
# AcoustID call can't fail for cert reasons regardless of pyacoustid's backend.
try:
    import certifi
    os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
except ImportError:
    pass

RESULTS_TSV = Path("data/identify_results.tsv")
SIDECAR = Path("data/recovered_tags.json")
THROTTLE_S = 0.34  # AcoustID: keep under ~3 requests/second


def needs_id(path: Path) -> bool:
    """A recognition candidate = no usable title from existing tags."""
    try:
        tags = mutagen.File(path, easy=True)
    except Exception:  # noqa: BLE001
        return True
    if not tags:
        return True
    title = _first(dict(tags).get("title"))
    return not (title and str(title).strip())


def gather(root: Path, from_list: str | None) -> list[Path]:
    if from_list:
        out = []
        for line in Path(from_list).read_text(encoding="utf-8").splitlines():
            p = Path(line.split("\t")[0].strip())
            if p.is_file():
                out.append(p)
        return out
    files = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        ext = p.suffix.lower()
        if ext in VIDEO_EXTS or ext in DRM_EXTS or ext not in AUDIO_EXTS:
            continue
        if needs_id(p):
            files.append(p)
    return files


def already_done() -> set[str]:
    if not RESULTS_TSV.exists():
        return set()
    return {ln.split("\t", 1)[0]
            for ln in RESULTS_TSV.read_text(encoding="utf-8").splitlines() if ln}


def write_id3(path: Path, artist: str, title: str, album: str | None) -> None:
    tags = mutagen.File(path, easy=True)
    if tags is None:
        raise RuntimeError("mutagen cannot open for writing")
    tags["artist"] = artist
    tags["title"] = title
    if album:
        tags["album"] = album
    tags.save()


def main() -> int:
    ap = argparse.ArgumentParser(description="AcoustID recognition for untagged audio.")
    ap.add_argument("root", nargs="?", help="Music root (else $AIDJ_MUSIC / $AIDJ_BUNDLE/Music)")
    ap.add_argument("--from-list", help="file of paths to identify (e.g. data/no_id3.txt)")
    ap.add_argument("--apply", action="store_true", help="actually act on matches (default: dry-run)")
    ap.add_argument("--mode", choices=("sidecar", "id3"), default="sidecar")
    ap.add_argument("--min-score", type=float, default=0.85, help="min AcoustID match score [0-1]")
    ap.add_argument("--force", action="store_true", help="allow --mode id3 (writes into files)")
    ap.add_argument("--limit", type=int, default=0, help="stop after N lookups (0 = all)")
    args = ap.parse_args()

    key = os.environ.get("AIDJ_ACOUSTID_KEY")
    if not key:
        sys.exit("Set AIDJ_ACOUSTID_KEY (your AcoustID API key).")
    if args.apply and args.mode == "id3" and not args.force:
        sys.exit("--mode id3 writes into the files (risky on the full E: drive). "
                 "Re-run with --force if you really mean it.")

    if args.from_list:
        root = None
    elif args.root:
        root = Path(args.root)
    elif os.environ.get("AIDJ_MUSIC"):
        root = Path(os.environ["AIDJ_MUSIC"])
    elif os.environ.get("AIDJ_BUNDLE"):
        root = Path(os.environ["AIDJ_BUNDLE"]) / "Music"
    else:
        sys.exit("Give --from-list or a MUSIC_ROOT / $AIDJ_MUSIC / $AIDJ_BUNDLE.")
    if root is not None:
        root = root.resolve()

    files = gather(root, args.from_list)
    done = already_done()
    pending = [f for f in files if str(f) not in done]
    resolved = len(files) - len(pending)
    todo = pending[: args.limit] if args.limit else pending
    cap = f", capped at {len(todo)} this run" if args.limit and len(todo) < len(pending) else ""
    print(f"{len(files)} candidates, {resolved} already resolved, "
          f"{len(pending)} pending{cap}.")

    sidecar = {}
    if SIDECAR.exists():
        sidecar = json.loads(SIDECAR.read_text(encoding="utf-8"))

    RESULTS_TSV.parent.mkdir(parents=True, exist_ok=True)
    matched = low = err = 0
    with open(RESULTS_TSV, "a", encoding="utf-8") as res:
        for i, f in enumerate(todo, 1):
            try:
                best = None
                for score, _rid, title, artist in acoustid.match(key, str(f)):
                    if best is None or score > best[0]:
                        best = (score, artist, title)
            except acoustid.NoBackendError:
                sys.exit("fpcalc not found — run: sudo apt-get install -y libchromaprint-tools")
            except acoustid.FingerprintGenerationError as e:
                err += 1
                print(f"  [{i}/{len(todo)}] FP-FAIL {f.name}: {e}", file=sys.stderr)
                time.sleep(THROTTLE_S)
                continue
            except acoustid.WebServiceError as e:
                # Surface AcoustID's real code/message (e.g. "invalid API
                # key" / rate limit) instead of a bare "status: error".
                msg = getattr(e, "message", None) or str(e)
                err += 1
                print(f"  [{i}/{len(todo)}] API-ERR {f.name}: {msg}", file=sys.stderr)
                if "api key" in msg.lower():
                    sys.exit("AcoustID rejects the key — fix AIDJ_ACOUSTID_KEY and rerun.")
                time.sleep(THROTTLE_S)
                continue
            except Exception as e:  # noqa: BLE001
                err += 1
                print(f"  [{i}/{len(todo)}] ERR {f.name}: {type(e).__name__}: {e}",
                      file=sys.stderr)
                time.sleep(THROTTLE_S)
                continue

            if not best or best[0] < args.min_score or not (best[1] and best[2]):
                low += 1
                sc = f"{best[0]:.2f}" if best else "none"
                print(f"  [{i}/{len(todo)}] LOW  {f.name}  (score {sc}) — manual")
                res.write(f"{f}\tLOW\t{sc}\t\t\n")
                time.sleep(THROTTLE_S)
                continue

            score, artist, title = best
            matched += 1
            print(f"  [{i}/{len(todo)}] OK {score:.2f}  {f.name}  ->  {artist} — {title}")
            res.write(f"{f}\tOK\t{score:.2f}\t{artist}\t{title}\n")

            if args.apply:
                if args.mode == "sidecar":
                    sidecar[str(f)] = {"artist": artist, "title": title, "album": None,
                                       "source": "acoustid", "score": round(score, 3)}
                else:  # id3
                    try:
                        write_id3(f, artist, title, None)
                    except Exception as e:  # noqa: BLE001
                        err += 1
                        print(f"      ID3 write failed: {e}", file=sys.stderr)
            time.sleep(THROTTLE_S)

    if args.apply and args.mode == "sidecar":
        SIDECAR.write_text(json.dumps(sidecar, indent=2, ensure_ascii=False),
                           encoding="utf-8")

    print(f"\nmatched: {matched}   low/uncertain: {low}   errors: {err}")
    print(f"results log: {RESULTS_TSV}")
    if args.apply and args.mode == "sidecar":
        print(f"recovered metadata (safe overlay): {SIDECAR}")
    elif not args.apply:
        print("DRY-RUN — no tags written. Add --apply (and pick --mode) to act.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
