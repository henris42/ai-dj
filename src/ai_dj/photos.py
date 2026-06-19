"""Public artist-photo fetching, factored so both the batch script and the
GUI can use it.

Sources (free, no API key):
  - Wikipedia REST: page summary (infobox image) + media-list (every image
    on the article — live shots, lineups, etc.)
  - Wikimedia Commons: full-text image search, biased to titles that
    actually mention the artist name to avoid generic-noun noise.

Polite UA + ~10 rps throttle. Photos resize on arrival (long edge capped at
`max_edge`, JPEG q85, EXIF stripped). Safe to call concurrently with the
running GUI / player — the only writes are JPEG files under PHOTO_ROOT.
"""
from __future__ import annotations

import io
import json
import os
import re
import ssl
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import certifi
from PIL import Image, ImageOps

UA = "ai-dj/0.1 (personal music player; hs@stor.ax)"
THROTTLE_S = 0.12
_SSL = ssl.create_default_context(cafile=certifi.where())
_ILLEGAL = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_IMG_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def safe(s: str) -> str:
    return _ILLEGAL.sub("", s).strip(". ").strip()[:120]


def photo_root() -> Path:
    """Default photo bundle: D:\\ai-dj\\Photos natively on Windows,
    /mnt/d/ai-dj/Photos under WSL — both addresses of the same folder.
    Overridable via AIDJ_PHOTOS / PHOTO_ROOT."""
    p = os.environ.get("AIDJ_PHOTOS") or os.environ.get("PHOTO_ROOT")
    if p:
        return Path(p)
    import sys
    return Path(r"D:\ai-dj\Photos") if sys.platform == "win32" else Path("/mnt/d/ai-dj/Photos")


def target_photos(n_tracks: int) -> int:
    """Scaled photo budget per artist (tiered by track count)."""
    if n_tracks <= 1:
        return 1
    if n_tracks <= 5:
        return 2
    if n_tracks <= 20:
        return 3
    if n_tracks <= 50:
        return 5
    return 8


def already_have(artist_dir: Path, target: int) -> bool:
    if not artist_dir.is_dir():
        return False
    have = sum(1 for p in artist_dir.iterdir()
               if p.is_file() and p.suffix.lower() in _IMG_EXTS)
    return have >= target


# ---- HTTP --------------------------------------------------------------
def _get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urllib.request.urlopen(req, timeout=timeout, context=_SSL) as r:
        return r.read()


# ---- Wikipedia ---------------------------------------------------------
def _wiki_summary(title: str) -> dict | None:
    url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + urllib.parse.quote(title, safe="")
    try:
        return json.loads(_get(url))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return None
        raise


def _wiki_media_list(title: str) -> list[str]:
    url = ("https://en.wikipedia.org/api/rest_v1/page/media-list/"
           + urllib.parse.quote(title, safe=""))
    try:
        data = json.loads(_get(url))
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return []
        raise
    out: list[str] = []
    for item in data.get("items", []):
        if item.get("type") != "image":
            continue
        srcset = item.get("srcset") or []
        if not srcset:
            continue
        src = srcset[0].get("src", "")
        if src.startswith("//"):
            src = "https:" + src
        if src:
            out.append(src)
    return out


# ---- Wikimedia Commons -------------------------------------------------
def _commons_search_files(query: str, limit: int = 40) -> list[str]:
    url = ("https://commons.wikimedia.org/w/api.php?"
           + urllib.parse.urlencode({
               "action": "query", "format": "json", "list": "search",
               "srsearch": query, "srnamespace": 6, "srlimit": limit,
           }))
    data = json.loads(_get(url))
    return [item.get("title", "")
            for item in data.get("query", {}).get("search", [])
            if item.get("title", "").lower().endswith(
                (".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff"))]


def _commons_resolve_urls(titles: list[str]) -> list[str]:
    if not titles:
        return []
    out: list[str] = []
    for i in range(0, len(titles), 40):
        chunk = titles[i:i + 40]
        url = ("https://commons.wikimedia.org/w/api.php?"
               + urllib.parse.urlencode({
                   "action": "query", "format": "json", "prop": "imageinfo",
                   "iiprop": "url", "titles": "|".join(chunk),
               }))
        data = json.loads(_get(url))
        by_title = {p.get("title"): p
                    for p in data.get("query", {}).get("pages", {}).values()}
        for t in chunk:
            p = by_title.get(t)
            if not p:
                continue
            ii = (p.get("imageinfo") or [{}])[0]
            u = ii.get("url")
            if u:
                out.append(u)
        time.sleep(THROTTLE_S)
    return out


def _commons_for_artist(artist: str, want: int) -> list[str]:
    qbase = artist.replace("_", " ")
    titles = _commons_search_files(f'"{qbase}" musician', limit=40)
    time.sleep(THROTTLE_S)
    if len(titles) < want * 2:
        titles += [t for t in _commons_search_files(f'"{qbase}"', limit=40)
                   if t not in titles]
        time.sleep(THROTTLE_S)
    needle = qbase.lower()
    titles.sort(key=lambda t: 0 if needle in t.lower() else 1)
    return _commons_resolve_urls(titles[: want * 2])


# ---- Composite discovery ----------------------------------------------
def _dedup_key(url: str) -> str:
    """Wikipedia/Wikimedia often returns the same file at multiple sizes
    (full + a thumbnail). Canonicalize to the underlying filename so we
    don't waste a slot on duplicates."""
    base = url.rsplit("/", 1)[-1]
    base = re.sub(r"^\d+px-", "", base, count=1)   # strip "250px-" prefix
    return base.lower()


def find_image_urls(artist: str, want: int) -> list[str]:
    """Wikipedia page (infobox + media-list) first, then Commons search to
    fill the remainder. Deduped by canonical filename, most-prominent-first."""
    candidates = [artist, f"{artist} (band)", f"{artist} (musician)",
                  f"{artist} (singer)", f"{artist} (rapper)"]
    chosen_title: str | None = None
    primary: str | None = None
    for t in candidates:
        s = _wiki_summary(t)
        time.sleep(THROTTLE_S)
        if not s or (s.get("type") or "").startswith("disambig"):
            continue
        img = (s.get("originalimage") or s.get("thumbnail") or {}).get("source")
        if img:
            chosen_title = (s.get("title") or t).replace(" ", "_")
            primary = img
            break
    urls: list[str] = [primary] if primary else []
    if chosen_title and want > len(urls):
        for u in _wiki_media_list(chosen_title):
            if _dedup_key(u) not in {_dedup_key(x) for x in urls}:
                urls.append(u)
            if len(urls) >= want:
                break
        time.sleep(THROTTLE_S)
    if want > len(urls):
        for u in _commons_for_artist(artist, want - len(urls)):
            if _dedup_key(u) not in {_dedup_key(x) for x in urls}:
                urls.append(u)
            if len(urls) >= want:
                break
    return urls[:want]


# ---- Download + resize -------------------------------------------------
def normalise_and_save(blob: bytes, dest: Path, max_edge: int) -> None:
    im = Image.open(io.BytesIO(blob))
    im = ImageOps.exif_transpose(im)
    if im.mode not in ("RGB", "L"):
        im = im.convert("RGB")
    w, h = im.size
    if max(w, h) > max_edge:
        if w >= h:
            im = im.resize((max_edge, int(h * max_edge / w)), Image.LANCZOS)
        else:
            im = im.resize((int(w * max_edge / h), max_edge), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    im.save(dest, "JPEG", quality=85, optimize=True)


def fetch_for_artist(artist: str, root: Path, target: int,
                     max_edge: int = 1920) -> dict:
    """End-to-end: locate up to `target` photos for `artist` and save them
    into `root/<safe(artist)>/artist_NN.jpg`. Returns counts. Safe to call
    from a worker thread (only writes JPEGs to disk, never touches Qdrant
    or the GUI)."""
    out = {"found": 0, "saved": 0, "errors": 0}
    artist_dir = root / safe(artist)
    if already_have(artist_dir, target):
        return out
    try:
        urls = find_image_urls(artist, target)
    except Exception:  # noqa: BLE001
        out["errors"] += 1
        return out
    out["found"] = len(urls)
    for j, url in enumerate(urls, 1):
        dest = artist_dir / f"artist_{j:02d}.jpg"
        if dest.exists():
            continue
        try:
            blob = _get(url, timeout=30)
            time.sleep(THROTTLE_S)
            normalise_and_save(blob, dest, max_edge)
            out["saved"] += 1
        except Exception:  # noqa: BLE001
            out["errors"] += 1
    return out
