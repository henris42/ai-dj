# AI DJ

Maps a personal music library to a vector DB and plans listening paths through
embedding space. Tracks are embedded with [MERT](https://huggingface.co/m-a-p/MERT-v1-95M),
indexed in [Qdrant](https://qdrant.tech/), genre-tagged with
[Discogs-Effnet](https://essentia.upf.edu/models/), and visualised in 3D
(scatter / first-person flythrough / fractal planet).

## What you'll need

- A music library. Anything `librosa` can decode: `mp3`, `m4a`, `flac`, `wav`,
  `aac`, `ogg`, `opus`. iTunes-purchased `.m4p` is skipped (FairPlay DRM).
- Linux or WSL2. The whole pipeline runs on CPU; if you have a GPU the MERT
  embed pass uses it (CUDA out of the box, ROCm needs a couple of extra
  steps — see [ROCm on WSL](#gpu-amd-rocm-on-wsl) below).
- [uv](https://github.com/astral-sh/uv) for the Python env. `pip install uv`
  if you don't have it.
- [Podman](https://podman.io/) or Docker for the Qdrant container.
- ~25 GB of disk for embeddings + Qdrant data on a 10 k-track library.

## 1. Start Qdrant

```bash
./scripts/qdrant_up.sh
```

This brings up Qdrant in a container on port 6333 with persistent storage in
`data/qdrant/`. `./scripts/qdrant_down.sh` stops it.

## 2. Install Python deps

```bash
uv sync
```

This creates `.venv/` from `pyproject.toml` (PyTorch, transformers, librosa,
qdrant-client, PySide6, …). `pyproject.toml` pins PyTorch built against ROCm
6.4 — if you're on CUDA or CPU only, edit the `torch*` lines in
`pyproject.toml` to drop the `+rocm6.4` suffix and remove the `[tool.uv.sources]`
ROCm index, then `uv sync` again.

## 3. Index your library

```bash
uv run python scripts/build_index.py --library /path/to/your/music
```

Walks the library, reads tags via `mutagen`, embeds each track with MERT
(30 s clip from the middle of each track, mean-pooled to a 768-d vector), and
upserts into Qdrant. Resumable: re-running skips tracks already indexed.

For a 10 k-track library expect 1–2 h on a recent dGPU, 5–10 h on CPU. Errors
go to `data/embed_errors.log`.

## 4. Tag genres

CLAP zero-shot tagging (the original approach) was unreliable, so the
canonical tagger is now a supervised model — MTG's Discogs-Effnet 400-class
genre head:

```bash
mkdir -p data/models
cd data/models
curl -LO https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json
cd ../..
uv run python scripts/tag_library_discogs.py
```

CPU-only (essentia-tensorflow ships its own bundled TF without GPU support).
~2 s/track; budget a few hours for a 10 k library. Re-running skips already
tagged tracks unless you pass `--all`.

The 400 fine-grained Discogs styles are aggregated to 12 high-level steering
buckets (Rock, Pop, Electronic, Dance, Ambient, Classical, Jazz, Metal,
Hip-Hop, Soul, Folk, Soundtrack) — matches the GUI buttons. The mapping is in
[`src/ai_dj/discogs_tag.py`](src/ai_dj/discogs_tag.py); tweak it if your
taste needs different buckets.

## 5. Run the GUI

```bash
uv run python -m ai_dj.gui.app
```

You get:

- **Library** (left, iTunes-style Artists / Albums / Tracks) — drag tracks
  into the queue.
- **Now Playing + Up Next + Queue** — auto-extending; drop tracks anywhere.
- **Steering buttons** — toggle to constrain the path to those genres.
- **Pinned artists** — type-ahead at the top to restrict to specific artists.
- **3D visualizer** with several modes: free orbit, follow the path,
  first-person along the path, top-down, or "Planet" (fractal world generated
  from the embeddings, plane flies between waypoints in time with the music).

`R` resets the camera, `Space` plays/pauses, `Backspace` is "previous track".

## GPU: AMD ROCm on WSL

For an RX 7000-series card on WSL2, the MERT embed pass needs:

- ROCm 6.4 user-space (don't install `rocm-dkms`; WSL uses a kernel driver
  shim). Follow the [official WSL guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/wsl.html)
  but skip the `usecase=wsl` step (broken at time of writing).
- The HSA runtime swap: `sudo cp /usr/lib/wsl/lib/libhsa-runtime64.so* /opt/rocm/lib/`.
- `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` (set automatically by
  `build_index.py`) — without it SDPA hangs on gfx1101 with batched input.

Sanity-check: `uv run python scripts/gpu_smoke.py`.

## Why the Windows pieces?

`pyproject-win.toml` and `launch.ps1` are for running the GUI natively on
Windows while keeping the embed/tag pipeline + Qdrant in WSL. WSLg's audio
output glitches under crossfades; native Windows mpv is reliable. If you're on
Linux without WSL, ignore them.

## Layout

```
src/ai_dj/
    scan.py            # walk library, mutagen tag read
    embed.py           # MERT encoder, batched
    index.py           # Qdrant collection wrapper
    clap_tag.py        # legacy zero-shot CLAP tagger (kept for reference)
    discogs_tag.py     # supervised Discogs-Effnet tagger (current)
    path.py            # path planner: cosine-neighbour walk with context
    projection.py      # UMAP fit, cached to data/projection.npz
    styles.py          # the 12 high-level genre buckets
    player.py          # python-mpv wrapper, optional crossfades
    paths.py           # WSL ↔ Windows path translation
    gui/app.py         # PySide6 GUI
    visualizers/       # plug-in 3D modes; @register-decorated classes
scripts/
    build_index.py            # scan + embed
    tag_library_discogs.py    # tag with Discogs-Effnet (current)
    tag_library.py            # tag with CLAP (legacy)
    qdrant_up.sh / qdrant_down.sh
data/                  # gitignored: embeddings, models, logs, qdrant volume
```

## Add another visualizer

Drop a new module in `src/ai_dj/visualizers/` that defines a class
inheriting `Visualizer` and decorated with `@register`. The GUI auto-discovers
it via `pkgutil.iter_modules` and adds a button. See `landscape.py` for a
non-trivial example.
