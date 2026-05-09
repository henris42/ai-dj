# AI DJ

Maps your music library into a vector DB ([Qdrant](https://qdrant.tech/)),
tags it with supervised genre classification ([Discogs-Effnet](https://essentia.upf.edu/models/)),
and plans listening paths through the resulting embedding space — visualised
in 3D, including a fractal planet you fly between songs.

![icon](assets/icon-256.png)

---

## Quick start

You need: **Linux or WSL2**, **[uv](https://github.com/astral-sh/uv)**, and
**Docker / Podman** (for Qdrant).

```bash
git clone https://github.com/henris42/ai-dj && cd ai-dj
uv sync                                     # installs Python deps
./scripts/qdrant_up.sh                      # starts Qdrant on :6333
```

Download the genre tagger (~20 MB) once:

```bash
mkdir -p data/models && cd data/models
curl -LO https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.pb
curl -LO https://essentia.upf.edu/models/classification-heads/genre_discogs400/genre_discogs400-discogs-effnet-1.json
cd ../..
```

Index your library (this is the long part — a few hours for 10 k tracks):

```bash
uv run python scripts/build_index.py --library /path/to/your/music
uv run python scripts/tag_library_discogs.py
```

Run it:

```bash
uv run python -m ai_dj.gui.app
```

That's it. The window has a library browser on the left, a queue, an "Up
Next" panel, genre-steer buttons across the top, and a 3D visualizer on the
right (try the **Planet** mode).

---

## Notes

- **GPU**: `build_index.py` uses CUDA out of the box. For AMD on WSL, see
  [ROCm setup notes](#rocm-on-wsl) below.
- **Re-runnable**: both `build_index.py` and `tag_library_discogs.py` skip
  tracks they've already processed. Re-run anytime you add music.
- **Supported formats**: anything `librosa` decodes — mp3, m4a, flac, wav,
  aac, ogg, opus. iTunes DRM (`.m4p`) is skipped.
- **Disk**: budget ~25 GB for embeddings + Qdrant data on a 10 k library.

---

## Run on Windows (with the GUI native)

WSL's audio output (WSLg) glitches under crossfades, so the GUI can run
natively on Windows while Qdrant + indexing stay in WSL.

Install Python deps for the Windows side once:

```powershell
cd C:\Users\<you>\ai-dj
uv sync --no-dev
```

Then either:

- run `launch.ps1` directly, or
- run `install-shortcut.ps1` once → AI DJ appears in the **Start menu**
  (Win → "AI DJ"). Right-click the running window's taskbar icon → Pin to
  taskbar to keep it there.

`launch.ps1` boots Qdrant in WSL, waits for it, and surfaces failures as a
Windows MessageBox (full log: `data/launch.log`).

---

## ROCm on WSL

For an RX 7000-series card on WSL2:

- ROCm 6.4 user-space (no `rocm-dkms` — WSL uses a kernel shim). Follow the
  [official guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/wsl.html)
  but skip `usecase=wsl` (broken at time of writing).
- Swap the HSA runtime: `sudo cp /usr/lib/wsl/lib/libhsa-runtime64.so* /opt/rocm/lib/`.
- The embed script auto-sets `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` —
  without it SDPA hangs on gfx1101 with batched input.

Sanity check: `uv run python scripts/gpu_smoke.py`.

---

## Layout

```
src/ai_dj/
    scan.py            walk library, mutagen tag read
    embed.py           MERT encoder, batched
    index.py           Qdrant collection wrapper
    discogs_tag.py     supervised Discogs-Effnet tagger
    clap_tag.py        legacy CLAP zero-shot tagger (kept for reference)
    path.py            path planner — context-aware cosine-neighbour walk
    projection.py      UMAP fit, cached to data/projection.npz
    styles.py          12 high-level genre buckets
    player.py          python-mpv wrapper, optional crossfades
    gui/app.py         PySide6 GUI
    visualizers/       plug-in 3D modes; @register-decorated classes
scripts/
    build_index.py            scan + embed
    tag_library_discogs.py    tag with Discogs-Effnet
    qdrant_up.sh / qdrant_down.sh
data/                  gitignored: embeddings, models, logs, qdrant volume
```

Add a visualizer: drop a module in `src/ai_dj/visualizers/` with a class
decorated `@register`. The GUI auto-discovers it. See `landscape.py` for a
non-trivial example (the planet).
