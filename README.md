# AI DJ

Maps your music library into a vector DB ([Qdrant](https://qdrant.tech/)),
tags it with supervised genre classification ([Discogs-Effnet](https://essentia.upf.edu/models/)),
and plans listening paths through the resulting embedding space — visualised
in 3D, including a fractal planet you fly between songs.

![icon](assets/icon-256.png)

---

## Quick start

You need:

- **Python 3.10 or 3.11**
- **[uv](https://github.com/astral-sh/uv)** — `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Docker or Podman** (for the Qdrant container)

One command sets everything up and launches the app:

```bash
git clone https://github.com/henris42/ai-dj && cd ai-dj
python3 run.py --library /path/to/your/music
```

`run.py` does:

1. Detects your hardware and installs the right torch (CPU / NVIDIA CUDA /
   AMD ROCm / Apple MPS — automatically).
2. Downloads the genre-tagger model files (~20 MB, once).
3. Brings up Qdrant in a container and waits for it.
4. Scans + embeds + genre-tags your library (this is the long part — a
   couple of hours for 10 k tracks on CPU, faster on GPU).
5. Launches the GUI.

After the first run, just:

```bash
python3 run.py
```

— same script, idempotent, skips the steps that are already done. Re-run
with `--library PATH` whenever you add music; only new tracks are processed.

Supported formats: anything `librosa` decodes — mp3, m4a, flac, wav, aac,
ogg, opus. iTunes DRM (`.m4p`) is skipped.

---

## What you get

A window with:

- **Library** browser (Artists / Albums / Tracks, iTunes-style) — drag
  tracks into the queue.
- **Now Playing + Up Next + Queue** — auto-extending; drop tracks anywhere.
- **Steering buttons** across the top (Rock, Pop, Electronic, …) to
  constrain the path.
- **Pinned artists** — type-ahead to restrict to specific artists.
- **3D visualizer** — Free orbit, Follow path, First-person, Top-down, or
  **Planet** (fractal world generated from the embeddings, plane flies
  between waypoints in time with the music).

`R` resets the camera, `Space` plays/pauses, `Backspace` is "previous track".

---

## ROCm on WSL2

`run.py` will install ROCm torch automatically when it sees `/dev/kfd` +
`rocminfo` on the host. For an RX 7000-series card on WSL2 specifically,
ROCm itself needs a couple of extra steps before that detection works:

- ROCm 6.4 user-space (no `rocm-dkms` — WSL uses a kernel shim). Follow the
  [official guide](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/install/3rd-party/wsl.html)
  but skip `usecase=wsl` (broken at time of writing).
- Swap the HSA runtime: `sudo cp /usr/lib/wsl/lib/libhsa-runtime64.so* /opt/rocm/lib/`.
- The embed script auto-sets `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` —
  without it SDPA hangs on gfx1101 with batched input.

Sanity check: `uv run python scripts/gpu_smoke.py`.

---

## Native Windows GUI (optional)

WSL's audio (WSLg) glitches under crossfades, so on Windows you can keep
Qdrant + indexing in WSL and run the GUI natively:

```powershell
cd C:\path\to\ai-dj
python run.py --no-launch     # set up the Windows venv only
.\install-shortcut.ps1        # adds "AI DJ" to the Start menu
```

`launch.ps1` boots Qdrant in WSL, waits for it, and surfaces failures as a
Windows MessageBox (full log: `data\launch.log`). After launching once,
right-click the running window's taskbar icon → Pin to taskbar to keep it.

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
