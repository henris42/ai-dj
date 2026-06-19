# PyInstaller spec for AI DJ.
#
# Run from the project root on the host OS you want to target:
#
#     uv run pyinstaller aidj.spec
#
# Output: dist/AI DJ/AI DJ.exe (Windows) / dist/AI DJ.app (macOS) /
#         dist/AI DJ/AI DJ (Linux).
#
# What this bundles:
#   - The full GUI stack (PySide6 + Qt plugins, pyqtgraph, python-mpv,
#     qdrant-client, mutagen, Pillow, certifi, requests, pyacoustid,
#     musicbrainzngs).
#   - libmpv (the project-bundled libmpv-2.dll on Windows; system libmpv
#     elsewhere — we leave the system copy to do its job).
#   - assets/icon.ico for the window/taskbar icon.
#
# What this does NOT bundle (intentionally — keeps the base install small):
#   - torch + transformers + MERT model weights. These are the ~2 GB
#     embedding stack. The packaged app browses & plays a pre-built
#     index just fine without them; Teach (initial indexing) needs
#     them. First-run "Install AI engine?" prompt is the planned UX.
#
# Architectural fit:
#   - Embedded Qdrant (Phase 1) → no server / Docker / WSL to bundle.
#   - rel_path identity (Phase 1) → the db/ dir is portable; ship-ability
#     is real.
#
# pylint: skip-file
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(SPECPATH).resolve()

binaries: list[tuple[str, str]] = []
datas: list[tuple[str, str]] = [
    (str(PROJECT_ROOT / "assets" / "icon.ico"), "assets"),
    (str(PROJECT_ROOT / "assets" / "icon.svg"), "assets"),
    (str(PROJECT_ROOT / "assets" / "icon-256.png"), "assets"),
]

# libmpv: the project ships libmpv-2.dll at the root for Windows. python-mpv
# already discovers it via the PATH shim in ai_dj.player; PyInstaller needs
# it in the bundle root too.
if sys.platform == "win32":
    libmpv = PROJECT_ROOT / "libmpv-2.dll"
    if libmpv.exists():
        binaries.append((str(libmpv), "."))

hiddenimports = [
    # PySide6 plugins PyInstaller occasionally misses on minimal installs.
    "PySide6.QtSvg",
    "PySide6.QtNetwork",
    # python-mpv loads its bindings lazily.
    "mpv",
    # qdrant-client local mode pulls these via lazy imports.
    "qdrant_client.local.qdrant_local",
    # certifi is a runtime path lookup; PyInstaller usually finds it but
    # belt-and-braces.
    "certifi",
]

excludes = [
    # Indexing stack — deliberately *not* bundled. See header.
    "torch", "transformers", "tokenizers", "librosa", "numba", "llvmlite",
    "essentia", "umap",
    # Test / dev tooling.
    "pytest", "ruff", "mypy",
]


a = Analysis(
    [str(PROJECT_ROOT / "src" / "ai_dj" / "gui" / "app.py")],
    pathex=[str(PROJECT_ROOT / "src")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AI DJ",
    console=False,                       # GUI app — no terminal window
    disable_windowed_traceback=False,
    icon=str(PROJECT_ROOT / "assets" / "icon.ico"),
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,                           # UPX often breaks PySide6 DLLs
    name="AI DJ",
)
