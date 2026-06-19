"""First-launch UX: gather music sources, then run the indexer in the
background and transition to the main window when it's done.

Two pages in a QStackedWidget:

  - **Sources** — Welcome + list + Add/Remove + Teach. What an empty
    install opens to.
  - **Indexing** — phase label + progress bar + tail of recent files +
    Cancel. Shown while the indexer worker runs.

When the indexer finishes successfully (or with no work to do), we emit
`indexing_complete` so `main()` can swap to the planner. Errors stay on
the indexing page with a "Close" button so the user can see what happened.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import (QObject, QRunnable, Qt, QThreadPool, Signal, Slot)
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import settings as settings_mod
from ..index import TrackIndex
from ..indexer import IndexProgress, IndexResult, index_sources
from ..scan import AUDIO_EXTS


def _quick_count(root: Path) -> int:
    try:
        return sum(1 for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    except OSError:
        return 0


class _IndexerSignals(QObject):
    """Cross-thread bridge — QRunnable can't have signals itself."""
    progress = Signal(str, int, int, str)        # phase, current, total, file
    finished = Signal(object)                    # IndexResult


class _IndexerTask(QRunnable):
    def __init__(self, sources, idx: TrackIndex, signals: _IndexerSignals) -> None:
        super().__init__()
        self.sources = list(sources)
        self.idx = idx
        self.signals = signals

    def run(self) -> None:
        def cb(p: IndexProgress) -> None:
            self.signals.progress.emit(p.phase, p.current, p.total, p.file or p.message)
        try:
            result = index_sources(self.sources, self.idx, progress=cb)
        except Exception as e:  # noqa: BLE001
            result = IndexResult()
            result.errors.append(f"{type(e).__name__}: {e}")
        self.signals.finished.emit(result)


class _EngineInstallSignals(QObject):
    line = Signal(str)
    finished = Signal(int)               # returncode (0 = OK)


class _EngineInstallTask(QRunnable):
    """Runs the AI-engine install via `uv pip install`, streaming each
    output line back to the GUI."""

    def __init__(self, signals: _EngineInstallSignals) -> None:
        super().__init__()
        self.signals = signals

    def run(self) -> None:
        from .. import ai_engine
        rc = ai_engine.install(
            project_root=Path(__file__).resolve().parents[3],
            on_line=lambda s: self.signals.line.emit(s),
        )
        self.signals.finished.emit(rc)


class SetupWindow(QMainWindow):
    """First-launch screen. Add sources → Teach → indexer runs → planner.

    `indexing_complete` fires when the worker finishes successfully (or
    finds the library was already indexed). Errors stay shown here; the
    caller decides whether to retry."""

    indexing_complete = Signal()

    def __init__(self, idx: TrackIndex) -> None:
        super().__init__()
        self.idx = idx
        self.setWindowTitle("AI DJ — set up your library")
        self.resize(720, 540)

        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._signals: _IndexerSignals | None = None

        self.stack = QStackedWidget()
        self.setCentralWidget(self.stack)
        self.stack.addWidget(self._build_sources_page())          # 0
        self.stack.addWidget(self._build_engine_page())           # 1
        self.stack.addWidget(self._build_indexing_page())         # 2

        self._reload_from_settings()

    # ===== sources page =====
    def _build_sources_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(24, 22, 24, 22)
        v.setSpacing(10)

        title = QLabel("<h2 style='margin:0'>Welcome to AI DJ</h2>")
        title.setTextFormat(Qt.RichText)
        v.addWidget(title)

        intro = QLabel(
            "Point AI DJ at the folders that hold your music — one or many. "
            "It will scan them, learn how the tracks sound, and play paths "
            "through your library that nothing else can.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #bbb;")
        v.addWidget(intro)
        v.addSpacing(6)

        sub = QLabel("<b>Music sources</b>")
        sub.setTextFormat(Qt.RichText)
        v.addWidget(sub)

        self.sources_list = QListWidget()
        self.sources_list.setAlternatingRowColors(True)
        self.sources_list.itemSelectionChanged.connect(self._refresh_buttons)
        v.addWidget(self.sources_list, stretch=1)

        row = QHBoxLayout()
        self.add_btn = QPushButton("+ Add music source…")
        self.add_btn.clicked.connect(self._on_add)
        row.addWidget(self.add_btn)
        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._on_remove)
        self.remove_btn.setEnabled(False)
        row.addWidget(self.remove_btn)
        row.addStretch(1)
        v.addLayout(row)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet("color: #333;")
        v.addWidget(sep)

        bottom = QHBoxLayout()
        self.summary = QLabel("Add at least one music source to begin.")
        self.summary.setStyleSheet("color: #888;")
        bottom.addWidget(self.summary, stretch=1)
        self.teach_btn = QPushButton("Teach")
        self.teach_btn.setMinimumWidth(140)
        self.teach_btn.setStyleSheet(
            "QPushButton { padding: 8px 18px; font-weight: bold; }")
        self.teach_btn.clicked.connect(self._on_teach)
        self.teach_btn.setEnabled(False)
        bottom.addWidget(self.teach_btn)
        v.addLayout(bottom)
        return page

    def _reload_from_settings(self) -> None:
        self.sources_list.clear()
        for src in settings_mod.load().library_sources:
            n = _quick_count(Path(src.path))
            item = QListWidgetItem(f"{src.name}    —    {src.path}    ({n:,} files)")
            item.setData(Qt.UserRole, src.path)
            self.sources_list.addItem(item)
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        any_source = self.sources_list.count() > 0
        self.teach_btn.setEnabled(any_source)
        self.remove_btn.setEnabled(self.sources_list.currentRow() >= 0)
        if any_source:
            total = self.sources_list.count()
            self.summary.setText(f"{total} source{'s' if total != 1 else ''} ready to teach.")
            self.summary.setStyleSheet("color: #ddd;")
        else:
            self.summary.setText("Add at least one music source to begin.")
            self.summary.setStyleSheet("color: #888;")

    def _on_add(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose a music folder")
        if not chosen:
            return
        path = Path(chosen).resolve()
        settings_mod.add_source(name=path.name or str(path), path=str(path))
        self._reload_from_settings()

    def _on_remove(self) -> None:
        item = self.sources_list.currentItem()
        if not item:
            return
        settings_mod.remove_source(item.data(Qt.UserRole))
        self._reload_from_settings()

    def _on_teach(self) -> None:
        sources = settings_mod.load().library_sources
        if not sources:
            return
        # If the embedding stack isn't installed yet, walk through engine
        # install first; the indexing page kicks in after a successful pip.
        from .. import ai_engine
        if not ai_engine.is_available():
            self._pending_sources = sources
            self._engine_missing.setText(
                "AI engine missing: " + ", ".join(ai_engine.missing()))
            self.stack.setCurrentIndex(1)
            return
        self._start_indexing(sources)

    # ===== engine install page =====
    def _build_engine_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(24, 22, 24, 22)
        v.setSpacing(8)

        title = QLabel("<h2 style='margin:0'>Install the AI engine</h2>")
        title.setTextFormat(Qt.RichText)
        v.addWidget(title)

        intro = QLabel(
            "AI DJ needs the embedding engine (torch + transformers + librosa) "
            "to learn how your tracks sound. This is a one-time install of "
            "roughly 1–2 GB, fetched via uv.")
        intro.setWordWrap(True)
        intro.setStyleSheet("color: #bbb;")
        v.addWidget(intro)

        self._engine_missing = QLabel("")
        self._engine_missing.setStyleSheet("color: #888;")
        v.addWidget(self._engine_missing)

        self._engine_log = QPlainTextEdit()
        self._engine_log.setReadOnly(True)
        self._engine_log.setStyleSheet("font-family: monospace; font-size: 11px;")
        v.addWidget(self._engine_log, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self._engine_back_btn = QPushButton("Back")
        self._engine_back_btn.clicked.connect(lambda: self.stack.setCurrentIndex(0))
        btn_row.addWidget(self._engine_back_btn)
        self._engine_install_btn = QPushButton("Install AI engine")
        self._engine_install_btn.setStyleSheet(
            "QPushButton { padding: 8px 18px; font-weight: bold; }")
        self._engine_install_btn.clicked.connect(self._on_engine_install)
        btn_row.addWidget(self._engine_install_btn)
        v.addLayout(btn_row)
        return page

    def _on_engine_install(self) -> None:
        self._engine_install_btn.setEnabled(False)
        self._engine_back_btn.setEnabled(False)
        self._engine_log.appendPlainText("Starting AI engine install — this may take several minutes.\n")
        self._engine_signals = _EngineInstallSignals()
        self._engine_signals.line.connect(self._engine_log.appendPlainText,
                                          Qt.QueuedConnection)
        self._engine_signals.finished.connect(self._on_engine_finished,
                                              Qt.QueuedConnection)
        self._pool.start(_EngineInstallTask(self._engine_signals))

    @Slot(int)
    def _on_engine_finished(self, rc: int) -> None:
        from .. import ai_engine
        if rc == 0 and ai_engine.is_available():
            self._engine_log.appendPlainText("\nAI engine installed. Continuing with Teach.")
            if getattr(self, "_pending_sources", None):
                self._start_indexing(self._pending_sources)
        else:
            self._engine_log.appendPlainText(f"\nInstall failed (code {rc}).")
            self._engine_install_btn.setEnabled(True)
            self._engine_back_btn.setEnabled(True)

    # ===== indexing page =====
    def _build_indexing_page(self) -> QWidget:
        page = QWidget()
        v = QVBoxLayout(page)
        v.setContentsMargins(24, 22, 24, 22)
        v.setSpacing(10)

        title = QLabel("<h2 style='margin:0'>Teaching AI DJ your library</h2>")
        title.setTextFormat(Qt.RichText)
        v.addWidget(title)

        self.idx_phase = QLabel("Starting…")
        self.idx_phase.setStyleSheet("color: #ccc;")
        v.addWidget(self.idx_phase)

        self.idx_bar = QProgressBar()
        self.idx_bar.setRange(0, 0)            # indeterminate until we have a total
        self.idx_bar.setTextVisible(True)
        v.addWidget(self.idx_bar)

        self.idx_file = QLabel("")
        self.idx_file.setStyleSheet("color: #888; font-family: monospace;")
        self.idx_file.setWordWrap(True)
        v.addWidget(self.idx_file)

        v.addStretch(1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        self.idx_close_btn = QPushButton("Close")
        self.idx_close_btn.clicked.connect(self.close)
        self.idx_close_btn.setVisible(False)         # only after finish/error
        btn_row.addWidget(self.idx_close_btn)
        v.addLayout(btn_row)
        return page

    def _start_indexing(self, sources) -> None:
        # Disable the source page controls so the user can't re-Teach mid-run.
        for w in (self.add_btn, self.remove_btn, self.teach_btn, self.sources_list):
            w.setEnabled(False)
        self.stack.setCurrentIndex(2)            # indexing page (engine page is 1)

        self._signals = _IndexerSignals()
        self._signals.progress.connect(self._on_progress, Qt.QueuedConnection)
        self._signals.finished.connect(self._on_finished, Qt.QueuedConnection)
        self._pool.start(_IndexerTask(sources, self.idx, self._signals))

    @Slot(str, int, int, str)
    def _on_progress(self, phase: str, current: int, total: int, file: str) -> None:
        if phase == "scanning":
            self.idx_phase.setText(f"Scanning your library — {current:,} files seen")
            self.idx_bar.setRange(0, 0)
        elif phase == "embedding":
            self.idx_phase.setText(f"Learning your tracks — {current:,} of {total:,}")
            self.idx_bar.setRange(0, max(1, total))
            self.idx_bar.setValue(current)
        elif phase == "error":
            self.idx_phase.setText("Indexing error")
        else:                                       # done
            self.idx_phase.setText("Finished")
        self.idx_file.setText(file or "")

    @Slot(object)
    def _on_finished(self, result: IndexResult) -> None:
        if result.errors and result.embedded == 0 and result.scanned == 0:
            # Hard failure (e.g. missing ML stack). Show the message; user can close.
            self.idx_phase.setText("Couldn't teach the library")
            self.idx_file.setText("\n".join(result.errors[:5]))
            self.idx_close_btn.setVisible(True)
            return
        if result.scanned == 0:
            self.idx_phase.setText("No audio files in the configured sources")
            self.idx_close_btn.setVisible(True)
            return
        # Success (full or already-indexed). Hand off to the main window.
        self.indexing_complete.emit()

    # ---- the empty-sources fallback (kept for older callers) ----
    def _legacy_save_only(self) -> None:
        QMessageBox.information(
            self, "Sources saved",
            "Your music sources are saved.",
        )
