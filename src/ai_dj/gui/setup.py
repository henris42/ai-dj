"""First-launch UX: gather music sources before the planner ever opens.

Per the customer-facing flow ("show your mp3 databases, select them all,
click Teach"), this is what comes up on a fresh install — empty Qdrant,
no `library_sources` in settings. The user adds one or more folders, names
them if they want, clicks Teach.

For now Teach persists the sources to settings.json. The actual embedding
pass (scan → MERT → upsert) is folded into the GUI in Phase 2.5; until
then Teach saves sources and emits a `teach_requested` signal the host
window can hook to start a background indexer once it exists.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .. import settings as settings_mod
from ..scan import AUDIO_EXTS


def _quick_count(root: Path) -> int:
    """Cheap preview count — file walk only, no tag read."""
    try:
        return sum(1 for p in root.rglob("*")
                   if p.is_file() and p.suffix.lower() in AUDIO_EXTS)
    except OSError:
        return 0


class SetupWindow(QMainWindow):
    """Welcome screen for empty installs. Emits `teach_requested(sources)`
    when the user has at least one source and clicks Teach."""

    teach_requested = Signal(list)   # list[settings_mod.LibrarySource]

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AI DJ — set up your library")
        self.resize(720, 520)

        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
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
        v.addSpacing(8)

        sub = QLabel("<b>Music sources</b>")
        sub.setTextFormat(Qt.RichText)
        v.addWidget(sub)

        self.sources_list = QListWidget()
        self.sources_list.setAlternatingRowColors(True)
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

        self.sources_list.itemSelectionChanged.connect(self._refresh_buttons)
        self._reload_from_settings()

    # ---- data ---------------------------------------------------------
    def _reload_from_settings(self) -> None:
        self.sources_list.clear()
        s = settings_mod.load()
        for src in s.library_sources:
            self._append_row(src)
        self._refresh_buttons()

    def _append_row(self, src: settings_mod.LibrarySource) -> None:
        n = _quick_count(Path(src.path))
        item = QListWidgetItem(f"{src.name}    —    {src.path}    ({n:,} files)")
        item.setData(Qt.UserRole, src.path)
        self.sources_list.addItem(item)

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

    # ---- actions ------------------------------------------------------
    def _on_add(self) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Choose a music folder")
        if not chosen:
            return
        path = Path(chosen).resolve()
        name = path.name or str(path)
        settings_mod.add_source(name=name, path=str(path))
        self._reload_from_settings()

    def _on_remove(self) -> None:
        item = self.sources_list.currentItem()
        if not item:
            return
        path = item.data(Qt.UserRole)
        settings_mod.remove_source(path)
        self._reload_from_settings()

    def _on_teach(self) -> None:
        s = settings_mod.load()
        if not s.library_sources:
            return
        # Phase 1.5: persist the sources and surface what'll happen next.
        # The actual scan+embed pipeline lands in Phase 2.5 — once the
        # background indexer exists, this signal wires it.
        self.teach_requested.emit(list(s.library_sources))
        QMessageBox.information(
            self, "Sources saved",
            "Your music sources are saved.\n\n"
            "Indexing (scan + embed) will run once it's available in the "
            "next version. For now, the planner will open as soon as your "
            "library has been indexed.",
        )
