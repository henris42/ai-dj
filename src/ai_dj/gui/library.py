"""Library screen — iTunes-style Songs / Albums / Artists views.

Phase 2 first cut: Songs is the working view (sortable columns, live search,
right-click → "Start a mix from this"). Albums and Artists grids land
incrementally; their tabs are present but show placeholders for now.

The screen reads from the indexed Qdrant collection — so it lights up the
moment the indexer (Phase 2.5) populates anything. Until then, an empty
install shows the SetupWindow instead.
"""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..index import COLLECTION, TrackIndex


def _fmt_dur(seconds: "float | None") -> str:
    if seconds is None or seconds <= 0:
        return ""
    s = int(seconds)
    m, s = divmod(s, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class _SongsView(QWidget):
    """Sortable table of every track, with live search and a context menu."""

    seed_requested = Signal(str)        # emits track_id when "Start a mix from this"

    COLS = ("Artist", "Album", "Title", "Time", "Genre")

    def __init__(self, idx: TrackIndex) -> None:
        super().__init__()
        self.idx = idx
        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        row = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search artist, album, title, genre…")
        self.search.setClearButtonEnabled(True)
        self.search.textChanged.connect(self._filter)
        row.addWidget(self.search, stretch=1)
        self.count_lbl = QLabel("")
        self.count_lbl.setStyleSheet("color: #888;")
        row.addWidget(self.count_lbl)
        v.addLayout(row)

        self.table = QTableWidget(0, len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(22)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Interactive)
        h.setSectionResizeMode(1, QHeaderView.Interactive)
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        h.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(4, QHeaderView.Interactive)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self._on_context_menu)
        v.addWidget(self.table, stretch=1)

        self._all_rows: list[tuple[str, str, str, str, str, str]] = []
        # tuple = (track_id, artist, album, title, dur_str, genre)
        self._load()

    def _load(self) -> None:
        """Pull every track from Qdrant. 11k rows is small enough that we
        load once and filter client-side — no virtual model needed."""
        offset = None
        rows: list[tuple[str, str, str, str, str, str]] = []
        while True:
            recs, offset = self.idx.client.scroll(
                COLLECTION, limit=1024, offset=offset,
                with_payload=True, with_vectors=False,
            )
            for r in recs:
                pl = r.payload or {}
                rows.append((
                    str(r.id),
                    (pl.get("artist") or "?"),
                    (pl.get("album") or ""),
                    (pl.get("title")
                     or Path(pl.get("rel_path") or pl.get("path") or "").stem
                     or "(unknown)"),
                    _fmt_dur(pl.get("duration_s")),
                    (pl.get("ai_genre") or pl.get("genre") or ""),
                ))
            if offset is None:
                break
        self._all_rows = rows
        self._fill(rows)

    def _fill(self, rows) -> None:
        self.table.setSortingEnabled(False)
        self.table.setRowCount(len(rows))
        for r, (tid, artist, album, title, dur, genre) in enumerate(rows):
            for c, val in enumerate((artist, album, title, dur, genre)):
                it = QTableWidgetItem(val)
                if c == 0:
                    it.setData(Qt.UserRole, tid)        # store id on first cell
                self.table.setItem(r, c, it)
        self.table.setSortingEnabled(True)
        self.count_lbl.setText(f"{len(rows):,} of {len(self._all_rows):,} tracks")

    def _filter(self, q: str) -> None:
        q = q.strip().lower()
        if not q:
            self._fill(self._all_rows)
            return
        terms = q.split()
        def match(r):
            blob = " ".join(r[1:]).lower()
            return all(t in blob for t in terms)
        self._fill([r for r in self._all_rows if match(r)])

    def _on_context_menu(self, pos) -> None:
        row = self.table.indexAt(pos).row()
        if row < 0:
            return
        tid_item = self.table.item(row, 0)
        if not tid_item:
            return
        tid = tid_item.data(Qt.UserRole)
        menu = QMenu(self)
        act = QAction("Start a mix from this track", self)
        act.triggered.connect(lambda: self.seed_requested.emit(tid))
        menu.addAction(act)
        menu.exec(self.table.viewport().mapToGlobal(pos))


class LibraryWindow(QMainWindow):
    """Non-modal Library viewer — opens from the main window's transport.

    Emits `seed_requested(track_id)` so the host can start a mix from the
    selected seed without this window having to know about the planner."""

    seed_requested = Signal(str)

    def __init__(self, idx: TrackIndex, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI DJ — Library")
        self.resize(1100, 700)

        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(8, 6, 8, 6)
        v.setSpacing(4)

        # Tab strip across the top — simple toggle buttons rather than a
        # full QTabWidget so we control the look later (iTunes-y).
        tabs = QHBoxLayout()
        self._buttons: dict[str, QPushButton] = {}
        for name in ("Songs", "Albums", "Artists"):
            b = QPushButton(name)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, n=name: self._show(n))
            tabs.addWidget(b)
            self._buttons[name] = b
        tabs.addStretch(1)
        v.addLayout(tabs)

        self.stack = QStackedWidget()
        v.addWidget(self.stack, stretch=1)

        self.songs = _SongsView(idx)
        self.songs.seed_requested.connect(self.seed_requested)
        self.stack.addWidget(self.songs)

        albums_placeholder = QLabel("Albums grid — coming in the next pass")
        albums_placeholder.setAlignment(Qt.AlignCenter)
        albums_placeholder.setStyleSheet("color: #666; padding: 80px;")
        self.stack.addWidget(albums_placeholder)

        artists_placeholder = QLabel("Artists grid — coming in the next pass\n"
                                     "(will use the photos auto-fetched into "
                                     "<bundle>/Photos/<Artist>/)")
        artists_placeholder.setAlignment(Qt.AlignCenter)
        artists_placeholder.setStyleSheet("color: #666; padding: 80px;")
        self.stack.addWidget(artists_placeholder)

        self._show("Songs")

    def _show(self, name: str) -> None:
        order = ("Songs", "Albums", "Artists")
        for n, btn in self._buttons.items():
            btn.setChecked(n == name)
        self.stack.setCurrentIndex(order.index(name))

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(1100, 700)
