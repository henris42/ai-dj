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

import random

from PySide6.QtCore import QObject, QRunnable, QSize, Qt, QThreadPool, Signal
from PySide6.QtGui import QAction, QIcon, QPixmap
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListView,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMenu,
    QPushButton,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import photos as photosmod
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


class _GroupGrid(QListWidget):
    """Generic icon-mode grid: one cell per group (album or artist), text
    label and an optional photo. Activating a cell emits `seed_requested`
    with a random track id from that group — the planner takes it from
    there."""

    seed_requested = Signal(str)

    def __init__(self, *, cell: QSize, icon: QSize) -> None:
        super().__init__()
        self.setViewMode(QListView.IconMode)
        self.setResizeMode(QListView.Adjust)
        self.setMovement(QListView.Static)
        self.setSpacing(8)
        self.setIconSize(icon)
        self.setGridSize(cell)
        self.setUniformItemSizes(True)
        self.setSelectionMode(QAbstractItemView.SingleSelection)
        self.itemActivated.connect(self._on_activate)

    def _on_activate(self, item: QListWidgetItem) -> None:
        ids = item.data(Qt.UserRole)
        if ids:
            self.seed_requested.emit(random.choice(ids))


def _by_group(rows, group_key) -> "dict[tuple, list]":
    """Group song rows by (key tuple) → list of (track_id, ...)."""
    groups: dict[tuple, list] = {}
    for r in rows:
        k = group_key(r)
        if not k or k[0] in (None, ""):
            continue
        groups.setdefault(k, []).append(r)
    return groups


def _placeholder_pixmap(size: QSize, glyph: str, hex_color: str) -> QPixmap:
    """Solid color tile with a letter — used when no photo exists yet."""
    px = QPixmap(size)
    from PySide6.QtGui import QColor, QPainter, QFont
    px.fill(QColor(hex_color))
    p = QPainter(px)
    p.setPen(QColor("white"))
    f = QFont()
    f.setPointSize(int(size.height() * 0.45))
    f.setBold(True)
    p.setFont(f)
    p.drawText(px.rect(), Qt.AlignCenter, glyph[:1].upper())
    p.end()
    return px


class _AlbumsView(_GroupGrid):
    """Grid of albums (artist + album), one tile per album. Click activates
    a random track from that album as the planner's seed."""

    CELL = QSize(180, 180)
    ICON = QSize(150, 110)

    def __init__(self, all_rows: list) -> None:
        super().__init__(cell=self.CELL, icon=self.ICON)
        # rows are (tid, artist, album, title, dur, genre)
        groups = _by_group(all_rows, lambda r: (r[2], r[1]))   # (album, artist)
        # Stable order: by artist then album.
        order = sorted(groups.keys(), key=lambda k: (k[1].lower(), k[0].lower()))
        for album, artist in order:
            ids = [r[0] for r in groups[(album, artist)]]
            item = QListWidgetItem(f"{album}\n{artist}")
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            item.setData(Qt.UserRole, ids)
            item.setIcon(QIcon(_placeholder_pixmap(self.ICON, album, "#2c3340")))
            self.addItem(item)


class _ArtistsView(_GroupGrid):
    """Grid of artists, using the auto-fetched photo when present
    (<bundle>/Photos/<Artist>/artist_01.jpg). Click activates a random
    track from that artist."""

    CELL = QSize(190, 220)
    ICON = QSize(160, 160)

    def __init__(self, all_rows: list) -> None:
        super().__init__(cell=self.CELL, icon=self.ICON)
        groups = _by_group(all_rows, lambda r: (r[1],))        # (artist,)
        order = sorted(groups.keys(), key=lambda k: k[0].lower())
        root = photosmod.photo_root()
        for (artist,) in order:
            ids = [r[0] for r in groups[(artist,)]]
            item = QListWidgetItem(f"{artist}\n{len(ids)} tracks")
            item.setTextAlignment(Qt.AlignHCenter | Qt.AlignTop)
            item.setData(Qt.UserRole, ids)
            item.setIcon(self._artist_icon(root, artist))
            self.addItem(item)

    def _artist_icon(self, root, artist: str) -> QIcon:
        safe = photosmod.safe(artist)
        candidate = root / safe / "artist_01.jpg"
        if candidate.exists():
            pm = QPixmap(str(candidate)).scaled(
                self.ICON, Qt.KeepAspectRatioByExpanding, Qt.SmoothTransformation)
            return QIcon(pm)
        return QIcon(_placeholder_pixmap(self.ICON, artist, "#4a3340"))


class _OutliersSignals(QObject):
    finished = Signal(list)              # list[str] of track ids


class _OutliersTask(QRunnable):
    def __init__(self, idx: TrackIndex, signals: _OutliersSignals,
                 sample: int, n: int, refresh: bool) -> None:
        super().__init__()
        self.idx = idx
        self.signals = signals
        self.sample = sample
        self.n = n
        self.refresh = refresh

    def run(self) -> None:
        from .. import discover as discmod
        try:
            ids = discmod.outliers(self.idx, sample=self.sample, n=self.n,
                                   refresh=self.refresh)
        except Exception:  # noqa: BLE001
            ids = []
        self.signals.finished.emit(ids)


class _DiscoverView(QWidget):
    """Discover: Surprise (random) + Outliers (embedding-driven isolation).

    Bridges, twins, and unexplored regions are next iterations on top of
    this. Outliers runs in a QThreadPool — first call computes and
    caches to <bundle>/db/discover_outliers.json, subsequent opens are
    instant. "Recompute" forces a refresh."""

    seed_requested = Signal(str)
    SAMPLE_RAND = 30
    SAMPLE_OUT = 250

    def __init__(self, all_rows: list, idx: TrackIndex) -> None:
        super().__init__()
        self._all = all_rows
        self._row_by_id = {r[0]: r for r in all_rows}
        self.idx = idx
        self._pool = QThreadPool.globalInstance()
        self._signals: _OutliersSignals | None = None
        self._mode = "surprise"

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        # Sub-tab buttons
        subrow = QHBoxLayout()
        self._sub_buttons: dict[str, QPushButton] = {}
        for label in ("Surprise", "Outliers"):
            b = QPushButton(label)
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, m=label.lower(): self._switch(m))
            subrow.addWidget(b)
            self._sub_buttons[label.lower()] = b
        self._intro = QLabel("Hidden corners of your library — re-roll for more.")
        self._intro.setStyleSheet("color: #aaa;")
        subrow.addWidget(self._intro, stretch=1)
        self.action_btn = QPushButton("Reshuffle")
        self.action_btn.clicked.connect(self._action)
        subrow.addWidget(self.action_btn)
        v.addLayout(subrow)

        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(("Artist", "Album", "Title", "Genre"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.verticalHeader().setDefaultSectionSize(24)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Interactive)
        h.setSectionResizeMode(1, QHeaderView.Interactive)
        h.setSectionResizeMode(2, QHeaderView.Stretch)
        h.setSectionResizeMode(3, QHeaderView.Interactive)
        self.table.itemDoubleClicked.connect(self._on_double_click)
        v.addWidget(self.table, stretch=1)
        self._switch("surprise")

    # ----- mode switching -----
    def _switch(self, mode: str) -> None:
        self._mode = mode
        for k, btn in self._sub_buttons.items():
            btn.setChecked(k == mode)
        if mode == "surprise":
            self._intro.setText("Hidden corners of your library — re-roll for more.")
            self.action_btn.setText("Reshuffle")
            self.action_btn.setEnabled(True)
            self._roll_surprise()
        else:
            self._intro.setText(
                "Outliers — tracks the embedding finds isolated. "
                "(First compute may take ~10–30 s; cached afterwards.)")
            self.action_btn.setText("Recompute")
            self._load_outliers(refresh=False)

    def _action(self) -> None:
        if self._mode == "surprise":
            self._roll_surprise()
        else:
            self._load_outliers(refresh=True)

    # ----- surprise -----
    def _roll_surprise(self) -> None:
        sample = (random.sample(self._all, min(self.SAMPLE_RAND, len(self._all)))
                  if self._all else [])
        self._fill_rows(sample)

    # ----- outliers -----
    def _load_outliers(self, *, refresh: bool) -> None:
        self.action_btn.setEnabled(False)
        self._fill_rows([])
        self._intro.setText("Computing outliers…")
        self._signals = _OutliersSignals()
        self._signals.finished.connect(self._on_outliers, Qt.QueuedConnection)
        self._pool.start(_OutliersTask(
            self.idx, self._signals,
            sample=self.SAMPLE_OUT, n=self.SAMPLE_RAND, refresh=refresh))

    def _on_outliers(self, ids: list) -> None:
        rows = [self._row_by_id[tid] for tid in ids if tid in self._row_by_id]
        self._fill_rows(rows)
        self.action_btn.setEnabled(True)
        self._intro.setText(
            f"Outliers — {len(rows)} most isolated tracks "
            "(re-compute to refresh after Teach adds new music).")

    # ----- shared -----
    def _fill_rows(self, rows) -> None:
        self.table.setRowCount(len(rows))
        for r, (tid, artist, album, title, _dur, genre) in enumerate(rows):
            for c, val in enumerate((artist, album, title, genre)):
                it = QTableWidgetItem(val)
                if c == 0:
                    it.setData(Qt.UserRole, tid)
                self.table.setItem(r, c, it)

    def _on_double_click(self, item: QTableWidgetItem) -> None:
        row = item.row()
        first = self.table.item(row, 0)
        if first:
            tid = first.data(Qt.UserRole)
            if tid:
                self.seed_requested.emit(tid)


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
        for name in ("Songs", "Albums", "Artists", "Discover"):
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

        # Albums + Artists share the Songs view's loaded rows — no second
        # scroll over Qdrant.
        self.albums = _AlbumsView(self.songs._all_rows)
        self.albums.seed_requested.connect(self.seed_requested)
        self.stack.addWidget(self.albums)

        self.artists = _ArtistsView(self.songs._all_rows)
        self.artists.seed_requested.connect(self.seed_requested)
        self.stack.addWidget(self.artists)

        self.discover = _DiscoverView(self.songs._all_rows, idx)
        self.discover.seed_requested.connect(self.seed_requested)
        self.stack.addWidget(self.discover)

        self._show("Songs")

    def _show(self, name: str) -> None:
        order = ("Songs", "Albums", "Artists", "Discover")
        for n, btn in self._buttons.items():
            btn.setChecked(n == name)
        self.stack.setCurrentIndex(order.index(name))

    def sizeHint(self) -> QSize:  # noqa: N802
        return QSize(1100, 700)
