"""AI DJ — live-mix desktop app.

Layout:
  [ Rock ][ Pop ][ Electronic ][ Dance ][ Ambient ][ ... ]      <- style toggles
  ┌──────────────────┬────────────────────────────────────┐
  │ ▶ now playing    │                                    │
  │                  │        library map                 │
  │ queue:           │   (grey dots = library,            │
  │   1. next        │    green dots = style matches,     │
  │   2. then        │    yellow = upcoming path,         │
  │   ...            │    red pulsing = current)          │
  └──────────────────┴────────────────────────────────────┘
  [search song...] [Start] [Pause] [Skip] [ ] xfade [===] 6s

The path is continuous — when a track ends, the planner picks the next one
based on the current pivot, active styles, and already-played history. Styles
and search adds steer the path in real time without interrupting playback.
"""
from __future__ import annotations

import logging
import math
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide6.QtCore import QPointF, QSize, Qt, QStringListModel, QTimer, Signal
from PySide6.QtGui import QColor, QCursor, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import QButtonGroup, QToolTip
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QCompleter,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ai_dj import path as planpath
from ai_dj import player as playermod
from ai_dj import projection as projmod
from ai_dj import styles as stylesmod
from ai_dj import visualizers as vizmod
from ai_dj.index import COLLECTION, TrackIndex

logger = logging.getLogger(__name__)

QUEUE_TARGET = 50         # always keep at least this many upcoming tracks
RECENT_ARTISTS_AVOID = 3  # don't pick a track whose artist matches the last N upcoming
ROTATE_STEP_DEG = 10
ZOOM_STEP = 1.25
ADVANCE_MARGIN_S = 0.5    # start the next track this far before the current one ends
PLAY_LOAD_TIMEOUT_S = 4.0 # if mpv hasn't reported a duration this long after play(), skip the track


MIME_TRACK_ID = "application/x-aidj-track-id"


def _fmt_time(seconds: float | None) -> str:
    """Format seconds as mm:ss (or h:mm:ss for tracks ≥ 1h)."""
    if seconds is None or seconds < 0:
        return "--:--"
    s = int(seconds)
    m, s = divmod(s, 60)
    if m >= 60:
        h, m = divmod(m, 60)
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m}:{s:02d}"


class TrackDragList(QListWidget):
    """QListWidget that puts the item's `Qt.UserRole` track_id into the drag
    mime data so the queue can pick it up on drop."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setDragEnabled(True)

    def mimeData(self, items):  # noqa: N802
        md = super().mimeData(items)
        if items:
            tid = items[0].data(Qt.UserRole)
            if tid:
                md.setData(MIME_TRACK_ID, str(tid).encode("utf-8"))
        return md


class QueueDropList(QListWidget):
    """Queue list that accepts drops carrying a track_id mime type and emits a
    signal so the Window can splice the track into `upcoming` at the drop row."""

    track_dropped = Signal(str, int)  # track_id, target row index

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)

    def dragEnterEvent(self, ev):  # noqa: N802
        if ev.mimeData().hasFormat(MIME_TRACK_ID):
            ev.acceptProposedAction()
        else:
            super().dragEnterEvent(ev)

    def dragMoveEvent(self, ev):  # noqa: N802
        if ev.mimeData().hasFormat(MIME_TRACK_ID):
            ev.acceptProposedAction()
        else:
            super().dragMoveEvent(ev)

    def dropEvent(self, ev):  # noqa: N802
        if ev.mimeData().hasFormat(MIME_TRACK_ID):
            tid = bytes(ev.mimeData().data(MIME_TRACK_ID)).decode("utf-8")
            row = self.indexAt(ev.position().toPoint() if hasattr(ev, "position") else ev.pos()).row()
            if row < 0:
                row = self.count()
            self.track_dropped.emit(tid, row)
            ev.acceptProposedAction()
        else:
            super().dropEvent(ev)


class NextPanel(QFrame):
    """Big, prominent 'up next' card — separate from the rest of the queue so it's
    obvious where to drop a track to make it play next."""

    track_dropped = Signal(str)
    replan_requested = Signal()
    remove_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "NextPanel { background: #1c2027; border: 1px solid #3d434c; border-radius: 6px; }"
            "NextPanel[empty='true'] { border: 1.5px dashed #525a66; }"
        )
        self.setAcceptDrops(True)
        self._track_id: str | None = None

        v = QVBoxLayout(self)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(2)
        head = QLabel("UP NEXT")
        head.setStyleSheet("color: #8a93a3; font-size: 10pt; letter-spacing: 1px;")
        v.addWidget(head)

        row = QHBoxLayout()
        row.setSpacing(6)
        self.label = QLabel("(drop a track here)")
        self.label.setStyleSheet("font-size: 13pt; color: #ddd;")
        self.label.setWordWrap(True)
        row.addWidget(self.label, stretch=1)

        self.replan_btn = QPushButton("↻")
        self.replan_btn.setFixedSize(28, 28)
        self.replan_btn.setToolTip("Replan everything after this")
        self.replan_btn.clicked.connect(self.replan_requested.emit)
        row.addWidget(self.replan_btn)

        self.remove_btn = QPushButton("✕")
        self.remove_btn.setFixedSize(28, 28)
        self.remove_btn.setToolTip("Remove from queue")
        self.remove_btn.clicked.connect(self.remove_requested.emit)
        row.addWidget(self.remove_btn)

        v.addLayout(row)
        self._set_empty(True)

    def set_track(self, track_id: str | None, text: str) -> None:
        self._track_id = track_id
        self.label.setText(text if text else "(drop a track here)")
        self._set_empty(track_id is None)

    def _set_empty(self, empty: bool) -> None:
        self.setProperty("empty", "true" if empty else "false")
        self.replan_btn.setEnabled(not empty)
        self.remove_btn.setEnabled(not empty)
        # Force re-evaluation of dynamic property in stylesheet
        self.style().unpolish(self)
        self.style().polish(self)

    def dragEnterEvent(self, ev):  # noqa: N802
        if ev.mimeData().hasFormat(MIME_TRACK_ID):
            ev.acceptProposedAction()

    def dragMoveEvent(self, ev):  # noqa: N802
        if ev.mimeData().hasFormat(MIME_TRACK_ID):
            ev.acceptProposedAction()

    def dropEvent(self, ev):  # noqa: N802
        if ev.mimeData().hasFormat(MIME_TRACK_ID):
            tid = bytes(ev.mimeData().data(MIME_TRACK_ID)).decode("utf-8")
            self.track_dropped.emit(tid)
            ev.acceptProposedAction()


class QueueRow(QWidget):
    """A queue row with [N. Artist — Title] [duration] and per-row action buttons."""

    replan_requested = Signal(str)
    remove_requested = Signal(str)

    def __init__(self, text: str, track_id: str, duration_s: float | None = None,
                 parent: QWidget | None = None):
        super().__init__(parent)
        self._track_id = track_id
        self.setMinimumHeight(26)
        h = QHBoxLayout(self)
        h.setContentsMargins(4, 1, 4, 1)
        h.setSpacing(4)
        label = QLabel(text)
        label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        h.addWidget(label, stretch=1)
        if duration_s:
            dur_label = QLabel(_fmt_time(duration_s))
            dur_label.setStyleSheet("color: #888; font-family: monospace;")
            h.addWidget(dur_label)
        replan_btn = QPushButton("↻")
        replan_btn.setFixedSize(22, 22)
        replan_btn.setToolTip("Replan everything after this track")
        replan_btn.setStyleSheet("QPushButton { padding: 0; }")
        replan_btn.clicked.connect(lambda: self.replan_requested.emit(self._track_id))
        h.addWidget(replan_btn)
        remove_btn = QPushButton("✕")
        remove_btn.setFixedSize(22, 22)
        remove_btn.setToolTip("Remove from queue")
        remove_btn.setStyleSheet("QPushButton { padding: 0; }")
        remove_btn.clicked.connect(lambda: self.remove_requested.emit(self._track_id))
        h.addWidget(remove_btn)


class Window(QMainWindow):
    def __init__(self, idx: TrackIndex, proj: projmod.Projection):
        super().__init__()
        self.setWindowTitle("AI DJ")
        self.resize(1280, 820)

        self.idx = idx
        self.proj = proj
        self.planner = planpath.Planner(idx)
        self.player = playermod.Player(crossfade_enabled=False, crossfade_seconds=6.0)

        # State
        self.current: planpath.PlannedTrack | None = None
        self.upcoming: list[planpath.PlannedTrack] = []
        # Browser-style history. `history_list` is chronological; `history_cursor`
        # is the index of `current` within it. Prev decrements the cursor; auto-
        # advance increments. We only pull from `upcoming` when the cursor is at
        # the end (i.e. no forward history to walk through). This makes Prev →
        # auto-advance behave symmetrically without re-pushing the previous track.
        self.history_list: list[planpath.PlannedTrack] = []
        self.history_cursor: int = -1
        self.played_ever: set[str] = set()  # session-wide; the planner avoids these forever
        self.active_styles: set[str] = set()
        self.pinned_artists: set[str] = set()
        self.style_buttons: dict[str, QPushButton] = {}
        self._last_poll_advanced_for: str | None = None
        self._track_started_at: float | None = None

        self._all_artists: list[str] = []  # populated in _populate_search_model
        # Visualizer plugin state
        self._viz_classes = vizmod.registered_visualizers()
        self._active_viz: vizmod.Visualizer | None = None
        self._viz_activated_at: float = 0.0


        self._build_ui()
        self._populate_search_model()
        self._wire_timers()
        self._wire_shortcuts()
        # Activate the first registered visualizer as the default
        if self._viz_classes:
            first = self._viz_classes[0]
            self._viz_buttons[first.name].setChecked(True)
            self._activate_visualizer(first)

    # ---------- UI construction ----------

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(8, 8, 8, 8)

        # Top transport row
        transport = QHBoxLayout()
        self.prev_btn = QPushButton("⏮")
        self.prev_btn.setToolTip("Previous track (Backspace)")
        self.prev_btn.setFixedWidth(40)
        self.prev_btn.clicked.connect(self.prev_track)
        transport.addWidget(self.prev_btn)

        self.stop_btn = QPushButton("⏹")
        self.stop_btn.setToolTip("Stop")
        self.stop_btn.setFixedWidth(40)
        self.stop_btn.clicked.connect(self.stop_playback)
        transport.addWidget(self.stop_btn)

        self.play_btn = QPushButton("▶")
        self.play_btn.setToolTip("Play / Pause (Space)")
        self.play_btn.setFixedWidth(40)
        self.play_btn.setCheckable(True)
        self.play_btn.toggled.connect(self._on_play_toggle)
        transport.addWidget(self.play_btn)

        self.next_btn = QPushButton("⏭")
        self.next_btn.setToolTip("Next track")
        self.next_btn.setFixedWidth(40)
        self.next_btn.clicked.connect(self.skip)
        transport.addWidget(self.next_btn)

        transport.addSpacing(20)

        self.new_mix_btn = QPushButton("🎲  New mix")
        self.new_mix_btn.setToolTip("Pick a random seed and rebuild the path")
        self.new_mix_btn.clicked.connect(self.start_mix)
        transport.addWidget(self.new_mix_btn)

        transport.addSpacing(20)

        self.xfade_chk = QCheckBox("Crossfade")
        self.xfade_chk.toggled.connect(self._on_xfade_toggle)
        transport.addWidget(self.xfade_chk)
        self.xfade_slider = QSlider(Qt.Horizontal)
        self.xfade_slider.setRange(1, 15)
        self.xfade_slider.setValue(6)
        self.xfade_slider.setFixedWidth(120)
        self.xfade_slider.valueChanged.connect(self._on_xfade_slider)
        transport.addWidget(self.xfade_slider)
        self.xfade_label = QLabel("6s")
        transport.addWidget(self.xfade_label)
        transport.addStretch(1)
        outer.addLayout(transport)

        # Artists filter row (pin 1-N artists -> mix is restricted to their tracks)
        artists_row = QHBoxLayout()
        artists_row.addWidget(QLabel("Artists:"))
        self.artist_input = QLineEdit()
        self.artist_input.setPlaceholderText("type artist name and press Enter to pin…")
        self.artist_input.setMaximumWidth(280)
        self.artist_completer = QCompleter()
        self.artist_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.artist_completer.setFilterMode(Qt.MatchContains)
        self.artist_input.setCompleter(self.artist_completer)
        self.artist_input.returnPressed.connect(self._on_artist_input_submit)
        artists_row.addWidget(self.artist_input)
        self.artist_chips_holder = QWidget()
        self.artist_chips_layout = QHBoxLayout(self.artist_chips_holder)
        self.artist_chips_layout.setContentsMargins(0, 0, 0, 0)
        self.artist_chips_layout.setSpacing(4)
        artists_row.addWidget(self.artist_chips_holder, stretch=1)
        outer.addLayout(artists_row)

        # Style toggle row
        style_row = QHBoxLayout()
        style_row.addWidget(QLabel("Steer:"))
        for name in stylesmod.STYLES:
            color = stylesmod.STYLE_COLORS.get(name, "#888")
            btn = QPushButton(name)
            btn.setCheckable(True)
            btn.setStyleSheet(
                f"QPushButton {{ border-left: 4px solid {color}; padding: 3px 10px; }}"
                f"QPushButton:checked {{ background: {color}; color: #111; font-weight: bold; }}"
            )
            btn.toggled.connect(lambda on, n=name: self._on_style_toggle(n, on))
            style_row.addWidget(btn)
            self.style_buttons[name] = btn
        style_row.addStretch(1)
        outer.addLayout(style_row)

        # Middle split
        split = QHBoxLayout()
        outer.addLayout(split, stretch=1)

        # Left side: library on top, queue + now-playing on bottom.
        # Library — iTunes-style 3 column miller view (Artists | Albums | Tracks)
        lib_frame = QWidget()
        lib_layout = QHBoxLayout(lib_frame)
        lib_layout.setContentsMargins(0, 0, 0, 0)
        lib_layout.setSpacing(2)
        self.lib_artists = QListWidget()
        self.lib_albums = QListWidget()
        self.lib_tracks = TrackDragList()  # supports drag-out with track_id mime
        self.lib_artists.currentRowChanged.connect(self._on_lib_artist_changed)
        self.lib_albums.currentRowChanged.connect(self._on_lib_album_changed)
        self.lib_tracks.itemDoubleClicked.connect(self._on_lib_track_double_clicked)
        lib_layout.addWidget(self.lib_artists, stretch=1)
        lib_layout.addWidget(self.lib_albums, stretch=1)
        lib_layout.addWidget(self.lib_tracks, stretch=1)

        # Queue + Now Playing
        queue_frame = QWidget()
        queue_layout = QVBoxLayout(queue_frame)
        queue_layout.setContentsMargins(0, 4, 0, 0)
        self.now_label = QLabel("nothing playing")
        self.now_label.setStyleSheet("font-size: 14pt; font-weight: bold;")
        self.now_label.setWordWrap(True)
        queue_layout.addWidget(self.now_label)
        self.now_sub = QLabel("")
        self.now_sub.setStyleSheet("color: #aaa;")
        queue_layout.addWidget(self.now_sub)
        self.now_time = QLabel("")
        self.now_time.setStyleSheet("color: #ccc; font-family: monospace;")
        queue_layout.addWidget(self.now_time)
        queue_layout.addSpacing(8)
        self.next_panel = NextPanel()
        self.next_panel.track_dropped.connect(self._on_next_panel_drop)
        self.next_panel.replan_requested.connect(self._on_next_replan)
        self.next_panel.remove_requested.connect(self._on_next_remove)
        queue_layout.addWidget(self.next_panel)
        queue_layout.addSpacing(6)
        queue_layout.addWidget(QLabel("Queue · drop tracks from Library to add"))
        self.queue_list = QueueDropList()
        self.queue_list.setMinimumHeight(200)
        self.queue_list.itemDoubleClicked.connect(self._on_queue_double_clicked)
        self.queue_list.track_dropped.connect(self._on_queue_track_dropped)
        queue_layout.addWidget(self.queue_list, stretch=1)

        left_split = QSplitter(Qt.Vertical)
        left_split.addWidget(lib_frame)
        left_split.addWidget(queue_frame)
        left_split.setStretchFactor(0, 1)
        left_split.setStretchFactor(1, 1)
        left_split.setSizes([400, 400])
        left_split.setMinimumWidth(520)
        split.addWidget(left_split, stretch=2)

        # Right: 3D GL scene with mode selector on top
        right_w = QWidget()
        right_v = QVBoxLayout(right_w)
        right_v.setContentsMargins(0, 0, 0, 0)
        right_v.setSpacing(2)

        # Mode bar — buttons are built from whatever visualizers are registered
        mode_bar = QHBoxLayout()
        mode_bar.addWidget(QLabel("Visualizer:"))
        self.mode_group = QButtonGroup(self)
        self.mode_group.setExclusive(True)
        self._viz_buttons: dict[str, QPushButton] = {}
        for cls in self._viz_classes:
            btn = QPushButton(cls.name)
            btn.setCheckable(True)
            btn.setStyleSheet("QPushButton:checked { background: #2c5fff; color: white; }")
            self.mode_group.addButton(btn)
            mode_bar.addWidget(btn)
            self._viz_buttons[cls.name] = btn
        self.mode_group.buttonClicked.connect(self._on_mode_change)
        mode_bar.addStretch(1)
        right_v.addLayout(mode_bar)

        # The Window owns only the GL canvas. Each visualizer plugin is
        # responsible for adding its own scene items (scatter, terrain, paths,
        # markers) on activate and removing them on deactivate.
        self.gl_view = gl.GLViewWidget()
        self.gl_view.setBackgroundColor("#0a0c10")
        self.gl_view.setCameraPosition(distance=14, elevation=22, azimuth=45)

        right_v.addWidget(self.gl_view, stretch=1)
        split.addWidget(right_w, stretch=3)

        # Bottom: search box only
        bot = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Search artist or title, press Enter to add to queue…")
        self.search_completer = QCompleter()
        self.search_completer.setCaseSensitivity(Qt.CaseInsensitive)
        self.search_completer.setFilterMode(Qt.MatchContains)
        self.search_edit.setCompleter(self.search_completer)
        self.search_edit.returnPressed.connect(self._on_search_submit)
        bot.addWidget(self.search_edit, stretch=1)
        outer.addLayout(bot)

        hint = QLabel("LMB drag rotates · RMB drag pans · MMB drag / wheel zooms · ←/→ rotate · R reset · hover for track")
        hint.setStyleSheet("color: #666; font-size: 10pt;")
        hint.setAlignment(Qt.AlignRight)
        outer.addWidget(hint)

    def _wire_shortcuts(self) -> None:
        for seq, fn in [
            ("R", self._reset_view),
            ("Space", self.play_btn.toggle),
            ("Backspace", self.prev_track),
        ]:
            QShortcut(QKeySequence(seq), self).activated.connect(fn)

    def _reset_view(self) -> None:
        # Re-activate the current visualizer to reset whatever camera state it
        # owns (Free's default orbit, Follow's azimuth, etc.).
        if self._active_viz is not None:
            self._active_viz.activate(self._build_viz_context())

    def _wire_timers(self) -> None:
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(400)
        self.poll_timer.timeout.connect(self._poll)
        self.poll_timer.start()

        # Animation tick: 30 fps for camera/marker animation
        self.anim_timer = QTimer(self)
        self.anim_timer.setInterval(33)
        self.anim_timer.timeout.connect(self._anim_tick)
        self.anim_timer.start()
        self._anim_t = 0.0

    def _populate_search_model(self) -> None:
        """Load track summaries once and use them for search autocomplete,
        artist autocomplete, hover labels, and the iTunes-style library view."""
        rows = self.idx.all_track_summaries()
        labels = []
        self._label_to_id: dict[str, str] = {}
        self._id_to_label: dict[str, str] = {}
        artist_set: set[str] = set()
        from collections import defaultdict
        # artist -> album -> list[(title, track_id)]
        library: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
        for tid, artist, title, album, path in rows:
            label = f"{artist or '?'} — {title or Path(path).stem}"
            if label in self._label_to_id and self._label_to_id[label] != tid:
                label = f"{label}  [{Path(path).parent.name}]"
            self._label_to_id[label] = tid
            self._id_to_label[tid] = label
            labels.append(label)
            if artist:
                artist_set.add(artist)
            library[artist or "Unknown Artist"][album or "Unknown Album"].append(
                (title or Path(path).stem, tid)
            )
        self.search_completer.setModel(QStringListModel(sorted(labels)))
        self._all_artists = sorted(artist_set)
        self.artist_completer.setModel(QStringListModel(self._all_artists))

        # Library: store + populate the artists column. Albums and tracks are
        # filled in lazily when a row is selected.
        self._library = library
        self._library_artists = sorted(library.keys(), key=str.lower)
        self.lib_artists.addItems(self._library_artists)

        logger.info("search model loaded with %d tracks, %d unique artists",
                    len(labels), len(self._all_artists))

    # ---------- library miller-column handlers ----------

    def _on_lib_artist_changed(self, row: int) -> None:
        self.lib_albums.clear()
        self.lib_tracks.clear()
        if row < 0 or row >= len(self._library_artists):
            return
        artist = self._library_artists[row]
        albums = sorted(self._library[artist].keys(), key=str.lower)
        # Aggregator on top: "All Albums" shows the full artist discography
        all_item = QListWidgetItem("All Albums")
        all_item.setData(Qt.UserRole, "__ALL__")
        f = all_item.font()
        f.setItalic(True)
        all_item.setFont(f)
        self.lib_albums.addItem(all_item)
        for album in albums:
            item = QListWidgetItem(album)
            item.setData(Qt.UserRole, album)
            self.lib_albums.addItem(item)
        self.lib_albums.setCurrentRow(0)

    def _on_lib_album_changed(self, row: int) -> None:
        self.lib_tracks.clear()
        if row < 0:
            return
        artist_row = self.lib_artists.currentRow()
        if artist_row < 0 or artist_row >= len(self._library_artists):
            return
        artist = self._library_artists[artist_row]
        album_item = self.lib_albums.item(row)
        if album_item is None:
            return
        key = album_item.data(Qt.UserRole)
        if key == "__ALL__":
            for album in sorted(self._library[artist].keys(), key=str.lower):
                for title, tid in sorted(self._library[artist][album], key=lambda t: t[0].lower()):
                    item = QListWidgetItem(f"{album}  ·  {title}")
                    item.setData(Qt.UserRole, tid)
                    self.lib_tracks.addItem(item)
        else:
            for title, tid in sorted(self._library[artist][key], key=lambda t: t[0].lower()):
                item = QListWidgetItem(title)
                item.setData(Qt.UserRole, tid)
                self.lib_tracks.addItem(item)

    def _insert_track_at(self, track_id: str, insert_index: int) -> None:
        """Splice the given track into `upcoming` at `insert_index`. If it's
        already queued, we move it rather than duplicate."""
        rec = self.idx.client.retrieve(COLLECTION, [track_id], with_payload=True, with_vectors=False)
        if not rec:
            return
        track = planpath.planned_from_payload(track_id, rec[0].payload or {})
        if any(t.track_id == track_id for t in self.upcoming):
            self.upcoming = [t for t in self.upcoming if t.track_id != track_id]
        insert_index = max(0, min(insert_index, len(self.upcoming)))
        self.upcoming.insert(insert_index, track)
        self._refresh_queue()
        self._refresh_upcoming_path()

    def _on_queue_track_dropped(self, track_id: str, row: int) -> None:
        # Queue list shows upcoming[1:], so list-row N corresponds to upcoming index N+1.
        self._insert_track_at(track_id, row + 1)

    def _on_next_panel_drop(self, track_id: str) -> None:
        self._insert_track_at(track_id, 0)

    def _on_next_replan(self) -> None:
        if self.upcoming:
            self._on_queue_replan(self.upcoming[0].track_id)

    def _on_next_remove(self) -> None:
        if self.upcoming:
            self._on_queue_remove(self.upcoming[0].track_id)

    def _on_queue_replan(self, track_id: str) -> None:
        """User clicked ↻ on a queue row: keep this track and everything before
        it; drop everything after; refill from the clicked track as pivot."""
        for i, t in enumerate(self.upcoming):
            if t.track_id == track_id:
                self.upcoming = self.upcoming[: i + 1]
                self._refill_queue()
                self._refresh_queue()
                self._refresh_upcoming_path()
                return

    def _on_queue_remove(self, track_id: str) -> None:
        self.upcoming = [t for t in self.upcoming if t.track_id != track_id]
        self._refresh_queue()
        self._refresh_upcoming_path()

    def _on_lib_track_double_clicked(self, item: QListWidgetItem) -> None:
        tid = item.data(Qt.UserRole)
        if not tid:
            return
        rec = self.idx.client.retrieve(COLLECTION, [tid], with_payload=True, with_vectors=False)
        if not rec:
            return
        track = planpath.planned_from_payload(tid, rec[0].payload or {})
        self._play_track(track)
        self._refill_queue()
        self._refresh_all()

# ---------- state transitions ----------

    def start_mix(self) -> None:
        """Generate a fresh path and switch to it immediately. The currently
        playing track is replaced — clicking "New mix" while something is
        playing means the user wants a different song, not more of the same."""
        self.upcoming.clear()
        self.played_ever = set()
        self.history_list = []
        self.history_cursor = -1
        seed = self.planner.random_seed_track()
        self._play_track(seed, is_seed=True)
        self._refill_queue()
        self._refresh_all()

    def skip(self) -> None:
        self._advance_forward()

    def _advance_forward(self) -> None:
        """Move one step forward — either step through forward-history (after a
        Prev) or pull from `upcoming`. Used by Skip, the auto-advance poll, and
        crossfade-end."""
        if self.history_cursor < len(self.history_list) - 1:
            # We're behind the head of history (came back via Prev). Step forward.
            self.history_cursor += 1
            self._play_track(self.history_list[self.history_cursor], push_history=False)
        else:
            if not self.upcoming:
                self._refill_queue()
                if not self.upcoming:
                    return
            nxt = self.upcoming.pop(0)
            self._play_track(nxt, push_history=True)
        self._refill_queue()
        self._refresh_all()

    def prev_track(self) -> None:
        """Jump back one step in history. No-op if we're already at the start."""
        if self.history_cursor <= 0:
            return
        self.history_cursor -= 1
        self._play_track(self.history_list[self.history_cursor], push_history=False)
        self._refresh_all()

    def stop_playback(self) -> None:
        try:
            self.player.pause()
        except Exception:  # noqa: BLE001
            pass
        self._reflect_play_state(playing=False)

    def _play_track(self, track: planpath.PlannedTrack, *, push_history: bool = True,
                    is_seed: bool = False) -> None:  # noqa: ARG002
        """Play `track` and (optionally) record it in history.

        push_history=True (the default): branch the timeline from the current
        cursor — anything past it is truncated, then `track` is appended.
        push_history=False: caller is navigating an existing history slot
        (Prev / forward step after Prev), so just sync `current` without
        touching the list."""
        if push_history:
            # Truncate any forward-history (anything after the cursor)
            if self.history_cursor < len(self.history_list) - 1:
                del self.history_list[self.history_cursor + 1:]
            self.history_list.append(track)
            self.history_cursor = len(self.history_list) - 1
        self.current = track
        self.played_ever.add(track.track_id)
        self._last_poll_advanced_for = None
        self._track_started_at = time.monotonic()
        self._reflect_play_state(playing=True)
        try:
            self.player.play(track.path)
        except Exception as e:  # noqa: BLE001
            logger.exception("player.play failed for %s: %s", track.path, e)

    def _reflect_play_state(self, playing: bool) -> None:
        """Sync the play/pause button label with reality without re-triggering the toggle."""
        self.play_btn.blockSignals(True)
        self.play_btn.setChecked(playing)
        self.play_btn.setText("⏸" if playing else "▶")
        self.play_btn.blockSignals(False)

    def _refill_queue(self) -> None:
        """Top up `upcoming` to QUEUE_TARGET in small chunks scheduled on the Qt
        event loop. This keeps the UI responsive when each plan_next round-trip
        to Qdrant takes 50–100 ms (a 50-track refill on the UI thread froze
        noticeably, especially with the artist filter)."""
        if self.current is None:
            return
        QTimer.singleShot(0, self._refill_chunk)

    REFILL_CHUNK = 5

    def _refill_chunk(self) -> None:
        if self.current is None or len(self.upcoming) >= QUEUE_TARGET:
            return
        pivot = self.upcoming[-1] if self.upcoming else self.current
        exclude = set(self.played_ever) | {t.track_id for t in self.upcoming} | {self.current.track_id}
        added = 0
        while added < self.REFILL_CHUNK and len(self.upcoming) < QUEUE_TARGET:
            avoid = self._recent_artists()
            # Recent context = the few tracks just before the pivot in the
            # playback chain (history up to current + queued so far). Lets
            # the planner pick something sensible relative to the recent
            # vibe rather than only the immediate pivot.
            chain = self.history_list[: self.history_cursor + 1] + self.upcoming
            ctx_ids = tuple(t.track_id for t in chain[-4:-1])  # 3 tracks before pivot
            nxt = self.planner.plan_next(
                pivot, exclude=exclude, avoid_artists=avoid, context_ids=ctx_ids,
            )
            if nxt is None:
                break
            self.upcoming.append(nxt)
            exclude.add(nxt.track_id)
            pivot = nxt
            added += 1
        if added:
            self._refresh_queue()
            self._refresh_upcoming_path()
        if len(self.upcoming) < QUEUE_TARGET and added > 0:
            QTimer.singleShot(0, self._refill_chunk)

    def _recent_artists(self, n: int = RECENT_ARTISTS_AVOID) -> set[str]:
        """Last n artists in playback order before the next-to-plan slot."""
        if self.upcoming:
            recent = [t.artist for t in self.upcoming[-n:] if t.artist]
        elif self.current and self.current.artist:
            recent = [self.current.artist]
        else:
            recent = []
        return set(recent)

    def _on_style_toggle(self, name: str, on: bool) -> None:
        if on:
            self.active_styles.add(name)
        else:
            self.active_styles.discard(name)
        self.planner.set_styles(self.active_styles)
        # Replan upcoming against new styles (keep currently playing intact)
        self.upcoming.clear()
        self._refill_queue()
        self._refresh_queue()
        self._refresh_map_highlights()
        self._refresh_upcoming_path()

    def _on_queue_double_clicked(self, item: QListWidgetItem) -> None:
        row = self.queue_list.row(item)
        if row < 0 or row >= len(self.upcoming):
            return
        # Skipped-over tracks go straight into played_ever so they don't recur,
        # but we keep the rest of the queue intact (only top up the tail).
        for t in self.upcoming[:row]:
            self.played_ever.add(t.track_id)
        jumped = self.upcoming[row]
        self.upcoming = self.upcoming[row + 1:]
        self._play_track(jumped)
        self._refill_queue()  # only appends; existing upcoming entries are untouched
        self._refresh_all()

    def _on_play_toggle(self, playing: bool) -> None:
        if playing:
            if self.current is None:
                # No track loaded yet — start a fresh mix (uses whatever
                # styles + artists the user has already toggled on).
                self.start_mix()
            else:
                self.player.resume()
            self.play_btn.setText("⏸")
        else:
            self.player.pause()
            self.play_btn.setText("▶")

    def _on_xfade_toggle(self, on: bool) -> None:
        self.player.set_crossfade(on, self.xfade_slider.value())

    def _on_xfade_slider(self, v: int) -> None:
        self.xfade_label.setText(f"{v}s")
        self.player.set_crossfade(self.xfade_chk.isChecked(), v)

    def _on_artist_input_submit(self) -> None:
        name = self.artist_input.text().strip()
        if not name:
            return
        canon = next((a for a in self._all_artists if a.lower() == name.lower()), None)
        if canon is None:
            # No exact match — first prefix match counts
            canon = next((a for a in self._all_artists if a.lower().startswith(name.lower())), None)
        if canon is None:
            return
        self.pinned_artists.add(canon)
        self.artist_input.clear()
        self._refresh_artist_chips()
        self._on_artist_filter_changed()

    def _refresh_artist_chips(self) -> None:
        # Remove existing chip widgets
        while self.artist_chips_layout.count():
            it = self.artist_chips_layout.takeAt(0)
            w = it.widget()
            if w:
                w.deleteLater()
        for artist in sorted(self.pinned_artists):
            chip = QPushButton(f"{artist}  ✕")
            chip.setCursor(Qt.PointingHandCursor)
            chip.setStyleSheet(
                "QPushButton { padding: 2px 10px; border-radius: 9px;"
                " background: #2c2f36; border: 1px solid #4a4f58; color: #ddd; }"
                "QPushButton:hover { background: #3a3f47; }"
            )
            chip.clicked.connect(lambda _checked=False, a=artist: self._unpin_artist(a))
            self.artist_chips_layout.addWidget(chip)
        self.artist_chips_layout.addStretch(1)

    def _unpin_artist(self, artist: str) -> None:
        if artist in self.pinned_artists:
            self.pinned_artists.discard(artist)
            self._refresh_artist_chips()
            self._on_artist_filter_changed()

    def _on_artist_filter_changed(self) -> None:
        self.planner.set_artists(self.pinned_artists)
        # Replan upcoming under the new constraint; current keeps playing.
        self.upcoming.clear()
        self._refill_queue()
        self._refresh_queue()
        self._refresh_upcoming_path()
        self._refresh_map_highlights()

    def _on_search_submit(self) -> None:
        label = self.search_edit.text().strip()
        if not label or label not in self._label_to_id:
            return
        track_id = self._label_to_id[label]
        rec = self.idx.client.retrieve(COLLECTION, [track_id], with_payload=True, with_vectors=False)
        if not rec:
            return
        track = planpath.planned_from_payload(track_id, rec[0].payload or {})
        self.upcoming.insert(0, track)
        # Trim queue from the back to stay near target length
        if len(self.upcoming) > QUEUE_TARGET + 2:
            self.upcoming = self.upcoming[: QUEUE_TARGET + 2]
        self.search_edit.clear()
        self._refresh_queue()
        self._refresh_upcoming_path()

    # ---------- polling ----------

    def _poll(self) -> None:
        # Update the time readout regardless of advance logic.
        if self.current is None:
            self.now_time.setText("")
            return
        pos = self.player.time_pos()
        dur = self.player.duration()
        rem = self.player.time_remaining()
        if dur is not None and pos is not None:
            self.now_time.setText(
                f"{_fmt_time(pos)} / {_fmt_time(dur)}    (-{_fmt_time(rem)})"
            )

        # Mpv silently fails to load empty/corrupt files (no exception, no
        # duration ever). If we've been "playing" for longer than the load
        # timeout and still have no duration, treat the track as broken and
        # skip to the next one — otherwise the queue stalls forever.
        if dur is None and self._track_started_at is not None:
            stalled_for = time.monotonic() - self._track_started_at
            if stalled_for > PLAY_LOAD_TIMEOUT_S and self._last_poll_advanced_for != self.current.track_id:
                logger.warning("track %s never loaded (stalled %.1fs) — skipping",
                               self.current.path, stalled_for)
                self._last_poll_advanced_for = self.current.track_id
                self._advance_forward()
                return

        if rem is None:
            return
        trigger_at = self.player.crossfade_seconds if self.player.crossfade_enabled else ADVANCE_MARGIN_S
        if rem <= trigger_at and self._last_poll_advanced_for != self.current.track_id:
            self._last_poll_advanced_for = self.current.track_id
            self._advance_forward()

    def _anim_tick(self) -> None:
        """30 fps tick: delegate to the active visualizer."""
        self._anim_t += 0.033
        if self._active_viz is not None:
            self._active_viz.tick(self._build_viz_context())

    # ---------- viz updates ----------

    def _refresh_all(self) -> None:
        self._refresh_now_label()
        self._refresh_queue()
        # Path/marker live inside the active visualizer's scene now.
        if self._active_viz is not None:
            self._active_viz.on_path_change(self._build_viz_context())

    def _refresh_now_label(self) -> None:
        if not self.current:
            self.now_label.setText("nothing playing")
            self.now_sub.setText("")
            return
        t = self.current
        self.now_label.setText(f"▶  {t.artist or '?'} — {t.title or Path(t.path).stem}")
        genre_bits = []
        if t.ai_genre:
            genre_bits.append(f"ai: {t.ai_genre}")
        if t.genre:
            genre_bits.append(t.genre)
        self.now_sub.setText(
            f"{t.album or ''}   ·   {(t.duration_s or 0):.0f}s   ·   "
            + (" / ".join(genre_bits) if genre_bits else "no genre")
        )

    def _refresh_queue(self) -> None:
        # Up-Next panel (upcoming[0])
        if self.upcoming:
            t0 = self.upcoming[0]
            dur_s = f"   {_fmt_time(t0.duration_s)}" if t0.duration_s else ""
            self.next_panel.set_track(
                t0.track_id,
                f"{t0.artist or '?'} — {t0.title or Path(t0.path).stem}{dur_s}",
            )
        else:
            self.next_panel.set_track(None, "")
        # Rest of queue (upcoming[1:]) in the list
        self.queue_list.clear()
        for i, t in enumerate(self.upcoming[1:], start=2):
            line = f"{i:>2}. {t.artist or '?'} — {t.title or Path(t.path).stem}"
            item = QListWidgetItem()
            item.setData(Qt.UserRole, t.track_id)
            item.setSizeHint(QSize(0, 28))
            self.queue_list.addItem(item)
            row = QueueRow(line, t.track_id, t.duration_s)
            row.replan_requested.connect(self._on_queue_replan)
            row.remove_requested.connect(self._on_queue_remove)
            self.queue_list.setItemWidget(item, row)

    def _refresh_map_highlights(self) -> None:
        """Filter changed — let the active visualizer recolor its scene."""
        if self._active_viz is not None:
            self._active_viz.on_filter_change(self._build_viz_context())

    def _refresh_upcoming_path(self) -> None:
        if self._active_viz is not None:
            self._active_viz.on_path_change(self._build_viz_context())

    # ---------- visualizer plugin glue ----------

    def _build_viz_context(self) -> vizmod.VisualizerContext:
        return vizmod.VisualizerContext(
            gl_view=self.gl_view,
            track_ids=self.proj.track_ids,
            genres=self.proj.genres,
            scores=self.proj.scores,
            umap_xyz=self.proj.xyz,
            current_track_id=self.current.track_id if self.current else None,
            upcoming_track_ids=tuple(t.track_id for t in self.upcoming),
            progress=self.player.progress() or 0.0,
            elapsed=max(0.0, self._anim_t - self._viz_activated_at),
            track_remaining=self.player.time_remaining(),
            track_duration=self.player.duration(),
            active_styles=set(self.active_styles),
            style_colors_hex=stylesmod.STYLE_COLORS,
            untagged_color_hex=stylesmod.UNTAGGED_COLOR,
        )

    def _activate_visualizer(self, cls: type[vizmod.Visualizer]) -> None:
        ctx = self._build_viz_context()
        if self._active_viz is not None:
            self._active_viz.deactivate(ctx)
        self._active_viz = cls()
        self._viz_activated_at = self._anim_t
        self._active_viz.activate(ctx)

    def _on_mode_change(self, btn: QPushButton) -> None:
        for cls in self._viz_classes:
            if cls.name == btn.text():
                self._activate_visualizer(cls)
                return

    def closeEvent(self, event) -> None:  # noqa: N802
        self.player.close()
        super().closeEvent(event)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    # Tell Windows this process is its own app, not "python.exe". Without
    # this the taskbar groups under Python and shows the Python icon — so
    # pinning the running window doesn't carry our identity. Must happen
    # before the QApplication is created.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("aidj.henris42")
        except Exception:  # noqa: BLE001
            pass

    app = QApplication(sys.argv)
    app.setApplicationName("AI DJ")
    app.setOrganizationName("henris42")

    # Window / taskbar icon. The SVG scales to all needed sizes; falls back to
    # the rasterized PNG if PySide's SVG plugin isn't available.
    project_root = Path(__file__).resolve().parents[3]
    for cand in (project_root / "assets" / "icon.svg",
                 project_root / "assets" / "icon-256.png"):
        if cand.exists():
            app.setWindowIcon(QIcon(str(cand)))
            break

    idx = TrackIndex()
    idx.ensure_collection()

    cache = projmod.CACHE_PATH
    if cache.exists():
        logger.info("loading cached projection from %s", cache)
        proj = projmod.Projection.load(cache)
    else:
        logger.info("fitting projection from Qdrant ...")
        proj = projmod.fit(idx)
        proj.save()

    w = Window(idx, proj)
    w.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
