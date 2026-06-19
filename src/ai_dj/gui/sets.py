"""Sets viewer — list, load, delete saved sets.

A *set* is a captured journey through the library (see ai_dj.sets). The
parent Window emits the current played history + queue into save_set when
the Save button on the planner is clicked; this viewer is the inverse
operation: pick a previously-walked path and replay it.

Replay semantics:
  - Loading a set emits `set_loaded(track_ids)` to the parent. The
    planner clears its history/queue, plays the first id as the seed,
    and queues the rest in order. From there the user can let it play
    out, skip forward, or branch off by toggling Steer / changing the
    DJ — exactly like a freshly-played path.
"""
from __future__ import annotations

import time

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import sets as sets_mod


def _fmt_when(epoch: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M", time.localtime(epoch))


class SetsWindow(QMainWindow):
    """Non-modal viewer for saved sets."""

    set_loaded = Signal(list)            # list[str] of track ids
    set_deleted = Signal(str)            # set id

    def __init__(self, parent: "QWidget | None" = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("AI DJ — Sets")
        self.resize(720, 500)

        central = QWidget()
        self.setCentralWidget(central)
        v = QVBoxLayout(central)
        v.setContentsMargins(12, 10, 12, 10)
        v.setSpacing(6)

        head = QLabel("<h3 style='margin:0'>Sets</h3>")
        head.setTextFormat(Qt.RichText)
        v.addWidget(head)

        intro = QLabel("Journeys you've walked. Double-click one to replay.")
        intro.setStyleSheet("color: #888;")
        v.addWidget(intro)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(("Name", "When", "Tracks"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setSortingEnabled(False)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        h = self.table.horizontalHeader()
        h.setSectionResizeMode(0, QHeaderView.Stretch)
        h.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        h.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.doubleClicked.connect(lambda *_: self._on_load())
        v.addWidget(self.table, stretch=1)

        row = QHBoxLayout()
        self.load_btn = QPushButton("Replay set")
        self.load_btn.clicked.connect(self._on_load)
        row.addWidget(self.load_btn)
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(self._on_delete)
        row.addWidget(self.delete_btn)
        row.addStretch(1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        row.addWidget(close_btn)
        v.addLayout(row)

        self._reload()

    def _reload(self) -> None:
        records = sets_mod.list_sets()
        self.table.setRowCount(len(records))
        for r, rec in enumerate(records):
            name_it = QTableWidgetItem(rec.name)
            name_it.setData(Qt.UserRole, rec.id)
            self.table.setItem(r, 0, name_it)
            self.table.setItem(r, 1, QTableWidgetItem(_fmt_when(rec.created_at)))
            self.table.setItem(r, 2, QTableWidgetItem(f"{len(rec.track_ids):,}"))
        self.load_btn.setEnabled(len(records) > 0)
        self.delete_btn.setEnabled(len(records) > 0)

    # ---- actions ----
    def _selected_id(self) -> "str | None":
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        return item.data(Qt.UserRole) if item else None

    def _on_load(self) -> None:
        sid = self._selected_id()
        if not sid:
            return
        rec = sets_mod.load_set(sid)
        if not rec or not rec.track_ids:
            QMessageBox.warning(self, "Empty set",
                                "That set has no tracks to replay.")
            return
        self.set_loaded.emit(list(rec.track_ids))
        self.close()

    def _on_delete(self) -> None:
        sid = self._selected_id()
        if not sid:
            return
        ok = QMessageBox.question(
            self, "Delete set",
            "Delete this set? The tracks themselves stay in your library.",
        )
        if ok != QMessageBox.Yes:
            return
        sets_mod.delete_set(sid)
        self.set_deleted.emit(sid)
        self._reload()
