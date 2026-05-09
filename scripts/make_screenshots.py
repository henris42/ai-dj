"""Launch the GUI, switch through every visualizer mode, save a screenshot
of each, then exit. Used to generate the images embedded in the README.

Requires the same prerequisites as a normal GUI launch: Qdrant up, the
collection populated, projection.npz cached. Output goes to
`assets/screenshots/<NN>-<mode>.png`.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from ai_dj import projection as projmod
from ai_dj.gui.app import Window
from ai_dj.index import TrackIndex


SHOT_DELAY_MS = 2500          # render time before grabbing
PER_MODE_MS = 4500            # total time spent in each mode (switch + render + grab)
WINDOW_W, WINDOW_H = 1400, 880

OUT_DIR = Path(__file__).resolve().parents[1] / "assets" / "screenshots"


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    app = QApplication(sys.argv)
    idx = TrackIndex()
    idx.ensure_collection()
    cache = projmod.CACHE_PATH
    proj = projmod.Projection.load(cache) if cache.exists() else projmod.fit(idx)

    w = Window(idx, proj)
    w.resize(WINDOW_W, WINDOW_H)
    w.show()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    modes = [cls.name for cls in w._viz_classes]
    print(f"will capture: {modes}")

    # Kick off a mix so the visualizers have a path to show.
    QTimer.singleShot(800, w.start_mix)

    def switch(mode_name: str) -> None:
        btn = w._viz_buttons.get(mode_name)
        if btn is None:
            return
        btn.setChecked(True)
        # Clicking the button triggers _on_mode_change, which activates the viz.
        for cls in w._viz_classes:
            if cls.name == mode_name:
                w._activate_visualizer(cls)
                break

    def capture(idx_i: int, mode_name: str) -> None:
        out = OUT_DIR / f"{idx_i:02d}-{mode_name.lower().replace(' ', '_')}.png"
        pix = w.grab()
        pix.save(str(out))
        print(f"saved {out}")

    schedule_t = 1500  # let the first start_mix settle a little
    for i, mode in enumerate(modes):
        QTimer.singleShot(schedule_t, lambda m=mode: switch(m))
        QTimer.singleShot(schedule_t + SHOT_DELAY_MS, lambda i=i, m=mode: capture(i, m))
        schedule_t += PER_MODE_MS

    QTimer.singleShot(schedule_t + 500, app.quit)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
