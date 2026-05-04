"""Audio playback with optional linear crossfade.

Two python-mpv players alternate as active/deck so we can ramp volumes between
them for crossfades. When crossfade is disabled we just stop one and start the
next on the same player, which gives gapless back-to-back playback."""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

import mpv

from .paths import resolve_for_player

logger = logging.getLogger(__name__)


class Player:
    """Threaded crossfading player. Thread-safe for the small set of ops used by the
    GUI: `play`, `pause`, `resume`, `skip`, `set_crossfade`, `set_volume`."""

    def __init__(
        self,
        crossfade_seconds: float = 6.0,
        crossfade_enabled: bool = False,
        on_track_start: Callable[[str], None] | None = None,
        on_track_end: Callable[[str], None] | None = None,
    ):
        self.crossfade_seconds = crossfade_seconds
        self.crossfade_enabled = crossfade_enabled
        self._on_start = on_track_start
        self._on_end = on_track_end

        self._deck_a = mpv.MPV(ytdl=False, audio_display="no", vid="no")
        self._deck_b = mpv.MPV(ytdl=False, audio_display="no", vid="no")
        self._active = self._deck_a
        self._standby = self._deck_b

        self._current_path: str | None = None
        self._lock = threading.Lock()
        self._stop_flag = threading.Event()

    def close(self) -> None:
        self._stop_flag.set()
        self._deck_a.terminate()
        self._deck_b.terminate()

    def set_crossfade(self, enabled: bool, seconds: float | None = None) -> None:
        with self._lock:
            self.crossfade_enabled = enabled
            if seconds is not None:
                self.crossfade_seconds = seconds

    def set_volume(self, pct: float) -> None:
        with self._lock:
            self._active.volume = pct

    def pause(self) -> None:
        self._active.pause = True

    def resume(self) -> None:
        self._active.pause = False

    def skip(self, next_path: str) -> None:
        """Start `next_path` immediately, skipping the rest of the current track.
        Uses a short fade-out+fade-in if crossfade is enabled, else hard cut."""
        self.play(next_path)

    def play(self, path: str) -> None:
        """Begin a track. If something is already playing, crossfade (or hard cut).

        For non-crossfade plays we don't block on `wait_until_playing` — mpv
        decodes the file asynchronously and the GUI's poll timer can pick up
        `time_remaining()` once it's ready. This keeps the UI thread responsive
        when the user mashes Next or starts a new mix while disks are slow."""
        local_path = resolve_for_player(path)
        with self._lock:
            if self._current_path is None or not self.crossfade_enabled:
                self._active.volume = 100
                self._active.play(local_path)
                self._current_path = path
                if self._on_start:
                    self._on_start(path)
                return

            # Crossfade: standby takes the new track at volume 0, ramp up while
            # active ramps down, then swap roles. Here we DO need the standby
            # actually playing before the fade ramp starts.
            self._standby.volume = 0
            self._standby.play(local_path)
            self._standby.wait_until_playing()
            self._fade(self._active, self._standby, self.crossfade_seconds)
            self._active, self._standby = self._standby, self._active
            self._current_path = path
            if self._on_start:
                self._on_start(path)

    def _fade(self, out: mpv.MPV, in_: mpv.MPV, seconds: float, steps: int = 30) -> None:
        step_dt = seconds / steps
        for i in range(1, steps + 1):
            frac = i / steps
            out.volume = max(0, 100 * (1 - frac))
            in_.volume = min(100, 100 * frac)
            if self._stop_flag.wait(step_dt):
                return
        out.stop()
        out.volume = 100  # reset for next use

    def time_remaining(self) -> float | None:
        """Seconds until the current track ends, or None if unknown."""
        try:
            pos = self._active.time_pos
            dur = self._active.duration
            if pos is None or dur is None:
                return None
            return max(0.0, dur - pos)
        except Exception:  # noqa: BLE001
            return None

    def time_pos(self) -> float | None:
        try:
            return self._active.time_pos
        except Exception:  # noqa: BLE001
            return None

    def duration(self) -> float | None:
        try:
            return self._active.duration
        except Exception:  # noqa: BLE001
            return None

    def progress(self) -> float | None:
        """Fractional playback position in [0, 1], or None if unknown."""
        pos = self.time_pos()
        dur = self.duration()
        if pos is None or dur is None or dur <= 0:
            return None
        return max(0.0, min(1.0, pos / dur))

    @property
    def current_path(self) -> str | None:
        return self._current_path
