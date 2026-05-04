"""Visualizer plugin contract.

A `Visualizer` is a pluggable behaviour that renders the 3D world for the
duration that it's the active mode. Each visualizer owns *all* of its scene
items — the scatter cloud, path lines, current-track marker, terrain meshes,
particle systems, whatever — and is fully responsible for adding them to the
view on `activate` and removing them on `deactivate`.

This means visualizers don't share state. Switching from a UMAP-based mode
(Free / Follow / First-person / Top-down) to Landscape tears the UMAP scene
down and builds the fractal world from scratch.

The Window passes track metadata + the playback cursor through
`VisualizerContext` and delegates four lifecycle events to the active
visualizer:

  - activate(ctx)        — build the scene
  - deactivate(ctx)      — destroy the scene
  - on_path_change(ctx)  — current/upcoming track changed
  - on_filter_change(ctx)— style filter changed
  - tick(ctx)            — animation tick (~30 fps)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl


_REGISTRY: list[type["Visualizer"]] = []


def register(cls: type["Visualizer"]) -> type["Visualizer"]:
    if cls not in _REGISTRY:
        _REGISTRY.append(cls)
    return cls


def registered_visualizers() -> list[type["Visualizer"]]:
    return list(_REGISTRY)


@dataclass
class VisualizerContext:
    """Snapshot of the world handed to the active visualizer.

    Visualizers receive *track metadata* (ids, genres, scores) and a
    UMAP-projected position array; UMAP-based visualizers use that array
    directly, others (e.g. Landscape) ignore it and compute their own
    positions. There are *no* scene-item references here — each visualizer
    owns its own scene state."""
    gl_view: gl.GLViewWidget

    # Track metadata (parallel arrays of length N)
    track_ids: np.ndarray
    genres: np.ndarray
    scores: np.ndarray

    # The default UMAP-projected positions for the library. Visualizers that
    # build their own coordinate system (e.g. Landscape) MAY ignore this.
    umap_xyz: np.ndarray

    # Playback cursor
    current_track_id: str | None
    upcoming_track_ids: tuple[str, ...]
    progress: float           # 0..1 within the current track
    elapsed: float            # seconds since this visualizer activated
    track_remaining: float | None = None    # seconds left of current track, or None
    track_duration: float | None = None     # total seconds of current track, or None

    # Filter state
    active_styles: set[str] = field(default_factory=set)

    # Style palette so visualizers can color things consistently
    style_colors_hex: dict[str, str] = field(default_factory=dict)
    untagged_color_hex: str = "#3a3f47"

    # ---- camera helpers ----

    def set_orbit(self, look_at: np.ndarray | None, distance: float,
                  elevation: float, azimuth: float) -> None:
        if look_at is None:
            self.gl_view.setCameraPosition(distance=distance, elevation=elevation, azimuth=azimuth)
        else:
            self.gl_view.setCameraPosition(
                pos=pg.Vector(*[float(v) for v in look_at]),
                distance=distance, elevation=elevation, azimuth=azimuth,
            )

    def look_at(self, target: np.ndarray, from_position: np.ndarray) -> None:
        offset = (np.asarray(from_position, dtype=np.float32)
                  - np.asarray(target, dtype=np.float32))
        dist = float(np.linalg.norm(offset))
        if dist < 1e-3:
            dist = 0.01
            offset = np.array([0.01, 0.0, 0.0], dtype=np.float32)
        u = offset / dist
        elev_deg = float(np.degrees(np.arcsin(float(np.clip(u[2], -1.0, 1.0)))))
        azim_deg = float(np.degrees(np.arctan2(float(u[1]), float(u[0]))))
        self.gl_view.setCameraPosition(
            pos=pg.Vector(*[float(v) for v in target]),
            distance=dist, elevation=elev_deg, azimuth=azim_deg,
        )


class Visualizer:
    """Base class for visualizer plugins."""

    name: str = ""

    def activate(self, ctx: VisualizerContext) -> None:
        """Build the scene. Add GL items here and remember the handles so
        deactivate can take them out again."""

    def deactivate(self, ctx: VisualizerContext) -> None:
        """Tear down whatever was built in activate."""

    def on_path_change(self, ctx: VisualizerContext) -> None:
        """Called when the current/upcoming tracks change — refresh path
        line, current-track marker, etc."""

    def on_filter_change(self, ctx: VisualizerContext) -> None:
        """Called when the user toggles a style filter — recolor / re-size."""

    def tick(self, ctx: VisualizerContext) -> None:
        """30 fps animation tick while active."""
