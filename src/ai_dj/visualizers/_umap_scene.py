"""Shared base for UMAP-driven visualizers.

Free / Follow / First-person / Top-down all render the same scene — the
UMAP-projected scatter, the path line, the current marker — they only differ
in how the camera moves. This class owns those scene items so subclasses just
implement `_camera_tick` (and optionally `_camera_on_activate`).

Landscape doesn't use this; it builds its own scene from fractal terrain.
"""
from __future__ import annotations

import numpy as np
import pyqtgraph.opengl as gl

from .base import Visualizer, VisualizerContext


def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


# Sizing
DOT_MIN_SIZE = 2.5
DOT_MAX_SIZE = 9.0
DOT_ACTIVE_BOOST = 1.6
DOT_DIMMED_SIZE = 1.8
SCORE_LO = 4.0
SCORE_HI = 13.0
DOT_ALPHA_DEFAULT = 0.85
DOT_ALPHA_DIMMED = 0.15


class UmapSceneVisualizer(Visualizer):
    """Base for visualizers that render the standard UMAP scene."""

    def __init__(self) -> None:
        self.grid: gl.GLGridItem | None = None
        self.scatter: gl.GLScatterPlotItem | None = None
        self.path_line: gl.GLLinePlotItem | None = None
        self.upcoming_dots: gl.GLScatterPlotItem | None = None
        self.marker: gl.GLScatterPlotItem | None = None

    # ---------- lifecycle ----------

    def activate(self, ctx: VisualizerContext) -> None:
        self.grid = gl.GLGridItem()
        self.grid.setSize(20, 20)
        self.grid.setSpacing(1, 1)
        self.grid.setColor((255, 255, 255, 25))
        ctx.gl_view.addItem(self.grid)

        colors, sizes = self._color_size(ctx)
        self.scatter = gl.GLScatterPlotItem(
            pos=ctx.umap_xyz, color=colors, size=sizes, pxMode=True,
        )
        self.scatter.setGLOptions("translucent")
        ctx.gl_view.addItem(self.scatter)

        self.path_line = gl.GLLinePlotItem(
            color=(1.0, 1.0, 1.0, 0.7), width=2.0, antialias=True, mode="line_strip",
        )
        ctx.gl_view.addItem(self.path_line)

        self.upcoming_dots = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(1.0, 0.95, 0.4, 0.95), size=10, pxMode=True,
        )
        ctx.gl_view.addItem(self.upcoming_dots)

        self.marker = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(1.0, 0.25, 0.25, 1.0), size=22, pxMode=True,
        )
        ctx.gl_view.addItem(self.marker)

        self._camera_on_activate(ctx)
        self.on_path_change(ctx)

    def deactivate(self, ctx: VisualizerContext) -> None:
        for item in (self.grid, self.scatter, self.path_line, self.upcoming_dots, self.marker):
            if item is not None:
                try:
                    ctx.gl_view.removeItem(item)
                except Exception:  # noqa: BLE001
                    pass
        self.grid = self.scatter = self.path_line = self.upcoming_dots = self.marker = None

    def on_path_change(self, ctx: VisualizerContext) -> None:
        if self.path_line is None or self.upcoming_dots is None or self.marker is None:
            return
        coords = self._chain_xyz(ctx)
        if len(coords) >= 2:
            self.path_line.setData(pos=coords)
        else:
            self.path_line.setData(pos=np.zeros((0, 3), dtype=np.float32))
        if len(coords) > 1:
            self.upcoming_dots.setData(pos=coords[1:])
        else:
            self.upcoming_dots.setData(pos=np.zeros((0, 3), dtype=np.float32))
        if len(coords) > 0:
            self.marker.setData(pos=coords[0:1])
        else:
            self.marker.setData(pos=np.zeros((0, 3), dtype=np.float32))

    def on_filter_change(self, ctx: VisualizerContext) -> None:
        if self.scatter is None:
            return
        colors, sizes = self._color_size(ctx)
        self.scatter.setData(pos=ctx.umap_xyz, color=colors, size=sizes)

    def tick(self, ctx: VisualizerContext) -> None:
        # Pulse the current-track marker, then let the subclass move the camera.
        if self.marker is not None and ctx.current_track_id is not None:
            size = 22 + 5 * float(np.sin(ctx.elapsed * 4.0))
            self.marker.setData(size=size)
        self._camera_tick(ctx)

    # ---------- subclass hooks ----------

    def _camera_on_activate(self, ctx: VisualizerContext) -> None:
        """Override to set the initial camera when the visualizer becomes active."""

    def _camera_tick(self, ctx: VisualizerContext) -> None:
        """Override to update the camera every frame."""

    # ---------- shared scene helpers ----------

    def _xyz_for(self, ctx: VisualizerContext, track_id: str) -> np.ndarray | None:
        idx = np.where(ctx.track_ids == track_id)[0]
        if len(idx) == 0:
            return None
        return ctx.umap_xyz[idx[0]]

    def _chain_xyz(self, ctx: VisualizerContext) -> np.ndarray:
        if ctx.current_track_id is None:
            return np.zeros((0, 3), dtype=np.float32)
        chain = [ctx.current_track_id, *ctx.upcoming_track_ids]
        out = []
        for tid in chain:
            xyz = self._xyz_for(ctx, tid)
            if xyz is not None:
                out.append(xyz)
        if not out:
            return np.zeros((0, 3), dtype=np.float32)
        return np.asarray(out, dtype=np.float32)

    def _color_size(self, ctx: VisualizerContext):
        n = len(ctx.genres)
        colors = np.zeros((n, 4), dtype=np.float32)
        s = ctx.scores
        norm = np.clip((s - SCORE_LO) / max(SCORE_HI - SCORE_LO, 1e-6), 0.0, 1.0)
        base = (DOT_MIN_SIZE + norm * (DOT_MAX_SIZE - DOT_MIN_SIZE)).astype(np.float32)
        sizes = base.copy()
        for i, g in enumerate(ctx.genres):
            hex_color = ctx.style_colors_hex.get(g, ctx.untagged_color_hex) if g else ctx.untagged_color_hex
            r, gg, b = _hex_to_rgb01(hex_color)
            if not ctx.active_styles:
                colors[i] = (r, gg, b, DOT_ALPHA_DEFAULT)
            elif g in ctx.active_styles:
                colors[i] = (r, gg, b, DOT_ALPHA_DEFAULT)
                sizes[i] = base[i] * DOT_ACTIVE_BOOST
            else:
                colors[i] = (r, gg, b, DOT_ALPHA_DIMMED)
                sizes[i] = DOT_DIMMED_SIZE
        return colors, sizes

    # current track xyz (used by camera modes)
    def _current_xyz(self, ctx: VisualizerContext) -> np.ndarray | None:
        if ctx.current_track_id is None:
            return None
        return self._xyz_for(ctx, ctx.current_track_id)

    def _next_xyz(self, ctx: VisualizerContext) -> np.ndarray | None:
        if not ctx.upcoming_track_ids:
            return None
        return self._xyz_for(ctx, ctx.upcoming_track_ids[0])

    def _further_xyz(self, ctx: VisualizerContext) -> np.ndarray | None:
        if len(ctx.upcoming_track_ids) < 2:
            return None
        return self._xyz_for(ctx, ctx.upcoming_track_ids[1])
