from __future__ import annotations

from .base import VisualizerContext, register
from ._umap_scene import UmapSceneVisualizer


@register
class FollowVisualizer(UmapSceneVisualizer):
    name = "Follow"
    distance = 4.0
    elevation = 18
    azimuth_drift_per_tick = 0.35

    def __init__(self) -> None:
        super().__init__()
        self._azimuth = 45.0

    def _camera_on_activate(self, ctx: VisualizerContext) -> None:
        self._azimuth = 45.0

    def _camera_tick(self, ctx: VisualizerContext) -> None:
        cur = self._current_xyz(ctx)
        if cur is None:
            return
        nxt = self._next_xyz(ctx)
        if nxt is None:
            nxt = cur
        look_at = cur + (nxt - cur) * float(ctx.progress)
        self._azimuth = (self._azimuth + self.azimuth_drift_per_tick) % 360
        ctx.set_orbit(look_at=look_at, distance=self.distance,
                      elevation=self.elevation, azimuth=self._azimuth)
