from __future__ import annotations

from .base import VisualizerContext, register
from ._umap_scene import UmapSceneVisualizer


@register
class TopDownVisualizer(UmapSceneVisualizer):
    name = "Top-down"

    def _camera_on_activate(self, ctx: VisualizerContext) -> None:
        ctx.set_orbit(look_at=None, distance=20, elevation=88, azimuth=0)
