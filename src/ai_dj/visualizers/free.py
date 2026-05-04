from __future__ import annotations

from .base import VisualizerContext, register
from ._umap_scene import UmapSceneVisualizer


@register
class FreeVisualizer(UmapSceneVisualizer):
    name = "Free"

    def _camera_on_activate(self, ctx: VisualizerContext) -> None:
        ctx.set_orbit(look_at=None, distance=22, elevation=22, azimuth=45)
