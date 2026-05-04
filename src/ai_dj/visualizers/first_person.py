from __future__ import annotations

import numpy as np

from .base import VisualizerContext, register
from ._umap_scene import UmapSceneVisualizer

MIN_OFFSET = 0.4
HANDOFF_START = 0.85
HANDOFF_BIAS = 0.6


@register
class FirstPersonVisualizer(UmapSceneVisualizer):
    name = "First-person"

    def _camera_tick(self, ctx: VisualizerContext) -> None:
        cur = self._current_xyz(ctx)
        if cur is None:
            return
        nxt = self._next_xyz(ctx)
        if nxt is None:
            nxt = cur + np.array([0.5, 0.0, 0.0], dtype=np.float32)
        t = float(ctx.progress)
        cam_pos = cur + (nxt - cur) * t
        look_at = nxt.astype(np.float32)
        further = self._further_xyz(ctx)
        if further is not None and t > HANDOFF_START:
            blend = (t - HANDOFF_START) / (1.0 - HANDOFF_START)
            look_at = nxt + (further - nxt) * (blend * HANDOFF_BIAS)
        if float(np.linalg.norm(cam_pos - look_at)) < MIN_OFFSET:
            forward = look_at - cur
            n = float(np.linalg.norm(forward))
            if n > 1e-6:
                cam_pos = look_at - (forward / n) * MIN_OFFSET
        ctx.look_at(target=look_at, from_position=cam_pos)
