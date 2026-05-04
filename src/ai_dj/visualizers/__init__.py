"""Visualizer plugin system.

A visualizer drives the 3D scene while it's the active mode (mostly camera
behaviour for now, but scene-level effects can hook in too). Built-ins live in
this package; new ones can be added by dropping a module in here that defines a
subclass of `Visualizer` decorated with `@register`. They will be picked up
automatically at startup.
"""
from __future__ import annotations

import importlib
import pkgutil

from .base import Visualizer, VisualizerContext, register, registered_visualizers

# Auto-import every submodule so its `@register`-decorated classes self-register.
for _, _name, _ in pkgutil.iter_modules(__path__):
    if not _name.startswith("_") and _name != "base":
        importlib.import_module(f"{__name__}.{_name}")

__all__ = ["Visualizer", "VisualizerContext", "register", "registered_visualizers"]
