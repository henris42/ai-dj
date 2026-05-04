"""AI DJ — music-library path planner + player."""
from .paths import configure_libmpv_dll as _configure_libmpv_dll

_configure_libmpv_dll()
del _configure_libmpv_dll
