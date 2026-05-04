"""Planet visualizer — fractal sphere world with airplane-style flight.

The world is an icosphere whose vertices are radially displaced by 3D ridged
multifractal noise. Tracks live on the surface at their UMAP-projected
direction so musical neighbours become geographic neighbours; the path
between consecutive tracks is the shortest great-circle arc.

The camera is a simulated airplane:
  * has explicit position + velocity vectors that integrate over real time
  * heading turns toward the next track at a bounded rate (no teleporting)
  * speed is calibrated so the plane reaches the next track when the song ends
  * a look-ahead probe detects upcoming terrain and the plane climbs to clear
"""
from __future__ import annotations

import logging
import math

import numpy as np
import pyqtgraph.opengl as gl
from PySide6.QtGui import QMatrix4x4, QVector3D

from .base import Visualizer, VisualizerContext, register

logger = logging.getLogger(__name__)


# ---------- planet ----------
# Big enough that the horizon is far compared to cruise altitude — a small
# planet feels like an asteroid; this RADIUS gives an Earth-ish horizon ratio.
RADIUS = 200.0
HEIGHT_SCALE = 18.0
TRACK_LIFT = 1.5
SUBDIVISIONS = 5             # 10 242 verts / 20 480 faces
OCTAVES = 6
LACUNARITY = 2.0
GAIN = 0.55
SEED = 0xA1D1

# ---------- sky ----------
SKY_BG = "#a8d4ff"
DEFAULT_BG = "#0a0c10"

# ---------- airplane ----------
# Plane has 3D position and velocity. Each tick it turns toward the next
# waypoint (next track) at a bounded rate, accelerates/decelerates so it
# arrives roughly when the song ends, and tracks an altitude above terrain
# via a smoothed PD controller. When the queue changes (track advance, drag-
# in, replan) only the *target* changes — the plane keeps flying with its
# existing position and velocity, then turns to the new heading.
INITIAL_ALT_OVER_TERRAIN = 5.0
CRUISE_SPEED = 12.0         # only used before the track's `remaining` time is known
MAX_SPEED = 80.0
ACCELERATION = 5.0          # max world-units/s² speed change (both accel and decel)
MAX_TURN_RATE = 0.45        # max heading change in radians/s
LOOK_AHEAD_DIST = 7.0       # camera looks this far ahead along velocity
# Altitude PD: acceleration-based, critically damped (KD = 2*sqrt(KP)).
# Linear PD analysis would predict no oscillation; the real source is
# terrain following with a *moving* target altitude that's discontinuous
# (sparse lookahead samples) and saturated v-clipping. Both are addressed:
#   - lookahead is dense and distance-based (continuous target);
#   - target falls slowly after a peak so the plane doesn't dive into the
#     valley behind it (asymmetric smoothing — the dominant exciter).
CRUISE_ALT_ABOVE_TERRAIN = 7.0
ALT_KP = 0.4
ALT_KD = 1.4                   # 2·√KP·1.1 — slightly over critical
ALT_MAX_VSPEED = 5.0           # caps how fast altitude can change visually
ALT_MAX_ACCEL = 3.5
LOOKAHEAD_DIST = 130.0         # wide enough that several peaks share a window
LOOKAHEAD_SAMPLES = 24
TARGET_FALL_TC = 4.0           # very slow target descent: glide, don't bob

# Genre→biome bands (kept just for the colour key on the terrain)
_DEEP_WATER = np.array([0.13, 0.45, 0.62], dtype=np.float32)
_SHORE      = np.array([0.32, 0.70, 0.78], dtype=np.float32)
_SAND       = np.array([0.95, 0.88, 0.55], dtype=np.float32)
_GRASS      = np.array([0.62, 0.70, 0.20], dtype=np.float32)
_FOREST     = np.array([0.30, 0.50, 0.20], dtype=np.float32)
_ROCK       = np.array([0.72, 0.70, 0.66], dtype=np.float32)
_SNOW       = np.array([0.99, 0.99, 1.00], dtype=np.float32)


# ---------- icosphere ----------

def _icosphere(subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
    t = (1 + 5 ** 0.5) / 2
    verts = [
        (-1,  t,  0), ( 1,  t,  0), (-1, -t,  0), ( 1, -t,  0),
        ( 0, -1,  t), ( 0,  1,  t), ( 0, -1, -t), ( 0,  1, -t),
        ( t,  0, -1), ( t,  0,  1), (-t,  0, -1), (-t,  0,  1),
    ]
    verts = [np.asarray(v, dtype=np.float64) / np.linalg.norm(v) for v in verts]
    faces: list[tuple[int, int, int]] = [
        (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
        (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
        (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
        (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
    ]
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}

        def midpoint(i: int, j: int) -> int:
            key = (i, j) if i < j else (j, i)
            if key in cache:
                return cache[key]
            m = (verts[i] + verts[j]) * 0.5
            m /= np.linalg.norm(m)
            verts.append(m)
            idx = len(verts) - 1
            cache[key] = idx
            return idx

        new_faces: list[tuple[int, int, int]] = []
        for a, b, c in faces:
            ab = midpoint(a, b); bc = midpoint(b, c); ca = midpoint(c, a)
            new_faces.extend([(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)])
        faces = new_faces
    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


# ---------- 3D noise ----------

def _value_noise_3d(rng: np.random.Generator, points: np.ndarray, freq: float) -> np.ndarray:
    grid_n = max(2, int(np.ceil(freq * 4)) + 2)
    grid = rng.standard_normal((grid_n, grid_n, grid_n)).astype(np.float32)
    p = (points * freq) % (grid_n - 1)
    i = p.astype(np.int32)
    f = p - i
    f = f * f * (3 - 2 * f)
    i1 = (i + 1) % (grid_n - 1)
    g000 = grid[i[:, 0], i[:, 1], i[:, 2]]
    g100 = grid[i1[:, 0], i[:, 1], i[:, 2]]
    g010 = grid[i[:, 0], i1[:, 1], i[:, 2]]
    g001 = grid[i[:, 0], i[:, 1], i1[:, 2]]
    g110 = grid[i1[:, 0], i1[:, 1], i[:, 2]]
    g101 = grid[i1[:, 0], i[:, 1], i1[:, 2]]
    g011 = grid[i[:, 0], i1[:, 1], i1[:, 2]]
    g111 = grid[i1[:, 0], i1[:, 1], i1[:, 2]]
    fx, fy, fz = f[:, 0], f[:, 1], f[:, 2]
    c00 = g000 * (1 - fx) + g100 * fx
    c01 = g001 * (1 - fx) + g101 * fx
    c10 = g010 * (1 - fx) + g110 * fx
    c11 = g011 * (1 - fx) + g111 * fx
    c0 = c00 * (1 - fy) + c10 * fy
    c1 = c01 * (1 - fy) + c11 * fy
    return c0 * (1 - fz) + c1 * fz


def _ridged_mf_3d(rng: np.random.Generator, points: np.ndarray) -> np.ndarray:
    out = np.zeros(len(points), dtype=np.float32)
    amp = 1.0
    freq = 1.5
    for _ in range(OCTAVES):
        layer = _value_noise_3d(rng, points, freq)
        ridge = (1.0 - np.abs(layer)) ** 2
        out += ridge * amp
        amp *= GAIN
        freq *= LACUNARITY
    out -= out.min()
    if out.max() > 1e-9:
        out /= out.max()
    return np.power(out, 1.2, dtype=np.float32)


def _color_for_elevation(z_norm: np.ndarray) -> np.ndarray:
    z = z_norm.astype(np.float32)
    out = np.zeros(z.shape + (4,), dtype=np.float32)
    out[..., 3] = 1.0
    rgb = out[..., :3]
    rgb[z < 0.10] = _DEEP_WATER
    m = (z >= 0.10) & (z < 0.16); rgb[m] = _SHORE
    m = (z >= 0.16) & (z < 0.22); rgb[m] = _SAND
    m = (z >= 0.22) & (z < 0.40); rgb[m] = _GRASS
    m = (z >= 0.40) & (z < 0.60); rgb[m] = _FOREST
    m = (z >= 0.60) & (z < 0.85); rgb[m] = _ROCK
    rgb[z >= 0.85] = _SNOW
    shade = 1.0 + 0.25 * z[..., None]
    out[..., :3] *= np.clip(shade, 0.8, 1.3)
    out[..., :3] = np.clip(out[..., :3], 0.0, 1.0)
    return out


# ---------- track placement ----------

def _place_tracks(unit_verts: np.ndarray, vert_displaced: np.ndarray,
                  umap_xyz: np.ndarray) -> np.ndarray:
    n = len(umap_xyz)
    norms = np.linalg.norm(umap_xyz, axis=1, keepdims=True)
    track_units = (umap_xyz / np.maximum(norms, 1e-9)).astype(np.float32)
    positions = np.zeros((n, 3), dtype=np.float32)
    chunk = 1024
    for start in range(0, n, chunk):
        sl = slice(start, min(start + chunk, n))
        dots = track_units[sl] @ unit_verts.T
        idx = np.argmax(dots, axis=1)
        for k, i in enumerate(idx):
            ground_alt = float(np.linalg.norm(vert_displaced[i]))
            positions[start + k] = track_units[start + k] * (ground_alt + TRACK_LIFT)
    return positions


# ---------- great-circle helpers ----------

def _slerp_unit(p0: np.ndarray, p1: np.ndarray, t: float) -> np.ndarray:
    p0 = p0.astype(np.float32)
    p1 = p1.astype(np.float32)
    d = float(np.clip(np.dot(p0, p1), -1.0, 1.0))
    if abs(d) > 0.9995:
        out = p0 + (p1 - p0) * t
        n = float(np.linalg.norm(out))
        return out / n if n > 1e-9 else p0
    omega = math.acos(d)
    s = math.sin(omega)
    return ((math.sin((1 - t) * omega) / s) * p0
            + (math.sin(t * omega) / s) * p1)


def _great_circle_points(p0_unit: np.ndarray, p1_unit: np.ndarray, steps: int = 24) -> np.ndarray:
    out = np.zeros((steps + 1, 3), dtype=np.float32)
    for k in range(steps + 1):
        out[k] = _slerp_unit(p0_unit, p1_unit, k / steps)
    return out


def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0], dtype=np.float32)
    return (v / n).astype(np.float32)


# ---------- track colours ----------

def _hex_to_rgb01(hex_color: str) -> tuple[float, float, float]:
    h = hex_color.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return (int(h[0:2], 16) / 255.0, int(h[2:4], 16) / 255.0, int(h[4:6], 16) / 255.0)


def _track_colors(genres: np.ndarray, style_colors: dict[str, str],
                  untagged: str, active_styles: set[str]) -> np.ndarray:
    n = len(genres)
    out = np.zeros((n, 4), dtype=np.float32)
    for i, g in enumerate(genres):
        hex_color = style_colors.get(g, untagged) if g else untagged
        r, gg, b = _hex_to_rgb01(hex_color)
        if not active_styles:
            a = 0.95
        elif g in active_styles:
            a = 0.95
        else:
            a = 0.18
        out[i] = (r, gg, b, a)
    return out


@register
class LandscapeVisualizer(Visualizer):
    name = "Planet"

    def __init__(self) -> None:
        # Scene items
        self._mesh: gl.GLMeshItem | None = None
        self._scatter: gl.GLScatterPlotItem | None = None
        self._path_line: gl.GLLinePlotItem | None = None
        self._upcoming_dots: gl.GLScatterPlotItem | None = None
        self._marker: gl.GLScatterPlotItem | None = None
        # World
        self._unit_verts: np.ndarray | None = None
        self._displaced: np.ndarray | None = None
        # Tracks
        self._track_xyz: np.ndarray | None = None
        self._track_unit: np.ndarray | None = None
        self._track_alt: np.ndarray | None = None
        self._id_to_idx: dict[str, int] = {}
        # Airplane state — persists across track changes; only the target
        # waypoint changes when the queue advances.
        self._plane_pos: np.ndarray | None = None
        self._plane_vel: np.ndarray | None = None
        self._smoothed_target_alt: float | None = None
        self._last_elapsed: float | None = None
        self._orig_view_matrix = None
        self._current_view_matrix = None

    # ---- helpers ----

    def _height_along(self, unit_dir: np.ndarray) -> float:
        if self._unit_verts is None or self._displaced is None:
            return RADIUS
        dots = self._unit_verts @ unit_dir
        v_idx = int(np.argmax(dots))
        return float(np.linalg.norm(self._displaced[v_idx]))

    def _xyz_for(self, track_id: str) -> np.ndarray | None:
        if self._track_xyz is None:
            return None
        i = self._id_to_idx.get(str(track_id))
        if i is None:
            return None
        return self._track_xyz[i]

    def _unit_for(self, track_id: str) -> np.ndarray | None:
        if self._track_unit is None:
            return None
        i = self._id_to_idx.get(str(track_id))
        if i is None:
            return None
        return self._track_unit[i]

    # ---- lifecycle ----

    def activate(self, ctx: VisualizerContext) -> None:
        ctx.gl_view.setBackgroundColor(SKY_BG)
        rng = np.random.default_rng(SEED)

        unit_verts, faces = _icosphere(SUBDIVISIONS)
        self._unit_verts = unit_verts
        z_norm = _ridged_mf_3d(rng, unit_verts)
        self._displaced = (unit_verts * (RADIUS + z_norm[:, None] * HEIGHT_SCALE)).astype(np.float32)
        logger.info("planet: %d verts / %d faces", len(unit_verts), len(faces))

        self._mesh = gl.GLMeshItem(
            meshdata=gl.MeshData(vertexes=self._displaced, faces=faces,
                                 vertexColors=_color_for_elevation(z_norm)),
            smooth=True, drawFaces=True, drawEdges=False, shader="shaded",
        )
        self._mesh.setGLOptions("opaque")
        ctx.gl_view.addItem(self._mesh)

        # Place tracks via UMAP, lying on the surface
        self._track_xyz = _place_tracks(unit_verts, self._displaced, ctx.umap_xyz)
        self._track_unit = (self._track_xyz / np.linalg.norm(self._track_xyz, axis=1, keepdims=True)).astype(np.float32)
        self._track_alt = np.linalg.norm(self._track_xyz, axis=1).astype(np.float32)
        self._id_to_idx = {str(tid): i for i, tid in enumerate(ctx.track_ids)}

        self._scatter = gl.GLScatterPlotItem(
            pos=self._track_xyz,
            color=_track_colors(ctx.genres, ctx.style_colors_hex, ctx.untagged_color_hex, ctx.active_styles),
            size=4.5, pxMode=True,
        )
        self._scatter.setGLOptions("translucent")
        ctx.gl_view.addItem(self._scatter)

        self._path_line = gl.GLLinePlotItem(
            color=(1.0, 0.92, 0.25, 1.0), width=5.0, antialias=True, mode="line_strip",
        )
        self._path_line.setGLOptions("opaque")
        ctx.gl_view.addItem(self._path_line)
        self._upcoming_dots = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(1.0, 0.95, 0.4, 0.95), size=12, pxMode=True,
        )
        ctx.gl_view.addItem(self._upcoming_dots)
        self._marker = gl.GLScatterPlotItem(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=(1.0, 0.3, 0.3, 1.0), size=24, pxMode=True,
        )
        ctx.gl_view.addItem(self._marker)

        # No track playing yet → orbital overview of the planet.
        self._plane_pos = None
        self._plane_vel = None
        self._smoothed_target_alt = None
        self._last_elapsed = None
        ctx.set_orbit(look_at=np.zeros(3, dtype=np.float32),
                      distance=RADIUS + HEIGHT_SCALE + 60, elevation=15, azimuth=45)

        # Override viewMatrix with our own (Qt's lookAt does what we want).
        self._orig_view_matrix = ctx.gl_view.viewMatrix
        self._current_view_matrix = ctx.gl_view.viewMatrix()
        ctx.gl_view.viewMatrix = lambda: self._current_view_matrix
        self.on_path_change(ctx)

    def deactivate(self, ctx: VisualizerContext) -> None:
        ctx.gl_view.setBackgroundColor(DEFAULT_BG)
        ctx.gl_view.opts["rotation"] = None
        # Restore pyqtgraph's default viewMatrix
        if getattr(self, "_orig_view_matrix", None) is not None:
            ctx.gl_view.viewMatrix = self._orig_view_matrix
            self._orig_view_matrix = None
        for item in (self._mesh, self._scatter, self._path_line,
                     self._upcoming_dots, self._marker):
            if item is not None:
                try:
                    ctx.gl_view.removeItem(item)
                except Exception:  # noqa: BLE001
                    pass
        self._mesh = self._scatter = self._path_line = None
        self._upcoming_dots = self._marker = None
        self._unit_verts = self._displaced = None
        self._track_xyz = self._track_unit = self._track_alt = None
        self._id_to_idx = {}
        self._plane_pos = self._plane_vel = None
        self._smoothed_target_alt = None
        self._last_elapsed = None
        self._current_view_matrix = None

    def on_path_change(self, ctx: VisualizerContext) -> None:
        if self._path_line is None or self._upcoming_dots is None or self._marker is None:
            return
        if ctx.current_track_id is None or self._track_unit is None or self._track_alt is None:
            self._path_line.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self._upcoming_dots.setData(pos=np.zeros((0, 3), dtype=np.float32))
            self._marker.setData(pos=np.zeros((0, 3), dtype=np.float32))
            return

        chain_ids = [ctx.current_track_id, *ctx.upcoming_track_ids][:24]
        # The path is the chain of waypoints; the lines between them follow
        # the planet's surface — for each intermediate point we sample the
        # terrain height directly (rather than linearly interpolating between
        # endpoint radii), so the path hugs mountains and dips into valleys.
        # Lifted enough that you can see the line when flying at cruise.
        PATH_LIFT = 2.5
        path_pts: list[np.ndarray] = []
        last_unit = None
        for tid in chain_ids:
            i = self._id_to_idx.get(str(tid))
            if i is None:
                continue
            u = self._track_unit[i]
            r = self._height_along(u) + PATH_LIFT
            if last_unit is None:
                path_pts.append(np.asarray(u * r, dtype=np.float32).reshape(1, 3))
            else:
                steps = 24
                arc = _great_circle_points(last_unit, u, steps=steps)
                radii = np.array(
                    [self._height_along(arc[k]) + PATH_LIFT for k in range(len(arc))],
                    dtype=np.float32,
                )
                path_pts.append((arc * radii[:, None])[1:])
            last_unit = u

        if path_pts:
            coords = np.concatenate(path_pts, axis=0).astype(np.float32)
            if len(coords) >= 2:
                self._path_line.setData(pos=coords)
            else:
                self._path_line.setData(pos=np.zeros((0, 3), dtype=np.float32))
        else:
            self._path_line.setData(pos=np.zeros((0, 3), dtype=np.float32))

        up_pos = []
        for tid in ctx.upcoming_track_ids[:12]:
            xyz = self._xyz_for(tid)
            if xyz is not None:
                up_pos.append(xyz)
        self._upcoming_dots.setData(
            pos=np.asarray(up_pos, dtype=np.float32) if up_pos else np.zeros((0, 3), dtype=np.float32)
        )

        cur_xyz = self._xyz_for(ctx.current_track_id)
        self._marker.setData(pos=cur_xyz.reshape(1, 3) if cur_xyz is not None
                             else np.zeros((0, 3), dtype=np.float32))

    def on_filter_change(self, ctx: VisualizerContext) -> None:
        if self._scatter is None or self._track_xyz is None:
            return
        self._scatter.setData(
            pos=self._track_xyz,
            color=_track_colors(ctx.genres, ctx.style_colors_hex,
                                ctx.untagged_color_hex, ctx.active_styles),
        )

    def tick(self, ctx: VisualizerContext) -> None:
        if self._marker is not None and ctx.current_track_id is not None:
            size = 24 + 5 * float(np.sin(ctx.elapsed * 4.0))
            self._marker.setData(size=size)

        if (self._track_unit is None or self._track_alt is None
                or ctx.current_track_id is None):
            return

        cur_xyz = self._xyz_for(ctx.current_track_id)
        if cur_xyz is None:
            return
        nxt_xyz = (self._xyz_for(ctx.upcoming_track_ids[0])
                   if ctx.upcoming_track_ids else cur_xyz)
        if nxt_xyz is None:
            nxt_xyz = cur_xyz

        # ---- airplane integration ----
        elapsed = float(ctx.elapsed)
        if self._last_elapsed is None:
            dt = 1.0 / 30.0
        else:
            dt = max(0.001, min(0.1, elapsed - self._last_elapsed))
        self._last_elapsed = elapsed

        # Initialize the plane the first frame we have a current track.
        # On subsequent ticks we never reinitialise — the plane keeps its
        # position and velocity even when the queue advances; only the
        # *target* (next waypoint) changes, and the heading turns toward it.
        if self._plane_pos is None or self._plane_vel is None:
            radial = _safe_normalize(cur_xyz)
            self._plane_pos = (cur_xyz + radial * INITIAL_ALT_OVER_TERRAIN).astype(np.float32)
            # Initial velocity should be TANGENT to the sphere (no radial
            # component) and at a speed pre-tuned for arrival at the next
            # waypoint when the song ends. Don't blast off with a cruise
            # speed — for nearby waypoints that overshoots in seconds.
            to_target = nxt_xyz - self._plane_pos
            radial_pos = _safe_normalize(self._plane_pos)
            tangent = to_target - float(np.dot(to_target, radial_pos)) * radial_pos
            forward0 = _safe_normalize(tangent)
            distance0 = float(np.linalg.norm(to_target))
            duration0 = ctx.track_duration if ctx.track_duration is not None else ctx.track_remaining
            if duration0 is not None and duration0 > 5.0:
                init_speed = float(np.clip(distance0 / duration0, 0.0, MAX_SPEED))
            else:
                init_speed = 0.0
            self._plane_vel = (forward0 * init_speed).astype(np.float32)

        # 1. Heading: turn current direction toward the next waypoint
        to_target = nxt_xyz - self._plane_pos
        dist_to_target = float(np.linalg.norm(to_target))
        desired_dir = to_target / max(dist_to_target, 1e-6)
        cur_speed = float(np.linalg.norm(self._plane_vel))
        cur_dir = (self._plane_vel / cur_speed) if cur_speed > 1e-6 else desired_dir.astype(np.float32)
        cos_a = float(np.clip(np.dot(cur_dir, desired_dir), -1.0, 1.0))
        angle = math.acos(cos_a)
        max_step = MAX_TURN_RATE * dt
        if angle > max_step and angle > 1e-6:
            new_dir = _slerp_unit(cur_dir, desired_dir.astype(np.float32), max_step / angle)
        else:
            new_dir = desired_dir.astype(np.float32)

        # 2. Speed: aim to arrive at the waypoint when the song ends.
        # `dist_to_target / remaining` is the exact speed needed; no minimum
        # — the plane decelerates to zero and rests at the waypoint until the
        # next target appears. Only use the timing info when it looks
        # reliable (>5s remaining); otherwise hold whatever speed we have so
        # mpv's first-frame zero-duration glitches can't make us blast off.
        remaining = ctx.track_remaining
        if remaining is not None and remaining > 5.0:
            target_speed = float(np.clip(dist_to_target / remaining, 0.0, MAX_SPEED))
        elif ctx.track_duration is not None and ctx.track_duration > 5.0:
            # Fallback for the first frames before remaining settles
            target_speed = float(np.clip(dist_to_target / ctx.track_duration, 0.0, MAX_SPEED))
        else:
            target_speed = cur_speed  # hold steady, don't suddenly accelerate
        delta_speed = float(np.clip(target_speed - cur_speed, -ACCELERATION * dt, ACCELERATION * dt))
        new_speed = max(0.0, cur_speed + delta_speed)
        tangential_vel = (new_dir * new_speed).astype(np.float32)

        # 3. Dense distance-based lookahead → continuous target altitude.
        plane_radius = float(np.linalg.norm(self._plane_pos))
        plane_radial_dir = (self._plane_pos / max(plane_radius, 1e-9)).astype(np.float32)
        heading_for_probe = _safe_normalize(tangential_vel)
        ground_max = self._height_along(plane_radial_dir)
        for k in range(1, LOOKAHEAD_SAMPLES + 1):
            d = (k / LOOKAHEAD_SAMPLES) * LOOKAHEAD_DIST
            sample_unit = _safe_normalize(self._plane_pos + heading_for_probe * d)
            ground_max = max(ground_max, self._height_along(sample_unit))
        raw_target_alt = ground_max + CRUISE_ALT_ABOVE_TERRAIN

        # Asymmetric smoothing: the target rises immediately when terrain
        # ahead is taller (so we always have time to climb) but falls
        # gradually after a peak passes — preventing the plane from diving
        # into the valley right behind a mountain, which is what produced
        # the visible up-and-down hunting.
        if self._smoothed_target_alt is None or raw_target_alt >= self._smoothed_target_alt:
            self._smoothed_target_alt = raw_target_alt
        else:
            alpha = min(1.0, dt / TARGET_FALL_TC)
            self._smoothed_target_alt += (raw_target_alt - self._smoothed_target_alt) * alpha

        # Strip the radial component from the new tangential velocity so the
        # PD has sole responsibility for altitude.
        v_tangent_new = (tangential_vel
                        - float(np.dot(tangential_vel, plane_radial_dir)) * plane_radial_dir)
        v_radial_now = float(np.dot(self._plane_vel, plane_radial_dir))
        altitude_error = self._smoothed_target_alt - plane_radius
        # Critically-damped PD on radial acceleration, integrated to velocity.
        radial_accel = float(np.clip(
            altitude_error * ALT_KP - v_radial_now * ALT_KD,
            -ALT_MAX_ACCEL, ALT_MAX_ACCEL,
        ))
        new_v_radial = float(np.clip(
            v_radial_now + radial_accel * dt,
            -ALT_MAX_VSPEED, ALT_MAX_VSPEED,
        ))
        self._plane_vel = (v_tangent_new + new_v_radial * plane_radial_dir).astype(np.float32)

        # 4. Integrate position
        self._plane_pos = (self._plane_pos + self._plane_vel * dt).astype(np.float32)

        # 5. Camera = at the plane, looking ahead along the *heading* (not the
        # velocity), projected onto the local tangent plane. Driving the camera
        # off velocity makes it stare at the sky on the first frames (when
        # tangential speed is 0 but the altitude PD is producing radial
        # motion); using the heading keeps the horizon level always.
        up_radial = _safe_normalize(self._plane_pos)
        forward = new_dir - float(np.dot(new_dir, up_radial)) * up_radial
        forward = _safe_normalize(forward)
        look_at = (self._plane_pos + forward * LOOK_AHEAD_DIST).astype(np.float32)
        m = QMatrix4x4()
        m.lookAt(
            QVector3D(float(self._plane_pos[0]), float(self._plane_pos[1]), float(self._plane_pos[2])),
            QVector3D(float(look_at[0]),         float(look_at[1]),         float(look_at[2])),
            QVector3D(float(up_radial[0]),       float(up_radial[1]),       float(up_radial[2])),
        )
        self._current_view_matrix = m
        ctx.gl_view.update()
