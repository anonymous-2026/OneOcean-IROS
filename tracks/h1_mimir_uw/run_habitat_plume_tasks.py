#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np


@dataclass(frozen=True)
class UnderwaterPostprocess:
    beta_rgb: tuple[float, float, float] = (0.06, 0.035, 0.018)  # attenuation per meter (R>G>B)
    ambient_rgb: tuple[float, float, float] = (12.0, 40.0, 85.0)  # haze color
    haze_strength: float = 0.78
    particle_count: int = 260
    particle_radius_px: int = 1
    particle_alpha: float = 0.18


def _apply_underwater(rgb_u8: np.ndarray, depth_m: np.ndarray, cfg: UnderwaterPostprocess, rng: np.random.Generator) -> np.ndarray:
    rgb = rgb_u8
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    rgb = rgb.astype(np.float32)
    depth = depth_m.astype(np.float32)
    depth = np.nan_to_num(depth, nan=50.0, posinf=50.0, neginf=50.0)
    depth = np.clip(depth, 0.0, 80.0)

    amb = np.array(cfg.ambient_rgb, dtype=np.float32).reshape(1, 1, 3)
    beta = np.array(cfg.beta_rgb, dtype=np.float32).reshape(1, 1, 3)
    trans = np.exp(-beta * depth[..., None])
    haze = (1.0 - trans) * cfg.haze_strength
    out = rgb * trans + amb * haze

    out = 0.92 * out + 0.08 * amb

    h, w = out.shape[:2]
    n = int(max(0, cfg.particle_count))
    if n > 0:
        xs = rng.integers(0, w, size=n)
        ys = rng.integers(0, h, size=n)
        for x, y in zip(xs.tolist(), ys.tolist()):
            r = int(cfg.particle_radius_px)
            x0 = max(0, x - r)
            x1 = min(w, x + r + 1)
            y0 = max(0, y - r)
            y1 = min(h, y + r + 1)
            out[y0:y1, x0:x1, :] = (1.0 - cfg.particle_alpha) * out[y0:y1, x0:x1, :] + cfg.particle_alpha * amb

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _alpha_blend(dst_u8: np.ndarray, mask: np.ndarray, color_rgb: tuple[int, int, int], alpha: float) -> None:
    if alpha <= 0.0:
        return
    alpha = float(np.clip(alpha, 0.0, 1.0))
    c = np.array(color_rgb, dtype=np.float32).reshape(1, 1, 3)
    dst = dst_u8.astype(np.float32)
    m = mask.astype(bool)
    if m.ndim != 2:
        raise ValueError("mask must be HxW")
    dst[m] = (1.0 - alpha) * dst[m] + alpha * c
    dst_u8[:] = np.clip(dst, 0.0, 255.0).astype(np.uint8)


def _draw_points(img: np.ndarray, xs: np.ndarray, ys: np.ndarray, color_rgb: tuple[int, int, int], alpha: float, radius_px: int = 1) -> None:
    h, w = img.shape[:2]
    rr = int(max(0, radius_px))
    for x, y in zip(xs.tolist(), ys.tolist()):
        x = int(round(x))
        y = int(round(y))
        if x < 0 or x >= w or y < 0 or y >= h:
            continue
        x0 = max(0, x - rr)
        x1 = min(w, x + rr + 1)
        y0 = max(0, y - rr)
        y1 = min(h, y + rr + 1)
        patch = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        patch[:, :] = True
        _alpha_blend(img[y0:y1, x0:x1], patch, color_rgb, alpha)


def _project_points(render_camera: Any, points_world: np.ndarray, width: int, height: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_world.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=bool)
    P = np.array(render_camera.projection_matrix, dtype=np.float32)
    V = np.array(render_camera.camera_matrix, dtype=np.float32)
    M = P @ V
    pts = np.asarray(points_world, dtype=np.float32).reshape(-1, 3)
    pts_h = np.concatenate([pts, np.ones((pts.shape[0], 1), dtype=np.float32)], axis=1)
    clip = pts_h @ M.T
    w = clip[:, 3]
    valid = w > 1e-6
    ndc = np.zeros((pts.shape[0], 3), dtype=np.float32)
    ndc[valid] = clip[valid, :3] / w[valid, None]
    in_view = valid & (np.abs(ndc[:, 0]) <= 1.0) & (np.abs(ndc[:, 1]) <= 1.0) & (ndc[:, 2] >= -1.0) & (ndc[:, 2] <= 1.0)
    xs = (ndc[:, 0] * 0.5 + 0.5) * float(width - 1)
    ys = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * float(height - 1)
    return xs[in_view], ys[in_view], in_view


@dataclass
class DriftField:
    u_grid: np.ndarray  # (lat, lon)
    v_grid: np.ndarray  # (lat, lon)
    latitude: np.ndarray  # (lat,)
    longitude: np.ndarray  # (lon,)
    mode: Literal["uniform", "spatial"] = "uniform"
    lat0: float | None = None
    lon0: float | None = None

    def summary(self) -> dict[str, Any]:
        u = self.u_grid
        v = self.v_grid
        fin = np.isfinite(u) & np.isfinite(v)
        if not np.any(fin):
            u0 = v0 = 0.0
        else:
            u0 = float(np.nanmedian(u[fin]))
            v0 = float(np.nanmedian(v[fin]))
        return {
            "mode": self.mode,
            "lat_n": int(self.latitude.size),
            "lon_n": int(self.longitude.size),
            "u_median": u0,
            "v_median": v0,
            "lat0": self.lat0,
            "lon0": self.lon0,
        }

    def velocity_world_xz(self, pos_world: np.ndarray) -> np.ndarray:
        # Returns (u_x, u_z) in m/s, mapped as east->x and north->z.
        if self.mode == "uniform":
            fin = np.isfinite(self.u_grid) & np.isfinite(self.v_grid)
            if not np.any(fin):
                return np.zeros((2,), dtype=np.float32)
            return np.array([np.nanmedian(self.u_grid[fin]), np.nanmedian(self.v_grid[fin])], dtype=np.float32)

        # Spatial mode: map local stage meters to lat/lon using a crude tangent-plane approximation.
        if self.lat0 is None or self.lon0 is None:
            self.lat0 = float(np.nanmedian(self.latitude))
            self.lon0 = float(np.nanmedian(self.longitude))

        x_m = float(pos_world[0])
        z_m = float(pos_world[2])
        lat = float(self.lat0 + (z_m / 111_000.0))
        lon = float(self.lon0 + (x_m / (111_000.0 * max(1e-6, math.cos(math.radians(self.lat0))))))
        i = int(np.argmin(np.abs(self.latitude - lat)))
        j = int(np.argmin(np.abs(self.longitude - lon)))
        u = float(self.u_grid[i, j]) if np.isfinite(self.u_grid[i, j]) else 0.0
        v = float(self.v_grid[i, j]) if np.isfinite(self.v_grid[i, j]) else 0.0
        return np.array([u, v], dtype=np.float32)


@dataclass
class PlumeParticles:
    positions: np.ndarray  # (N,3)
    weights: np.ndarray  # (N,)
    source_pos: np.ndarray  # (3,)

    @staticmethod
    def init(source_pos: np.ndarray, n: int, sigma_m: float, rng: np.random.Generator) -> "PlumeParticles":
        src = np.asarray(source_pos, dtype=np.float32).reshape(3)
        pos = src[None, :] + rng.normal(scale=float(sigma_m), size=(int(n), 3)).astype(np.float32)
        w = np.ones((int(n),), dtype=np.float32)
        return PlumeParticles(positions=pos, weights=w, source_pos=src)

    def emit(self, n: int, sigma_m: float, rng: np.random.Generator) -> None:
        if n <= 0:
            return
        new_pos = self.source_pos[None, :] + rng.normal(scale=float(sigma_m), size=(int(n), 3)).astype(np.float32)
        new_w = np.ones((int(n),), dtype=np.float32)
        self.positions = np.concatenate([self.positions, new_pos], axis=0)
        self.weights = np.concatenate([self.weights, new_w], axis=0)

    def advect(self, vel_xz: np.ndarray, dt: float, diffusion_sigma_m: float, rng: np.random.Generator) -> None:
        v = np.asarray(vel_xz, dtype=np.float32).reshape(2)
        self.positions[:, 0] += float(v[0]) * float(dt)
        self.positions[:, 2] += float(v[1]) * float(dt)
        if diffusion_sigma_m > 0.0:
            self.positions += rng.normal(scale=float(diffusion_sigma_m), size=self.positions.shape).astype(np.float32)

    def clip_bounds(self, lo: np.ndarray, hi: np.ndarray) -> None:
        self.positions = np.minimum(np.maximum(self.positions, lo[None, :]), hi[None, :])

    def cleanup_by_agents(self, agent_pos: np.ndarray, radius_m: float) -> int:
        if self.positions.size == 0:
            return 0
        r2 = float(radius_m) ** 2
        keep = np.ones((self.positions.shape[0],), dtype=bool)
        for p in np.asarray(agent_pos, dtype=np.float32).reshape(-1, 3):
            d2 = np.sum((self.positions - p[None, :]) ** 2, axis=1)
            keep &= d2 > r2
        removed = int(np.count_nonzero(~keep))
        self.positions = self.positions[keep]
        self.weights = self.weights[keep]
        return removed


@dataclass
class EpisodeMetrics:
    task: str
    seed: int
    n_agents: int
    steps: int
    dt_s: float
    success: bool
    time_to_success_s: float | None
    localization_error_m: float | None
    remaining_mass_frac: float | None
    collisions_agents: int
    collisions_obstacles: int
    energy_proxy: float


def _load_scene_meta(scene_path: Path) -> dict[str, Any]:
    meta = scene_path.parent / "scene_meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    return {}


def _load_drift(npz_path: Path, mode: Literal["uniform", "spatial"]) -> DriftField:
    payload = np.load(npz_path, allow_pickle=False)
    lat = np.asarray(payload["latitude"], dtype=np.float32)
    lon = np.asarray(payload["longitude"], dtype=np.float32)
    u = np.asarray(payload["u"], dtype=np.float32)
    v = np.asarray(payload["v"], dtype=np.float32)
    return DriftField(u_grid=u, v_grid=v, latitude=lat, longitude=lon, mode=mode)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="H1 (MIMIR-UW) plume tasks in Habitat-Sim, grounded by our drift cache.")
    ap.add_argument("--scene", type=str, required=True, help="Path to stage OBJ (e.g., runs/.../stage.obj)")
    ap.add_argument("--drift-npz", type=str, required=True, help="Path to drift cache .npz exported from combined_environment.nc")
    ap.add_argument("--drift-mode", type=str, default="uniform", choices=["uniform", "spatial"])
    ap.add_argument("--drift-scale", type=float, default=1.0, help="Visual/sim scale on dataset drift (m/s multiplier).")
    ap.add_argument("--task", type=str, required=True, choices=["localize_source", "cleanup_contain"])
    ap.add_argument("--n-agents", type=int, default=8)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--dt", type=float, default=0.25)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--hfov", type=float, default=78.0)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--assets-dir", type=str, default=str(Path(__file__).parent / "assets"))
    ap.add_argument("--n-rocks", type=int, default=18)
    ap.add_argument("--max-particles", type=int, default=2600)
    ap.add_argument("--debug-show-source", action="store_true", help="Overlay the hidden plume source marker (debug only).")
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    t_start = time.time()
    scene_path = Path(args.scene).resolve()
    drift_path = Path(args.drift_npz).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Capture run config early (before any import failures).
    run_cfg = {
        "cmd": " ".join([shlex.quote(x) for x in sys.argv]),
        "scene": str(scene_path),
        "drift_npz": str(drift_path),
        "task": str(args.task),
        "n_agents": int(args.n_agents),
        "steps": int(args.steps),
        "dt_s": float(args.dt),
        "render": {"width": int(args.width), "height": int(args.height), "hfov": float(args.hfov), "fps": float(args.fps)},
        "seed": int(args.seed),
        "drift": {"mode": str(args.drift_mode), "scale": float(args.drift_scale)},
        "assets_dir": str(Path(args.assets_dir).resolve()),
        "n_rocks": int(args.n_rocks),
        "max_particles": int(args.max_particles),
    }
    (out_dir / "run_config.json").write_text(json.dumps(run_cfg, indent=2), encoding="utf-8")

    import habitat_sim
    import habitat_sim.utils.common as c
    import habitat_sim.utils.settings as hs
    import imageio.v3 as iio

    rng = np.random.default_rng(int(args.seed))
    meta = _load_scene_meta(scene_path)
    bounds = meta.get("mesh", {}).get("bounds", None)
    if bounds is None:
        # conservative fallback
        lo = np.array([-4.0, -2.0, -4.0], dtype=np.float32)
        hi = np.array([4.0, 2.0, 4.0], dtype=np.float32)
        center = np.zeros((3,), dtype=np.float32)
        radius = 4.0
    else:
        lo = np.array(bounds[0], dtype=np.float32)
        hi = np.array(bounds[1], dtype=np.float32)
        center = (lo + hi) * 0.5
        ext = (hi - lo)
        radius = float(np.linalg.norm(ext) * 0.5)

    drift = _load_drift(drift_path, mode=str(args.drift_mode))  # type: ignore[arg-type]
    drift_info = drift.summary()
    (out_dir / "drift_summary.json").write_text(json.dumps(drift_info, indent=2), encoding="utf-8")

    # Stage-relative spawn volume.
    pad = np.array([0.25, 0.15, 0.25], dtype=np.float32)
    spawn_lo = lo + pad
    spawn_hi = hi - pad

    # Source location (near seafloor, within bounds).
    source_pos = center.copy()
    source_pos[0] = float(center[0] + 0.35 * (spawn_hi[0] - spawn_lo[0]) * rng.uniform(-0.35, 0.35))
    source_pos[2] = float(center[2] + 0.35 * (spawn_hi[2] - spawn_lo[2]) * rng.uniform(-0.35, 0.35))
    source_pos[1] = float(spawn_lo[1] + 0.15)

    # Habitat simulator config.
    settings = dict(hs.default_sim_settings)
    settings.update(
        {
            "scene": str(scene_path),
            "color_sensor": True,
            "depth_sensor": True,
            "semantic_sensor": False,
            "width": int(args.width),
            "height": int(args.height),
            "hfov": float(args.hfov),
            "sensor_height": 0.0,
            "zfar": 120.0,
            "clear_color": [0.0, 0.0, 0.0, 1.0],
            "seed": int(args.seed),
            "enable_physics": True,
        }
    )
    sim = habitat_sim.Simulator(hs.make_cfg(settings))
    agent = sim.initialize_agent(0)
    rom = sim.get_rigid_object_manager()
    otm = sim.get_object_template_manager()

    assets_dir = Path(args.assets_dir).resolve()
    uuv_cfg = assets_dir / "uuv_proxy.object_config.json"
    rock_cfg = assets_dir / "rock_proxy.object_config.json"
    if not uuv_cfg.exists() or not rock_cfg.exists():
        raise FileNotFoundError(f"Missing track assets in {assets_dir} (expected uuv_proxy/rock_proxy .object_config.json)")

    uuv_template_ids = otm.load_configs(str(uuv_cfg))
    rock_template_ids = otm.load_configs(str(rock_cfg))
    if not uuv_template_ids or not rock_template_ids:
        raise RuntimeError("Failed to load object templates")
    uuv_tid = int(uuv_template_ids[0])
    rock_tid = int(rock_template_ids[0])

    # Spawn rocks (static obstacles).
    rocks: list[Any] = []
    for _ in range(int(max(0, args.n_rocks))):
        o = rom.add_object_by_template_id(rock_tid)
        o.motion_type = habitat_sim.physics.MotionType.STATIC
        pos = np.array(
            [
                rng.uniform(float(spawn_lo[0]), float(spawn_hi[0])),
                float(spawn_lo[1] + 0.05 + 0.18 * rng.random()),
                rng.uniform(float(spawn_lo[2]), float(spawn_hi[2])),
            ],
            dtype=np.float32,
        )
        o.translation = pos
        o.rotation = c.quat_to_magnum(c.quat_from_angle_axis(float(rng.uniform(0, 2 * math.pi)), np.array([0, 1, 0], dtype=np.float32)))
        rocks.append(o)

    # Spawn agents (kinematic vehicles).
    n_agents = int(np.clip(args.n_agents, 2, 10))
    agents: list[Any] = []
    agent_pos = np.zeros((n_agents, 3), dtype=np.float32)
    for i in range(n_agents):
        o = rom.add_object_by_template_id(uuv_tid)
        o.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        pos = center.copy()
        pos[0] = float(center[0] + rng.uniform(-1.0, 1.0) * 0.35 * (spawn_hi[0] - spawn_lo[0]))
        pos[2] = float(center[2] + rng.uniform(-1.0, 1.0) * 0.35 * (spawn_hi[2] - spawn_lo[2]))
        pos[1] = float(spawn_lo[1] + 0.55 + 0.25 * rng.random())
        o.translation = pos
        o.rotation = c.quat_to_magnum(c.quat_from_angle_axis(float(rng.uniform(0, 2 * math.pi)), np.array([0, 1, 0], dtype=np.float32)))
        agents.append(o)
        agent_pos[i] = pos

    # Plume particles (used for rendering + cleanup objective).
    plume = PlumeParticles.init(source_pos=source_pos, n=900, sigma_m=0.55, rng=rng)

    # Task config.
    sense_sigma = 1.2
    success_dist_m = 0.75
    cleanup_radius_m = 0.65
    cleanup_target_frac = 0.30

    best_pos = agent_pos[0].copy()
    best_val = -1.0
    energy_proxy = 0.0
    collisions_agents = 0
    collisions_obstacles = 0

    mp4_path = out_dir / "rollout.mp4"
    keyframe_path = out_dir / "keyframe.png"
    frames: list[np.ndarray] = []
    uw_cfg = UnderwaterPostprocess()

    def concentration_at(p: np.ndarray) -> float:
        d2 = float(np.sum((p - source_pos) ** 2))
        return float(math.exp(-0.5 * d2 / (sense_sigma**2)))

    def set_camera(step_i: int) -> None:
        # Orbit around centroid for 3D parallax + readability.
        centroid = np.mean(agent_pos, axis=0)
        orbit_r = max(1.4, 0.75 * radius)
        elev = math.radians(18.0)
        theta = 2.0 * math.pi * (step_i / max(1, int(args.steps)))
        cam_pos = centroid + np.array([orbit_r * math.cos(theta), orbit_r * math.sin(elev) + 0.65, orbit_r * math.sin(theta)], dtype=np.float32)
        forward_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)
        to_center = (centroid - cam_pos).astype(np.float32)
        rot = c.quat_from_two_vectors(forward_axis, to_center / max(1e-6, float(np.linalg.norm(to_center))))
        state = habitat_sim.AgentState()
        state.position = cam_pos
        state.rotation = rot
        agent.set_state(state)

    def render_frame(step_i: int) -> np.ndarray:
        set_camera(step_i)
        obs = sim.get_sensor_observations()
        rgb = np.asarray(obs["color_sensor"])
        depth = np.asarray(obs["depth_sensor"]).astype(np.float32)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        out = _apply_underwater(rgb, depth, uw_cfg, rng)

        # Overlay plume particles + agent markers using camera matrices (fast screen-space hints).
        rc = agent._sensors["color_sensor"].render_camera
        # Subsample particles for rendering.
        if plume.positions.shape[0] > 0:
            k = int(min(480, plume.positions.shape[0]))
            idx = rng.choice(plume.positions.shape[0], size=k, replace=False)
            px = plume.positions[idx]
            xs, ys, _ = _project_points(rc, px, int(args.width), int(args.height))
            _draw_points(out, xs, ys, color_rgb=(255, 140, 45), alpha=0.25, radius_px=1)

        xs_a, ys_a, _ = _project_points(rc, agent_pos, int(args.width), int(args.height))
        _draw_points(out, xs_a, ys_a, color_rgb=(255, 255, 255), alpha=0.45, radius_px=2)

        if bool(args.debug_show_source):
            xs_s, ys_s, _ = _project_points(rc, source_pos[None, :], int(args.width), int(args.height))
            _draw_points(out, xs_s, ys_s, color_rgb=(255, 40, 40), alpha=0.6, radius_px=3)
        return out

    # Pre-render keyframe.
    frames.append(render_frame(0))
    iio.imwrite(keyframe_path, frames[0])

    t_success: float | None = None
    init_mass = float(plume.positions.shape[0])

    # Main loop.
    for step_i in range(1, int(args.steps)):
        # Dataset-driven drift.
        drift_xz = drift.velocity_world_xz(np.mean(agent_pos, axis=0)) * float(args.drift_scale)

        # Task controllers.
        vel_cmd = np.zeros((n_agents, 3), dtype=np.float32)
        if args.task == "localize_source":
            vals = np.array([concentration_at(p) for p in agent_pos], dtype=np.float32)
            k_best = int(np.argmax(vals))
            if float(vals[k_best]) > float(best_val):
                best_val = float(vals[k_best])
                best_pos = agent_pos[k_best].copy()

            # Leader gradient ascent (finite differences).
            leader = k_best
            p0 = agent_pos[leader].copy()
            eps = 0.35
            cx = concentration_at(p0 + np.array([eps, 0.0, 0.0], dtype=np.float32)) - concentration_at(p0 - np.array([eps, 0.0, 0.0], dtype=np.float32))
            cz = concentration_at(p0 + np.array([0.0, 0.0, eps], dtype=np.float32)) - concentration_at(p0 - np.array([0.0, 0.0, eps], dtype=np.float32))
            g = np.array([cx, 0.0, cz], dtype=np.float32)
            gn = float(np.linalg.norm(g))
            if gn < 1e-6:
                g = rng.normal(size=(3,)).astype(np.float32)
                g[1] = 0.0
                gn = float(np.linalg.norm(g))
            g = g / max(1e-6, gn)
            vel_cmd[leader] = 0.55 * g

            # Others in a ring formation around leader for cooperative sensing.
            ring_r = 1.1
            for i in range(n_agents):
                if i == leader:
                    continue
                angle = 2.0 * math.pi * (i / n_agents)
                target = p0 + np.array([ring_r * math.cos(angle), 0.0, ring_r * math.sin(angle)], dtype=np.float32)
                d = target - agent_pos[i]
                d[1] = 0.0
                dn = float(np.linalg.norm(d))
                vel_cmd[i] = (0.65 * d / max(1e-6, dn)) if dn > 1e-3 else 0.0

            err = float(np.linalg.norm(best_pos - source_pos))
            if t_success is None and err <= success_dist_m:
                t_success = float(step_i) * float(args.dt)

        else:
            # Multi-agent cleanup/containment: keep a rotating barrier around the (advected) plume density.
            plume_center = np.mean(plume.positions, axis=0) if plume.positions.size else source_pos
            barrier_r = 1.35
            spin = 0.35 * step_i * float(args.dt)
            for i in range(n_agents):
                angle = 2.0 * math.pi * (i / n_agents) + spin
                target = plume_center + np.array([barrier_r * math.cos(angle), 0.0, barrier_r * math.sin(angle)], dtype=np.float32)
                d = target - agent_pos[i]
                d[1] = 0.0
                dn = float(np.linalg.norm(d))
                vel_cmd[i] = (0.75 * d / max(1e-6, dn)) if dn > 1e-3 else 0.0

        # Apply drift + command and update kinematic objects.
        max_speed = 1.1
        for i in range(n_agents):
            v = vel_cmd[i].copy()
            v[0] += float(drift_xz[0])
            v[2] += float(drift_xz[1])
            sp = float(np.linalg.norm(v))
            if sp > max_speed:
                v = v * (max_speed / sp)
            agent_pos[i] = agent_pos[i] + v * float(args.dt)
            agent_pos[i] = np.minimum(np.maximum(agent_pos[i], spawn_lo), spawn_hi)
            agents[i].translation = agent_pos[i]
            energy_proxy += float(np.dot(vel_cmd[i], vel_cmd[i])) * float(args.dt)

        # Simple collision proxies.
        # Agent-agent.
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                if float(np.linalg.norm(agent_pos[i] - agent_pos[j])) < 0.28:
                    collisions_agents += 1
        # Agent-rock.
        rock_pos = np.array([np.asarray(r.translation) for r in rocks], dtype=np.float32) if rocks else np.zeros((0, 3), dtype=np.float32)
        if rock_pos.size:
            for i in range(n_agents):
                d2 = np.sum((rock_pos - agent_pos[i][None, :]) ** 2, axis=1)
                if np.any(d2 < (0.38**2)):
                    collisions_obstacles += 1

        # Plume evolution.
        plume.emit(n=9, sigma_m=0.45, rng=rng)
        plume.advect(vel_xz=drift_xz, dt=float(args.dt), diffusion_sigma_m=0.02, rng=rng)
        plume.clip_bounds(spawn_lo, spawn_hi)
        if args.task == "cleanup_contain":
            plume.cleanup_by_agents(agent_pos, radius_m=cleanup_radius_m)
            # Keep bounded particle budget for speed.
            if plume.positions.shape[0] > int(args.max_particles):
                keep = rng.choice(plume.positions.shape[0], size=int(args.max_particles), replace=False)
                plume.positions = plume.positions[keep]
                plume.weights = plume.weights[keep]

        # Render.
        frames.append(render_frame(step_i))

    # Final metrics.
    if args.task == "localize_source":
        loc_err = float(np.linalg.norm(best_pos - source_pos))
        success = loc_err <= success_dist_m
        remaining = None
    else:
        remaining_frac = float(plume.positions.shape[0]) / max(1.0, init_mass)
        remaining = remaining_frac
        success = remaining_frac <= cleanup_target_frac
        loc_err = None

    metrics = EpisodeMetrics(
        task=str(args.task),
        seed=int(args.seed),
        n_agents=int(n_agents),
        steps=int(args.steps),
        dt_s=float(args.dt),
        success=bool(success),
        time_to_success_s=float(t_success) if t_success is not None else None,
        localization_error_m=float(loc_err) if loc_err is not None else None,
        remaining_mass_frac=float(remaining) if remaining is not None else None,
        collisions_agents=int(collisions_agents),
        collisions_obstacles=int(collisions_obstacles),
        energy_proxy=float(energy_proxy),
    )
    (out_dir / "metrics.json").write_text(json.dumps(asdict(metrics), indent=2), encoding="utf-8")

    # Media.
    iio.imwrite(mp4_path, frames, fps=float(args.fps))

    media_manifest = {
        "track": "H1_MIMIR_UW",
        "task": str(args.task),
        "outputs": {"mp4": str(mp4_path), "keyframe_png": str(keyframe_path)},
        "render": {"frames": int(len(frames)), "fps": float(args.fps), "width": int(args.width), "height": int(args.height)},
        "underwater_postprocess": asdict(uw_cfg),
        "cmd": run_cfg["cmd"],
    }
    (out_dir / "media_manifest.json").write_text(json.dumps(media_manifest, indent=2), encoding="utf-8")

    results_manifest = {
        "track": "H1_MIMIR_UW",
        "scene": str(scene_path),
        "scene_meta": str(scene_path.parent / "scene_meta.json") if (scene_path.parent / "scene_meta.json").exists() else None,
        "scene_provenance": str((Path(__file__).parent / "scene_provenance.md").resolve()),
        "dataset_drift_cache": str(drift_path),
        "dataset_drift_summary": drift_info,
        "task": str(args.task),
        "metrics_json": str((out_dir / "metrics.json").resolve()),
        "run_config_json": str((out_dir / "run_config.json").resolve()),
    }
    (out_dir / "results_manifest.json").write_text(json.dumps(results_manifest, indent=2), encoding="utf-8")

    sim.close()

    payload = {
        "out_dir": str(out_dir),
        "success": bool(success),
        "elapsed_s": float(time.time() - t_start),
        "metrics": asdict(metrics),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
