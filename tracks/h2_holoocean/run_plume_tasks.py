from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    # HoloOcean may change CWD at runtime; make sure local packages remain importable.
    sys.path.insert(0, str(_REPO_ROOT))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tag_now_local() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class RunnerCfg:
    scenario_name: str = "PierHarbor-HoveringCamera"
    package_name: str = "Ocean"
    num_agents: int = 10
    seed: int = 0

    fps: int = 20
    window_width: int = 1280
    window_height: int = 720
    render_quality: int = 3

    # dataset current sampling
    combined_nc: str = "/data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/tiny/combined/combined_environment.nc"
    time_index: int = 0
    depth_index: int = 0
    current_scale: float = 1.0  # multiply dataset uo/vo [m/s]
    current_force_scale: float = 8.0  # convert m/s current -> planar force bias

    # pollution field (must use our diffusion model for official runs)
    pollution_model: str = "ocpnet_3d"  # "gaussian" | "ocpnet_3d"
    pollution_domain_xy_m: float = 160.0
    pollution_depth_range_m: tuple[float, float] = (3.0, 15.0)  # depth (positive down)
    pollution_warmup_s: float = 40.0
    pollution_update_period_s: float = 2.0
    pollution_min_source_dist_m: float = 10.0
    pollution_emission_rate: float = 0.08
    pollution_sink_radius_m: float = 10.0
    pollution_sink_strength_per_s: float = 0.35

    # task 1: plume localization (multi-agent)
    localize_seconds: float = 25.0
    plume_sigma_m: float = 5.0
    success_radius_m: float = 15.0
    grad_eps_m: float = 6.0
    grad_gain: float = 10.0
    explore_speed_mps: float = 2.0

    # task 2: plume containment+cleanup (multi-agent)
    contain_seconds: float = 20.0
    contain_radius_m: float = 10.0
    contain_tolerance_m: float = 5.0
    cleanup_fraction: float = 0.35  # fraction of agents assigned to cleanup role
    cleanup_radius_m: float = 4.0
    cleanup_mass_decay_per_s: float = 1.15
    cleanup_success_mass_frac: float = 0.97
    leakage_success_threshold: float = 0.15

    # controller gains (rough, but stable)
    kp_xy: float = 6.0
    kd_xy: float = 3.0
    kp_z: float = 10.0
    kd_z: float = 4.0
    max_planar_force: float = 18.0
    max_vertical_force: float = 25.0

    # camera follow
    cam_height_m: float = 10.0
    cam_back_m: float = 10.0

    # viewport visibility / lighting
    force_viewport_underwater: bool = True
    viewport_exposure: float = 1.35
    fpv_exposure: float = 1.0

    # visualization helpers
    debug_draw: bool = True
    debug_draw_thickness: float = 80.0
    debug_draw_lifetime_s: float = 0.4


_KNOWN_VARIANTS = (
    "tiny",
    "scene",
    "public",
    "public25_surface",
    "public25_japan_surface",
)


def _default_variant_root() -> Path:
    return Path("/data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants")


def _resolve_combined_nc(combined_nc: str, dataset_variant: str | None) -> str:
    if not dataset_variant:
        return str(Path(combined_nc).expanduser().resolve())
    variant = str(dataset_variant).strip()
    if variant not in _KNOWN_VARIANTS:
        raise ValueError(f"Unknown --dataset-variant='{variant}'. Expected one of: {', '.join(_KNOWN_VARIANTS)}")
    p = _default_variant_root() / variant / "combined" / "combined_environment.nc"
    return str(p.resolve())


def _world_roots_on_disk(package_name: str) -> list[str]:
    roots: list[str] = []
    base = Path.home() / ".local" / "share" / "holoocean"
    if base.is_dir():
        for p in base.glob(f"*/worlds/{package_name}"):
            if p.is_dir():
                roots.append(str(p.resolve()))
    return sorted(set(roots))


def _ensure_uint8_rgb(frame) -> "np.ndarray":
    import numpy as np

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC frame, got shape={arr.shape}")
    if arr.shape[-1] == 4:
        return arr[:, :, :3]
    if arr.shape[-1] == 3:
        return arr
    raise ValueError(f"Expected 3 or 4 channels, got shape={arr.shape}")


class _Mp4Writer:
    def __init__(self, path: Path, fps: int):
        import imageio.v2 as imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(self.path, fps=fps)

    def append(self, frame_rgb_u8) -> None:
        self._writer.append_data(frame_rgb_u8)

    def close(self) -> None:
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _downscale_for_gif(frame_rgb_u8, *, target_width: int = 480):
    from PIL import Image
    import numpy as np

    arr = np.asarray(frame_rgb_u8, dtype=np.uint8)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    tw = int(max(64, target_width))
    if w <= tw:
        return arr
    th = int(round(h * (tw / float(w))))
    img = Image.fromarray(arr, mode="RGB").resize((tw, max(1, th)), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _write_gif(path: Path, frames_rgb_u8: list, *, fps: int = 8) -> None:
    import imageio.v2 as imageio

    if not frames_rgb_u8:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / float(max(1, int(fps)))
    imageio.mimsave(path, frames_rgb_u8, duration=duration)


def _get_fpv_frame(agent_state: dict):
    for k in ("LeftCamera", "RightCamera", "FrontCamera", "FPVCamera"):
        if k in agent_state and agent_state[k] is not None:
            return agent_state[k]
    return None


def _safe_tick(env, *, publish: bool, retries: int = 6, base_sleep_s: float = 0.15):
    import time

    last_exc: Exception | None = None
    for i in range(int(retries)):
        try:
            return env.tick(publish=bool(publish))
        except Exception as e:
            last_exc = e
            if e.__class__.__name__ != "BusyError":
                raise
            # HoloOcean can occasionally stall on semaphore acquire; retry a few times.
            time.sleep(float(base_sleep_s) * float(i + 1))
    raise RuntimeError(f"HoloOcean tick failed after {retries} retries (BusyError).") from last_exc

def _pose_to_position(pose) -> list[float] | None:
    import numpy as np

    if pose is None:
        return None
    arr = np.asarray(pose, dtype=np.float32)
    if arr.shape != (4, 4):
        return None
    p = arr[:3, 3]
    return [float(p[0]), float(p[1]), float(p[2])]


def _look_at_rpy(camera_xyz: list[float], target_xyz: list[float]) -> list[float]:
    dx = target_xyz[0] - camera_xyz[0]
    dy = target_xyz[1] - camera_xyz[1]
    dz = target_xyz[2] - camera_xyz[2]
    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = math.hypot(dx, dy)
    pitch = math.degrees(math.atan2(dz, max(1e-6, dist_xy)))
    return [0.0, pitch, yaw]


def _clip_norm2(vx: float, vy: float, max_norm: float) -> tuple[float, float]:
    n = math.hypot(vx, vy)
    if n <= max_norm or n <= 1e-8:
        return vx, vy
    s = max_norm / n
    return vx * s, vy * s


def _thrusters_for_planar_force(fx: float, fy: float) -> tuple[float, float, float, float]:
    # HoveringAUV angled thrusters directions are +/- 45deg in XY.
    # Using a symmetric solution for 4 angled thrusters (indices 4..7):
    inv = 1.0 / (2.0 * math.sqrt(2.0))
    a = (fx + fy) * inv
    b = (fx - fy) * inv
    return a, b, a, b


def _action_from_force(fx: float, fy: float, fz: float, cfg: RunnerCfg) -> "np.ndarray":
    import numpy as np

    fx, fy = _clip_norm2(float(fx), float(fy), float(cfg.max_planar_force))
    fz = float(max(-cfg.max_vertical_force, min(cfg.max_vertical_force, float(fz))))
    t4, t5, t6, t7 = _thrusters_for_planar_force(fx, fy)
    v = fz / 4.0
    return np.array([v, v, v, v, t4, t5, t6, t7], dtype=np.float32)


class DatasetCurrent:
    def __init__(self, nc_path: str, time_index: int, depth_index: int):
        import xarray as xr
        import numpy as np

        ds = xr.open_dataset(nc_path)
        if "uo" not in ds or "vo" not in ds:
            raise ValueError("Dataset must include uo and vo.")

        self.latitude = ds["latitude"].values.astype(np.float32)
        self.longitude = ds["longitude"].values.astype(np.float32)
        self.u = ds["uo"].isel(time=int(time_index), depth=int(depth_index)).values.astype(np.float32)
        self.v = ds["vo"].isel(time=int(time_index), depth=int(depth_index)).values.astype(np.float32)

        self.lat0 = float(np.nanmedian(self.latitude))
        self.lon0 = float(np.nanmedian(self.longitude))
        fin = np.isfinite(self.u) & np.isfinite(self.v)
        self.u_med = float(np.nanmedian(self.u[fin])) if fin.any() else 0.0
        self.v_med = float(np.nanmedian(self.v[fin])) if fin.any() else 0.0

    def summary(self) -> dict:
        return {
            "lat_n": int(self.latitude.size),
            "lon_n": int(self.longitude.size),
            "lat0": self.lat0,
            "lon0": self.lon0,
            "u_median_mps": self.u_med,
            "v_median_mps": self.v_med,
        }

    def drift_payload(self, *, current_scale: float = 1.0, domain_size_m: tuple[float, float, float]) -> dict:
        import numpy as np

        return {
            "latitude": np.asarray(self.latitude, dtype=np.float64),
            "longitude": np.asarray(self.longitude, dtype=np.float64),
            "u": np.asarray(self.u, dtype=np.float64) * float(current_scale),
            "v": np.asarray(self.v, dtype=np.float64) * float(current_scale),
            "domain_size_m": [float(domain_size_m[0]), float(domain_size_m[1]), float(domain_size_m[2])],
        }

    def velocity_xy_mps(self, x_m: float, y_m: float) -> tuple[float, float]:
        import numpy as np

        lat = self.lat0 + (y_m / 111_000.0)
        lon = self.lon0 + (x_m / (111_000.0 * max(1e-6, math.cos(math.radians(self.lat0)))))
        i = int(np.argmin(np.abs(self.latitude - lat)))
        j = int(np.argmin(np.abs(self.longitude - lon)))
        u = float(self.u[i, j]) if np.isfinite(self.u[i, j]) else self.u_med
        v = float(self.v[i, j]) if np.isfinite(self.v[i, j]) else self.v_med
        return u, v


def _patch_scenario_for_runner(base_scenario: dict, cfg: RunnerCfg) -> dict:
    # IMPORTANT: HoloOcean sensors return None unless tick_count == tick_every.
    # We therefore set ticks_per_sec == fps and cap all sensor Hz <= fps to ensure tick_every=1.
    scenario = json.loads(json.dumps(base_scenario))
    scenario["package_name"] = cfg.package_name
    scenario["ticks_per_sec"] = int(cfg.fps)
    scenario["frames_per_sec"] = int(cfg.fps)
    scenario["window_width"] = int(cfg.window_width)
    scenario["window_height"] = int(cfg.window_height)

    if "agents" not in scenario or not scenario["agents"]:
        raise ValueError("Scenario has no agents.")

    base_agent = scenario["agents"][0]
    base_loc = [float(x) for x in base_agent.get("location", [0.0, 0.0, -5.0])]
    base_rot = [float(x) for x in base_agent.get("rotation", [0.0, 0.0, 0.0])]

    def _normalize_sensors(sensors: list[dict], *, keep_cameras: bool) -> list[dict]:
        out = []
        for s in sensors:
            s = dict(s)
            s["Hz"] = min(int(s.get("Hz", cfg.fps)), int(cfg.fps))
            if s.get("sensor_type") == "RGBCamera":
                if not keep_cameras:
                    continue
                s["Hz"] = int(cfg.fps)
                s.setdefault("configuration", {})
                s["configuration"]["CaptureWidth"] = min(int(s["configuration"].get("CaptureWidth", 512)), 768)
                s["configuration"]["CaptureHeight"] = min(int(s["configuration"].get("CaptureHeight", 512)), 768)
            out.append(s)

        want = {x.get("sensor_type") for x in out}
        if keep_cameras:
            # Ensure an FPV camera exists even for scenarios that don't ship RGBCamera sensors by default.
            has_rgb = any(x.get("sensor_type") == "RGBCamera" for x in out)
            if not has_rgb:
                out.append(
                    {
                        "sensor_type": "RGBCamera",
                        "sensor_name": "LeftCamera",
                        "socket": "CameraLeftSocket",
                        "Hz": int(cfg.fps),
                        "configuration": {"CaptureWidth": 512, "CaptureHeight": 512},
                    }
                )
        if "PoseSensor" not in want:
            out.append({"sensor_type": "PoseSensor", "socket": "IMUSocket", "Hz": int(cfg.fps)})
        if "VelocitySensor" not in want:
            out.append({"sensor_type": "VelocitySensor", "socket": "IMUSocket", "Hz": int(cfg.fps)})
        if "CollisionSensor" not in want:
            out.append({"sensor_type": "CollisionSensor", "Hz": int(cfg.fps)})

        if keep_cameras:
            out.append(
                {
                    "sensor_type": "ViewportCapture",
                    "sensor_name": "ViewportCapture",
                    "Hz": int(cfg.fps),
                    "configuration": {"CaptureWidth": int(cfg.window_width), "CaptureHeight": int(cfg.window_height)},
                }
            )
        return out

    agents = []
    for i in range(int(cfg.num_agents)):
        a = json.loads(json.dumps(base_agent))
        a["agent_name"] = f"auv{i}"
        a["agent_type"] = base_agent.get("agent_type", "HoveringAUV")
        a["control_scheme"] = 0  # thrusters
        a["rotation"] = base_rot

        # Spread initial positions slightly in XY.
        dx = 2.0 * (i % 5) - 4.0
        dy = 2.0 * (i // 5) - 1.0
        a["location"] = [base_loc[0] + dx, base_loc[1] + dy, base_loc[2]]
        a["sensors"] = _normalize_sensors(a.get("sensors", []), keep_cameras=(i == 0))
        agents.append(a)

    scenario["agents"] = agents
    scenario["main_agent"] = "auv0"
    return scenario


def _apply_viewport_underwater_hack(env, cfg: RunnerCfg, first_agent_z: float) -> None:
    if not cfg.force_viewport_underwater:
        return
    z = float(first_agent_z)
    for _ in range(6):
        env.move_viewport([0.0, 0.0, z + 8.0], [0.0, -25.0, 45.0])
        _safe_tick(env, publish=False)
    try:
        env.set_render_quality(int(cfg.render_quality))
    except Exception:
        pass


def _state_agent(state: dict, agent_name: str) -> dict:
    if agent_name in state:
        return state[agent_name]
    return state


def _count_collision_events(state: dict, *, num_agents: int, prev: dict[str, bool]) -> int:
    events = 0
    for i in range(int(num_agents)):
        name = f"auv{i}"
        ai = _state_agent(state, name)
        coll = False
        if ai.get("CollisionSensor") is not None:
            try:
                coll = bool(ai["CollisionSensor"][0])
            except Exception:
                coll = bool(ai["CollisionSensor"])
        if coll and (not prev.get(name, False)):
            events += 1
        prev[name] = coll
    return int(events)


def _gaussian_conc_xy(x: float, y: float, cx: float, cy: float, sigma: float) -> float:
    dx = x - cx
    dy = y - cy
    r2 = dx * dx + dy * dy
    s2 = max(1e-6, float(sigma) ** 2)
    return float(math.exp(-0.5 * r2 / s2))

class PollutionRuntime:
    def __init__(
        self,
        *,
        field,
        meta: dict,
        origin_world_xy: tuple[float, float],
        domain_xy_m: float,
        depth_range_m: tuple[float, float],
        update_period_s: float,
    ) -> None:
        self.field = field
        self.meta = dict(meta)
        self.origin_world_xy = (float(origin_world_xy[0]), float(origin_world_xy[1]))
        self.domain_xy_m = float(domain_xy_m)
        self.depth_range_m = (float(depth_range_m[0]), float(depth_range_m[1]))
        self.update_period_s = float(update_period_s)
        self._accum_s = 0.0
        self._mass0: float | None = None

    def world_to_local_xyz(self, world_xyz: list[float]) -> "np.ndarray":
        import numpy as np

        # HoloOcean: x/y are horizontal, z is vertical (negative down in our scenarios).
        wx, wy, wz = (float(world_xyz[0]), float(world_xyz[1]), float(world_xyz[2]))
        ox, oy = self.origin_world_xy
        half = 0.5 * self.domain_xy_m
        x = wx - ox + half
        z = wy - oy + half
        depth_abs = float(np.clip(-wz, self.depth_range_m[0], self.depth_range_m[1]))
        # PollutionModel3D expects depth coordinate in [0, depth_range], so we store depth relative to depth_min.
        depth = float(depth_abs - float(self.depth_range_m[0]))
        x = float(np.clip(x, 0.0, self.domain_xy_m))
        z = float(np.clip(z, 0.0, self.domain_xy_m))
        return np.array([x, depth, z], dtype=np.float64)

    def local_to_world_xyz(self, local_xyz: "np.ndarray") -> list[float]:
        x, depth, z = [float(v) for v in list(local_xyz)]
        ox, oy = self.origin_world_xy
        half = 0.5 * self.domain_xy_m
        wx = ox + (x - half)
        wy = oy + (z - half)
        depth_abs = float(self.depth_range_m[0]) + float(depth)
        wz = -float(depth_abs)
        return [wx, wy, wz]

    def set_source_local_xyz(self, local_xyz: "np.ndarray") -> None:
        import numpy as np

        p = np.asarray(local_xyz, dtype=np.float64).reshape(3)
        if hasattr(self.field, "source_xyz"):
            self.field.source_xyz = p.copy()
        if hasattr(self.field, "center_xyz"):
            self.field.center_xyz = p.copy()

        # OCPNet field requires updating the model's source list too.
        if hasattr(self.field, "model") and hasattr(self.field, "pollutant"):
            model = self.field.model
            pollutant = self.field.pollutant
            try:
                model.source_sink.point_sources.clear()
                model.source_sink.area_sources.clear()
                model.source_sink.line_sources.clear()
                model.add_source(
                    type="point",
                    pollutant=pollutant,
                    position=(float(p[0]), float(p[2]), float(p[1])),  # (x, y:=sim z, z:=depth)
                    emission_rate=float(getattr(self.field.cfg, "emission_rate", 0.02)),
                    time_function=None,
                )
                model.current_time = 0.0
            except Exception:
                pass

    def step(self, dt_s: float) -> None:
        self.field.step(float(dt_s))

    def step_if_due(self, dt_s: float) -> int:
        self._accum_s += float(dt_s)
        if self._accum_s + 1e-9 < float(self.update_period_s):
            return 0
        n = int(self._accum_s / max(1e-9, float(self.update_period_s)))
        n = max(1, n)
        for _ in range(n):
            self.step(self.update_period_s)
        self._accum_s -= float(n) * float(self.update_period_s)
        return int(n)

    def sample(self, world_xyz: list[float]) -> float:
        return float(self.field.sample(self.world_to_local_xyz(world_xyz)))

    def apply_sink(self, agent_world_xyz: list[list[float]]) -> None:
        import numpy as np

        if not agent_world_xyz:
            return
        local = np.stack([self.world_to_local_xyz(p) for p in agent_world_xyz], axis=0)
        try:
            self.field.apply_agent_sink(local)
        except TypeError:
            # Some fields take keyword args.
            self.field.apply_agent_sink(local, dt_s=float(self.update_period_s))

    def _ocpnet_concentration(self):
        if not (hasattr(self.field, "model") and hasattr(self.field, "pollutant")):
            return None
        try:
            return self.field.model.pollutant_fields[self.field.pollutant].get_concentration(self.field.pollutant)
        except Exception:
            return None

    def mass_proxy(self) -> float | None:
        import numpy as np

        conc = self._ocpnet_concentration()
        if conc is None:
            if hasattr(self.field, "mass"):
                return float(getattr(self.field, "mass"))
            return None
        return float(np.sum(conc))

    def mass_fraction(self) -> float | None:
        m = self.mass_proxy()
        if m is None:
            return None
        if self._mass0 is None:
            self._mass0 = float(max(1e-12, m))
        return float(m / max(1e-12, float(self._mass0)))


def _build_pollution_runtime(
    *,
    cfg: RunnerCfg,
    current: DatasetCurrent,
    out_dir: Path,
    origin_world_xy: tuple[float, float],
    spawn_world_z: float,
    freeze_source_after_warmup: bool = False,
) -> tuple[PollutionRuntime, dict]:
    import numpy as np

    from oneocean_sim_headless.pollution import build_pollution_field

    domain = float(cfg.pollution_domain_xy_m)
    depth_min, depth_max = cfg.pollution_depth_range_m
    depth_range = float(depth_max) - float(depth_min)
    if depth_range <= 0.0:
        raise ValueError("pollution_depth_range_m must have depth_max > depth_min")
    bounds_xyz = (
        np.array([0.0, 0.0, 0.0], dtype=np.float64),
        np.array([float(domain), float(depth_range), float(domain)], dtype=np.float64),
    )
    drift_payload = current.drift_payload(
        current_scale=float(cfg.current_scale),
        domain_size_m=(float(domain), float(domain), float(depth_max - depth_min)),
    )
    rng = np.random.default_rng(int(cfg.seed))
    field, meta = build_pollution_field(
        str(cfg.pollution_model),
        rng=rng,
        bounds_xyz=bounds_xyz,
        output_dir=out_dir,
        drift_payload=drift_payload,
    )

    # Optional: strengthen sink (used by containment/cleanup task).
    if hasattr(field, "cfg"):
        try:
            from dataclasses import replace

            if hasattr(field.cfg, "sink_radius_m") and hasattr(field.cfg, "sink_strength_per_s"):
                updates = {
                    "sink_radius_m": float(cfg.pollution_sink_radius_m),
                    "sink_strength_per_s": float(cfg.pollution_sink_strength_per_s),
                }
                if hasattr(field.cfg, "emission_rate"):
                    updates["emission_rate"] = float(cfg.pollution_emission_rate)
                field.cfg = replace(field.cfg, **updates)
        except Exception:
            pass

    runtime = PollutionRuntime(
        field=field,
        meta=meta,
        origin_world_xy=origin_world_xy,
        domain_xy_m=domain,
        depth_range_m=(float(depth_min), float(depth_max)),
        update_period_s=float(cfg.pollution_update_period_s),
    )

    # Choose a source away from the initial centroid (local center is (domain/2, *, domain/2)).
    center_depth_abs = float(np.clip(-spawn_world_z, float(depth_min), float(depth_max)))
    center_depth_rel = float(center_depth_abs - float(depth_min))
    center = np.array([0.5 * domain, float(center_depth_rel), 0.5 * domain], dtype=np.float64)
    min_d = float(cfg.pollution_min_source_dist_m)
    for _ in range(200):
        ang = float(rng.uniform(0.0, 2.0 * math.pi))
        r_hi = min(0.45 * float(domain), float(min_d) + 15.0)
        r = float(rng.uniform(float(min_d), float(max(min_d + 1e-6, r_hi))))
        cand = center.copy()
        cand[0] = float(np.clip(center[0] + r * math.cos(ang), 0.0, float(domain)))
        cand[2] = float(np.clip(center[2] + r * math.sin(ang), 0.0, float(domain)))
        runtime.set_source_local_xyz(cand)
        break

    # Warm up the diffusion field so the plume is visible and probe signals are non-degenerate.
    warm = float(max(0.0, float(cfg.pollution_warmup_s)))
    if warm > 0.0:
        runtime.step(warm)

    emission_after = None
    if bool(freeze_source_after_warmup) and hasattr(runtime.field, "cfg"):
        try:
            from dataclasses import replace

            if hasattr(runtime.field.cfg, "emission_rate"):
                runtime.field.cfg = replace(runtime.field.cfg, emission_rate=0.0)
                src_local = getattr(runtime.field, "source_xyz", None)
                if src_local is not None:
                    runtime.set_source_local_xyz(src_local)
                emission_after = 0.0
        except Exception:
            emission_after = None

    src_local = getattr(runtime.field, "source_xyz", None)
    src_world = runtime.local_to_world_xyz(src_local) if src_local is not None else None

    hotspot_local = None
    hotspot_world = None
    conc_stats = None
    conc = runtime._ocpnet_concentration()
    if conc is not None and hasattr(runtime.field, "model"):
        try:
            conc_stats = {
                "min": float(np.nanmin(conc)),
                "max": float(np.nanmax(conc)),
                "mean": float(np.nanmean(conc)),
            }
            idx = np.unravel_index(int(np.nanargmax(conc)), conc.shape)  # (ix, iy, iz)
            grid = runtime.field.model.grid
            dx, dy, dz = grid.get_grid_spacing()
            hotspot_local = np.array([float(idx[0]) * float(dx), float(idx[2]) * float(dz), float(idx[1]) * float(dy)], dtype=np.float64)
            hotspot_world = runtime.local_to_world_xyz(hotspot_local)
        except Exception:
            hotspot_local = None
            hotspot_world = None
            conc_stats = None
    meta2 = {
        "pollution_model": str(cfg.pollution_model),
        "pollution_domain_xy_m": float(domain),
        "pollution_depth_range_m": [float(depth_min), float(depth_max)],
        "pollution_update_period_s": float(cfg.pollution_update_period_s),
        "pollution_warmup_s": float(cfg.pollution_warmup_s),
        "origin_world_xy": [float(origin_world_xy[0]), float(origin_world_xy[1])],
        "source_local_xyz": src_local.tolist() if src_local is not None else None,
        "source_world_xyz": src_world,
        "hotspot_local_xyz": None if hotspot_local is None else hotspot_local.tolist(),
        "hotspot_world_xyz": hotspot_world,
        "concentration_stats_post_warmup": conc_stats,
        "emission_rate_after_warmup": emission_after,
        "emission_rate_maybe": float(getattr(getattr(runtime.field, "cfg", object()), "emission_rate", float("nan"))),
        "sink_radius_m": float(cfg.pollution_sink_radius_m),
        "sink_strength_per_s": float(cfg.pollution_sink_strength_per_s),
    }
    return runtime, meta2


def _centroid_xy(positions: list[list[float]]) -> tuple[float, float]:
    if not positions:
        return 0.0, 0.0
    return float(sum(p[0] for p in positions) / len(positions)), float(sum(p[1] for p in positions) / len(positions))


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _debug_draw_point(env, cfg: RunnerCfg, *, xyz: list[float], color: tuple[int, int, int]) -> None:
    if not cfg.debug_draw:
        return
    try:
        env.draw_point(
            [float(xyz[0]), float(xyz[1]), float(xyz[2])],
            color=list(color),
            thickness=float(cfg.debug_draw_thickness),
            lifetime=float(cfg.debug_draw_lifetime_s),
        )
    except Exception:
        pass


def _maybe_expose(rgb_u8, exposure: float):
    if exposure == 1.0:
        return rgb_u8
    return (rgb_u8.astype("float32") * float(exposure)).clip(0, 255).astype("uint8")


def run_localize_task(env, cfg: RunnerCfg, current: DatasetCurrent, out_dir: Path) -> dict:
    import numpy as np
    import imageio.v2 as imageio

    task_dir = out_dir / "task_plume_localize"
    task_dir.mkdir(parents=True, exist_ok=True)

    # Warm up until Pose+Viewport exist.
    st = None
    for _ in range(200):
        st = _safe_tick(env, publish=False)
        a0 = _state_agent(st, "auv0")
        if a0.get("PoseSensor") is not None and a0.get("ViewportCapture") is not None and (_get_fpv_frame(a0) is not None):
            break
    if st is None:
        raise RuntimeError("Failed to warm up environment.")

    p0 = _pose_to_position(_state_agent(st, "auv0").get("PoseSensor"))
    if p0 is None:
        raise RuntimeError("PoseSensor missing for auv0.")
    _apply_viewport_underwater_hack(env, cfg, first_agent_z=float(p0[2]))

    rng = np.random.default_rng(int(cfg.seed))
    init_positions = []
    for i in range(cfg.num_agents):
        pos = _pose_to_position(_state_agent(st, f"auv{i}").get("PoseSensor"))
        if pos is not None:
            init_positions.append(pos)
    origin_xy = _centroid_xy(init_positions or [p0])
    pollution, pollution_meta = _build_pollution_runtime(
        cfg=cfg,
        current=current,
        out_dir=task_dir,
        origin_world_xy=origin_xy,
        spawn_world_z=float(p0[2]),
        freeze_source_after_warmup=False,
    )
    src_world = pollution_meta.get("source_world_xyz") or [p0[0], p0[1], p0[2]]
    gt_world = pollution_meta.get("hotspot_world_xyz") or src_world
    gt_kind = "hotspot" if pollution_meta.get("hotspot_world_xyz") is not None else "source"

    steps = int(round(cfg.localize_seconds * cfg.fps))
    dt = 1.0 / float(cfg.fps)

    mp4_path = task_dir / "rollout.mp4"
    gif_path = task_dir / "rollout.gif"
    fpv_mp4_path = task_dir / "rollout_fpv.mp4"
    fpv_gif_path = task_dir / "rollout_fpv.gif"
    start_png = task_dir / "start.png"
    end_png = task_dir / "end.png"
    start_fpv_png = task_dir / "start_fpv.png"
    end_fpv_png = task_dir / "end_fpv.png"
    metrics_path = task_dir / "metrics.json"

    best_conc = -1.0
    best_xy = (float(p0[0]), float(p0[1]))
    success_step = None
    energy = 0.0
    collisions = 0
    prev_colliding: dict[str, bool] = {}

    explore_steps = max(1, int(round(0.55 * float(steps))))
    search_r = float(min(float(cfg.pollution_min_source_dist_m) + 8.0, 0.40 * float(cfg.pollution_domain_xy_m)))
    origin_x, origin_y = float(origin_xy[0]), float(origin_xy[1])
    waypoints = {
        f"auv{i}": [origin_x + search_r * math.cos(2.0 * math.pi * (i / float(cfg.num_agents))), origin_y + search_r * math.sin(2.0 * math.pi * (i / float(cfg.num_agents))), float(src_world[2])]
        for i in range(cfg.num_agents)
    }

    gif_frames = []
    fpv_gif_frames = []
    gif_stride = max(1, int(round(float(cfg.fps) / 8.0)))
    last_fpv = None
    with _Mp4Writer(mp4_path, fps=cfg.fps) as vw, _Mp4Writer(fpv_mp4_path, fps=cfg.fps) as fw:
        for t in range(steps):
            pollution.step_if_due(dt)

            # Camera follows auv0 to keep at least one vehicle visible.
            if t % 2 == 0:
                p_cam = _pose_to_position(_state_agent(st, "auv0").get("PoseSensor")) or [p0[0], p0[1], p0[2]]
                cam_target = [p_cam[0], p_cam[1], float(src_world[2])]
                a = 2.0 * math.pi * (t / float(max(1, steps)))
                r = float(cfg.cam_back_m)
                cam_pos = [p_cam[0] + r * math.cos(a), p_cam[1] + r * math.sin(a), float(src_world[2]) + cfg.cam_height_m]
                env.move_viewport(cam_pos, _look_at_rpy(cam_pos, cam_target))

            if t % 5 == 0:
                _debug_draw_point(env, cfg, xyz=[float(src_world[0]), float(src_world[1]), float(src_world[2])], color=(240, 60, 60))
                if gt_kind == "hotspot":
                    _debug_draw_point(env, cfg, xyz=[float(gt_world[0]), float(gt_world[1]), float(gt_world[2])], color=(80, 240, 120))

            for i in range(cfg.num_agents):
                name = f"auv{i}"
                ai = _state_agent(st, name)
                pos = _pose_to_position(ai.get("PoseSensor"))
                vel = ai.get("VelocitySensor")
                vxyz = np.asarray(vel, dtype=np.float32).reshape(3) if vel is not None else np.zeros((3,), dtype=np.float32)
                if pos is None:
                    continue

                x, y, _z = pos
                conc = pollution.sample(pos)
                if conc > best_conc:
                    best_conc = conc
                    best_xy = (x, y)

                if t < explore_steps:
                    target = waypoints[name]
                else:
                    target = [float(best_xy[0]), float(best_xy[1]), float(src_world[2])]

                ex = float(target[0]) - float(x)
                ey = float(target[1]) - float(y)
                fx = cfg.kp_xy * ex - cfg.kd_xy * float(vxyz[0])
                fy = cfg.kp_xy * ey - cfg.kd_xy * float(vxyz[1])
                fz = cfg.kp_z * (float(target[2]) - float(pos[2])) - cfg.kd_z * float(vxyz[2])

                u, v = current.velocity_xy_mps(x, y)
                fx += cfg.current_force_scale * cfg.current_scale * u
                fy += cfg.current_force_scale * cfg.current_scale * v

                act = _action_from_force(fx, fy, fz, cfg)
                energy += float(np.sum(act * act)) * dt
                env.act(name, act)

            st = _safe_tick(env, publish=False)
            collisions += _count_collision_events(st, num_agents=cfg.num_agents, prev=prev_colliding)

            a0 = _state_agent(st, "auv0")
            frame = a0.get("ViewportCapture")
            last_fpv = _get_fpv_frame(a0)
            if frame is not None:
                rgb = _maybe_expose(_ensure_uint8_rgb(frame), cfg.viewport_exposure)
                vw.append(rgb)
                if t == 10:
                    imageio.imwrite(start_png, rgb)
                if (t % gif_stride) == 0:
                    gif_frames.append(_downscale_for_gif(rgb, target_width=480))

            if last_fpv is not None:
                rgb = _maybe_expose(_ensure_uint8_rgb(last_fpv), cfg.fpv_exposure)
                fw.append(rgb)
                if t == 10:
                    imageio.imwrite(start_fpv_png, rgb)
                if (t % gif_stride) == 0:
                    fpv_gif_frames.append(_downscale_for_gif(rgb, target_width=480))

            if success_step is None:
                for i in range(cfg.num_agents):
                    ai = _state_agent(st, f"auv{i}")
                    pos = _pose_to_position(ai.get("PoseSensor"))
                    if pos is None:
                        continue
                    if math.hypot(pos[0] - float(gt_world[0]), pos[1] - float(gt_world[1])) <= cfg.success_radius_m:
                        success_step = t
                        break

        if frame is not None:
            imageio.imwrite(end_png, _maybe_expose(_ensure_uint8_rgb(frame), cfg.viewport_exposure))
        if last_fpv is not None:
            imageio.imwrite(end_fpv_png, _maybe_expose(_ensure_uint8_rgb(last_fpv), cfg.fpv_exposure))

    _write_gif(gif_path, gif_frames, fps=8)
    _write_gif(fpv_gif_path, fpv_gif_frames, fps=8)

    est = np.array([best_xy[0], best_xy[1]], dtype=np.float32)
    gt = np.array([float(gt_world[0]), float(gt_world[1])], dtype=np.float32)
    err = float(np.linalg.norm(est - gt))
    metrics = {
        "task": "plume_localize",
        "seed": int(cfg.seed),
        "n_agents": int(cfg.num_agents),
        "steps": int(steps),
        "dt_s": float(dt),
        "success": success_step is not None,
        "time_to_success_s": (float(success_step) * dt) if success_step is not None else None,
        "localization_error_m": err,
        "collisions": int(collisions),
        "energy_proxy": float(energy),
        "pollution_meta": pollution_meta,
        "gt_kind": gt_kind,
        "gt_xy": [float(gt_world[0]), float(gt_world[1])],
        "est_source_xy": [float(best_xy[0]), float(best_xy[1])],
    }
    _write_json(metrics_path, metrics)
    return {
        "task_dir": str(task_dir),
        "metrics": metrics,
        "video": str(mp4_path),
        "gif": str(gif_path),
        "video_fpv": str(fpv_mp4_path),
        "gif_fpv": str(fpv_gif_path),
        "start_png": str(start_png),
        "end_png": str(end_png),
        "start_fpv_png": str(start_fpv_png),
        "end_fpv_png": str(end_fpv_png),
    }


def run_contain_cleanup_task(env, cfg: RunnerCfg, current: DatasetCurrent, out_dir: Path) -> dict:
    import numpy as np
    import imageio.v2 as imageio

    task_dir = out_dir / "task_plume_contain_cleanup"
    task_dir.mkdir(parents=True, exist_ok=True)

    st = None
    for _ in range(200):
        st = _safe_tick(env, publish=False)
        a0 = _state_agent(st, "auv0")
        if a0.get("PoseSensor") is not None and a0.get("ViewportCapture") is not None and (_get_fpv_frame(a0) is not None):
            break
    if st is None:
        raise RuntimeError("Failed to warm up environment.")

    p0 = _pose_to_position(_state_agent(st, "auv0").get("PoseSensor"))
    if p0 is None:
        raise RuntimeError("PoseSensor missing for auv0.")
    _apply_viewport_underwater_hack(env, cfg, first_agent_z=float(p0[2]))

    init_positions = []
    for i in range(cfg.num_agents):
        pos = _pose_to_position(_state_agent(st, f"auv{i}").get("PoseSensor"))
        if pos is not None:
            init_positions.append(pos)
    origin_xy = _centroid_xy(init_positions or [p0])
    pollution, pollution_meta = _build_pollution_runtime(
        cfg=cfg,
        current=current,
        out_dir=task_dir,
        origin_world_xy=origin_xy,
        spawn_world_z=float(p0[2]),
        freeze_source_after_warmup=True,
    )
    src_world = pollution_meta.get("source_world_xyz") or [p0[0], p0[1], p0[2]]
    center_world = pollution_meta.get("hotspot_world_xyz") or src_world
    center_kind = "hotspot" if pollution_meta.get("hotspot_world_xyz") is not None else "source"
    center = [float(center_world[0]), float(center_world[1]), float(center_world[2])]

    # Initialize mass proxy baseline after warmup (and after optionally freezing the source).
    _ = pollution.mass_fraction()

    steps = int(round(cfg.contain_seconds * cfg.fps))
    dt = 1.0 / float(cfg.fps)

    mp4_path = task_dir / "rollout.mp4"
    gif_path = task_dir / "rollout.gif"
    fpv_mp4_path = task_dir / "rollout_fpv.mp4"
    fpv_gif_path = task_dir / "rollout_fpv.gif"
    start_png = task_dir / "start.png"
    end_png = task_dir / "end.png"
    start_fpv_png = task_dir / "start_fpv.png"
    end_fpv_png = task_dir / "end_fpv.png"
    metrics_path = task_dir / "metrics.json"

    energy = 0.0
    collisions = 0
    prev_colliding: dict[str, bool] = {}
    success_step = None
    coverage_sum = 0.0
    leakage_sum = 0.0
    leakage_max = 0.0
    mass_min: float | None = None

    n_clean = max(2, int(round(cfg.cleanup_fraction * cfg.num_agents)))
    clean_agents = {f"auv{i}" for i in range(n_clean)}
    sink_every = max(1, int(round(float(cfg.pollution_update_period_s) / max(1e-9, float(dt)))))

    gif_frames = []
    fpv_gif_frames = []
    gif_stride = max(1, int(round(float(cfg.fps) / 8.0)))
    last_fpv = None
    with _Mp4Writer(mp4_path, fps=cfg.fps) as vw, _Mp4Writer(fpv_mp4_path, fps=cfg.fps) as fw:
        for t in range(steps):
            pollution.step_if_due(dt)

            # Camera follows plume center to show the task region.
            if t % 2 == 0:
                cam_target = [float(center[0]), float(center[1]), float(center[2])]
                a = 2.0 * math.pi * (t / float(max(1, steps)))
                r = float(cfg.cam_back_m)
                cam_pos = [float(center[0]) + r * math.cos(a), float(center[1]) + r * math.sin(a), float(center[2]) + cfg.cam_height_m]
                env.move_viewport(cam_pos, _look_at_rpy(cam_pos, cam_target))

            if t % 5 == 0:
                _debug_draw_point(env, cfg, xyz=[float(center[0]), float(center[1]), float(center[2])], color=(80, 200, 255))

            in_ring = 0
            sink_world_positions: list[list[float]] = []
            for i in range(cfg.num_agents):
                name = f"auv{i}"
                ai = _state_agent(st, name)
                pos = _pose_to_position(ai.get("PoseSensor"))
                vel = ai.get("VelocitySensor")
                vxyz = np.asarray(vel, dtype=np.float32).reshape(3) if vel is not None else np.zeros((3,), dtype=np.float32)
                if pos is None:
                    continue

                x, y, z = pos
                if name in clean_agents:
                    target = [center[0], center[1], center[2]]
                else:
                    ang = 2.0 * math.pi * (i / float(cfg.num_agents))
                    target = [center[0] + cfg.contain_radius_m * math.cos(ang), center[1] + cfg.contain_radius_m * math.sin(ang), center[2]]

                d_ring = abs(math.hypot(x - center[0], y - center[1]) - cfg.contain_radius_m)
                if d_ring <= cfg.contain_tolerance_m and name not in clean_agents:
                    in_ring += 1

                ex = target[0] - x
                ey = target[1] - y
                ez = target[2] - z
                fx = cfg.kp_xy * ex - cfg.kd_xy * float(vxyz[0])
                fy = cfg.kp_xy * ey - cfg.kd_xy * float(vxyz[1])
                fz = cfg.kp_z * ez - cfg.kd_z * float(vxyz[2])

                cu, cv = current.velocity_xy_mps(x, y)
                fx += cfg.current_force_scale * cfg.current_scale * cu
                fy += cfg.current_force_scale * cfg.current_scale * cv

                act = _action_from_force(fx, fy, fz, cfg)
                energy += float(np.sum(act * act)) * dt
                env.act(name, act)

                # Approximate "contain+cleanup" by applying a sink near all agents (skimmer swarm).
                # (Cleanup-role agents are tasked to stay near the source; ring-role agents spread sink along the perimeter.)
                sink_world_positions.append([float(x), float(y), float(z)])

            coverage = float(in_ring) / float(max(1, cfg.num_agents - len(clean_agents)))
            # Leakage proxy: mean concentration at an "outer ring" probe, scaled by missing coverage.
            leakage_r = float(cfg.contain_radius_m) + 14.0
            probe_angles = [2.0 * math.pi * (k / 16.0) for k in range(16)]
            probe = []
            for ang in probe_angles:
                px = float(center[0] + leakage_r * math.cos(ang))
                py = float(center[1] + leakage_r * math.sin(ang))
                probe.append(pollution.sample([px, py, float(center[2])]))
            leakage = float((sum(probe) / float(len(probe))) * (1.0 - coverage))
            coverage_sum += float(coverage)
            leakage_sum += float(leakage)
            leakage_max = float(max(float(leakage_max), float(leakage)))

            if sink_world_positions and (t % sink_every == 0):
                pollution.apply_sink(sink_world_positions)

            st = _safe_tick(env, publish=False)
            collisions += _count_collision_events(st, num_agents=cfg.num_agents, prev=prev_colliding)

            a0 = _state_agent(st, "auv0")
            frame = a0.get("ViewportCapture")
            last_fpv = _get_fpv_frame(a0)
            if frame is not None:
                rgb = _maybe_expose(_ensure_uint8_rgb(frame), cfg.viewport_exposure)
                vw.append(rgb)
                if t == 10:
                    imageio.imwrite(start_png, rgb)
                if (t % gif_stride) == 0:
                    gif_frames.append(_downscale_for_gif(rgb, target_width=480))

            if last_fpv is not None:
                rgb = _maybe_expose(_ensure_uint8_rgb(last_fpv), cfg.fpv_exposure)
                fw.append(rgb)
                if t == 10:
                    imageio.imwrite(start_fpv_png, rgb)
                if (t % gif_stride) == 0:
                    fpv_gif_frames.append(_downscale_for_gif(rgb, target_width=480))

            mass_frac = pollution.mass_fraction()
            if mass_frac is not None:
                mass_min = float(mass_frac) if mass_min is None else float(min(float(mass_min), float(mass_frac)))
            if success_step is None and (mass_frac is not None) and (mass_frac <= cfg.cleanup_success_mass_frac) and (leakage <= cfg.leakage_success_threshold):
                success_step = t

        if frame is not None:
            imageio.imwrite(end_png, _maybe_expose(_ensure_uint8_rgb(frame), cfg.viewport_exposure))
        if last_fpv is not None:
            imageio.imwrite(end_fpv_png, _maybe_expose(_ensure_uint8_rgb(last_fpv), cfg.fpv_exposure))

    _write_gif(gif_path, gif_frames, fps=8)
    _write_gif(fpv_gif_path, fpv_gif_frames, fps=8)

    mass_frac_final = pollution.mass_fraction()
    mean_coverage = float(coverage_sum / float(max(1, steps)))
    mean_leakage = float(leakage_sum / float(max(1, steps)))
    metrics = {
        "task": "plume_contain_cleanup",
        "seed": int(cfg.seed),
        "n_agents": int(cfg.num_agents),
        "steps": int(steps),
        "dt_s": float(dt),
        "success": success_step is not None,
        "time_to_success_s": (float(success_step) * dt) if success_step is not None else None,
        "pollution_meta": pollution_meta,
        "center_kind": center_kind,
        "remaining_mass_frac": None if mass_frac_final is None else float(mass_frac_final),
        "min_mass_frac": None if mass_min is None else float(mass_min),
        "mean_contain_coverage_frac": float(mean_coverage),
        "mean_leakage_proxy": float(mean_leakage),
        "max_leakage_proxy": float(leakage_max),
        "collisions": int(collisions),
        "energy_proxy": float(energy),
        "cleanup_agents": sorted(clean_agents),
        "leakage_success_threshold": float(cfg.leakage_success_threshold),
    }
    _write_json(metrics_path, metrics)
    return {
        "task_dir": str(task_dir),
        "metrics": metrics,
        "video": str(mp4_path),
        "gif": str(gif_path),
        "video_fpv": str(fpv_mp4_path),
        "gif_fpv": str(fpv_gif_path),
        "start_png": str(start_png),
        "end_png": str(end_png),
        "start_fpv_png": str(start_fpv_png),
        "end_fpv_png": str(end_fpv_png),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=os.environ.get("OUT_DIR", f"runs/h2_holoocean/plume_tasks_{_tag_now_local()}"))
    ap.add_argument("--scenario", type=str, default=os.environ.get("SCENARIO_NAME", RunnerCfg.scenario_name))
    ap.add_argument("--num-agents", type=int, default=int(os.environ.get("NUM_AGENTS", "10")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    ap.add_argument("--fps", type=int, default=int(os.environ.get("FPS", str(RunnerCfg.fps))))
    ap.add_argument("--render-quality", type=int, default=int(os.environ.get("RENDER_QUALITY", str(RunnerCfg.render_quality))))
    ap.add_argument("--window-width", type=int, default=int(os.environ.get("WINDOW_WIDTH", str(RunnerCfg.window_width))))
    ap.add_argument("--window-height", type=int, default=int(os.environ.get("WINDOW_HEIGHT", str(RunnerCfg.window_height))))
    ap.add_argument(
        "--dataset-variant",
        type=str,
        default=os.environ.get("DATASET_VARIANT", ""),
        help=f"Shortcut for combined_environment.nc under OceanEnv variants ({', '.join(_KNOWN_VARIANTS)}). "
        "If set, overrides --combined-nc.",
    )
    ap.add_argument("--combined-nc", type=str, default=os.environ.get("COMBINED_NC", RunnerCfg.combined_nc))
    ap.add_argument("--time-index", type=int, default=int(os.environ.get("TIME_INDEX", "0")))
    ap.add_argument("--depth-index", type=int, default=int(os.environ.get("DEPTH_INDEX", "0")))
    ap.add_argument("--current-scale", type=float, default=float(os.environ.get("CURRENT_SCALE", str(RunnerCfg.current_scale))))
    ap.add_argument("--current-force-scale", type=float, default=float(os.environ.get("CURRENT_FORCE_SCALE", str(RunnerCfg.current_force_scale))))
    ap.add_argument(
        "--pollution-model",
        type=str,
        default=os.environ.get("POLLUTION_MODEL", RunnerCfg.pollution_model),
        choices=["gaussian", "ocpnet_3d"],
    )
    ap.add_argument("--pollution-domain-xy-m", type=float, default=float(os.environ.get("POLLUTION_DOMAIN_XY_M", str(RunnerCfg.pollution_domain_xy_m))))
    ap.add_argument("--pollution-depth-min-m", type=float, default=float(os.environ.get("POLLUTION_DEPTH_MIN_M", str(RunnerCfg.pollution_depth_range_m[0]))))
    ap.add_argument("--pollution-depth-max-m", type=float, default=float(os.environ.get("POLLUTION_DEPTH_MAX_M", str(RunnerCfg.pollution_depth_range_m[1]))))
    ap.add_argument("--pollution-warmup-s", type=float, default=float(os.environ.get("POLLUTION_WARMUP_S", str(RunnerCfg.pollution_warmup_s))))
    ap.add_argument("--pollution-update-period-s", type=float, default=float(os.environ.get("POLLUTION_UPDATE_PERIOD_S", str(RunnerCfg.pollution_update_period_s))))
    ap.add_argument("--pollution-min-source-dist-m", type=float, default=float(os.environ.get("POLLUTION_MIN_SOURCE_DIST_M", str(RunnerCfg.pollution_min_source_dist_m))))
    ap.add_argument("--pollution-emission-rate", type=float, default=float(os.environ.get("POLLUTION_EMISSION_RATE", str(RunnerCfg.pollution_emission_rate))))
    ap.add_argument("--pollution-sink-radius-m", type=float, default=float(os.environ.get("POLLUTION_SINK_RADIUS_M", str(RunnerCfg.pollution_sink_radius_m))))
    ap.add_argument("--pollution-sink-strength-per-s", type=float, default=float(os.environ.get("POLLUTION_SINK_STRENGTH_PER_S", str(RunnerCfg.pollution_sink_strength_per_s))))
    ap.add_argument("--localize-seconds", type=float, default=float(os.environ.get("LOCALIZE_SECONDS", str(RunnerCfg.localize_seconds))))
    ap.add_argument("--contain-seconds", type=float, default=float(os.environ.get("CONTAIN_SECONDS", str(RunnerCfg.contain_seconds))))
    ap.add_argument("--contain-radius-m", type=float, default=float(os.environ.get("CONTAIN_RADIUS_M", str(RunnerCfg.contain_radius_m))))
    ap.add_argument("--viewport-exposure", type=float, default=float(os.environ.get("VIEWPORT_EXPOSURE", str(RunnerCfg.viewport_exposure))))
    ap.add_argument("--fpv-exposure", type=float, default=float(os.environ.get("FPV_EXPOSURE", str(RunnerCfg.fpv_exposure))))
    ap.add_argument("--debug-draw", action=argparse.BooleanOptionalAction, default=bool(int(os.environ.get("DEBUG_DRAW", "1"))))
    args = ap.parse_args()

    dataset_variant = str(args.dataset_variant).strip() or None
    combined_nc = _resolve_combined_nc(str(args.combined_nc), dataset_variant)

    cfg = RunnerCfg(
        scenario_name=args.scenario,
        num_agents=int(args.num_agents),
        seed=int(args.seed),
        fps=int(args.fps),
        window_width=int(args.window_width),
        window_height=int(args.window_height),
        render_quality=int(args.render_quality),
        combined_nc=combined_nc,
        time_index=int(args.time_index),
        depth_index=int(args.depth_index),
        current_scale=float(args.current_scale),
        current_force_scale=float(args.current_force_scale),
        pollution_model=str(args.pollution_model),
        pollution_domain_xy_m=float(args.pollution_domain_xy_m),
        pollution_depth_range_m=(float(args.pollution_depth_min_m), float(args.pollution_depth_max_m)),
        pollution_warmup_s=float(args.pollution_warmup_s),
        pollution_update_period_s=float(args.pollution_update_period_s),
        pollution_min_source_dist_m=float(args.pollution_min_source_dist_m),
        pollution_emission_rate=float(args.pollution_emission_rate),
        pollution_sink_radius_m=float(args.pollution_sink_radius_m),
        pollution_sink_strength_per_s=float(args.pollution_sink_strength_per_s),
        localize_seconds=float(args.localize_seconds),
        contain_seconds=float(args.contain_seconds),
        contain_radius_m=float(args.contain_radius_m),
        viewport_exposure=float(args.viewport_exposure),
        fpv_exposure=float(args.fpv_exposure),
        debug_draw=bool(args.debug_draw),
    )

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import numpy as np
    import holoocean
    from holoocean.packagemanager import get_scenario

    np.random.seed(int(cfg.seed))

    current = DatasetCurrent(cfg.combined_nc, cfg.time_index, cfg.depth_index)
    base_scenario = get_scenario(cfg.scenario_name)
    scenario = _patch_scenario_for_runner(base_scenario, cfg)

    media_manifest = {
        "created_at_utc": _utc_now(),
        "track": "h2_holoocean",
        "task_suite": "plume_localize + plume_contain_cleanup",
        "script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "cfg": asdict(cfg),
        "dataset_current": current.summary(),
        "dataset_variant": dataset_variant,
        "scenario_name": cfg.scenario_name,
        "scenario_cfg_note": "scenario loaded from installed package then patched in-memory (package_name + multi-agent + ViewportCapture + sensor Hz caps).",
        "worlds_roots_on_disk": _world_roots_on_disk(cfg.package_name),
        "provenance_note": "For detailed external-scene provenance and licensing notes, see tracks/h2_holoocean/scene_provenance.md",
        "outputs": {},
        "command_hint": (
            "cd oneocean(iros-2026-code) && "
            f"{sys.executable} tracks/h2_holoocean/run_plume_tasks.py "
            f"--scenario {cfg.scenario_name} --num-agents {cfg.num_agents} --seed {cfg.seed} "
            f"--pollution-model {cfg.pollution_model}"
        ),
        "note_on_gt": "For visualization/debug-draw in videos we mark the diffusion-model plume source; tasks/metrics are computed from the same pollution field.",
    }

    with holoocean.make(
        scenario_cfg=scenario,
        show_viewport=False,
        ticks_per_sec=cfg.fps,
        frames_per_sec=cfg.fps,
        verbose=False,
        copy_state=True,
    ) as env:
        env.set_render_quality(int(cfg.render_quality))
        env.should_render_viewport(True)
        _safe_tick(env, publish=False)

        localize_res = run_localize_task(env, cfg, current, out_dir)
        env.reset()
        env.set_render_quality(int(cfg.render_quality))
        env.should_render_viewport(True)
        contain_res = run_contain_cleanup_task(env, cfg, current, out_dir)

    media_manifest["outputs"]["localize"] = localize_res
    media_manifest["outputs"]["contain_cleanup"] = contain_res
    _write_json(out_dir / "media_manifest.json", media_manifest)

    merged_metrics = {
        "created_at_utc": _utc_now(),
        "track": "h2_holoocean",
        "cfg": asdict(cfg),
        "localize": localize_res["metrics"],
        "contain_cleanup": contain_res["metrics"],
    }
    _write_json(out_dir / "metrics.json", merged_metrics)

    results_manifest = {
        "created_at_utc": _utc_now(),
        "track": "h2_holoocean",
        "scenario": cfg.scenario_name,
        "seed": int(cfg.seed),
        "num_agents": int(cfg.num_agents),
        "dataset_variant": dataset_variant,
        "combined_nc": cfg.combined_nc,
        "metrics_json": str((out_dir / "metrics.json").resolve()),
        "media_manifest_json": str((out_dir / "media_manifest.json").resolve()),
        "tasks": {
            "plume_localize": {
                "task_dir": localize_res["task_dir"],
                "metrics_json": str((out_dir / "task_plume_localize" / "metrics.json").resolve()),
                "video": localize_res["video"],
                "gif": localize_res["gif"],
                "video_fpv": localize_res["video_fpv"],
                "gif_fpv": localize_res["gif_fpv"],
                "start_png": localize_res["start_png"],
                "end_png": localize_res["end_png"],
                "start_fpv_png": localize_res["start_fpv_png"],
                "end_fpv_png": localize_res["end_fpv_png"],
            },
            "plume_contain_cleanup": {
                "task_dir": contain_res["task_dir"],
                "metrics_json": str((out_dir / "task_plume_contain_cleanup" / "metrics.json").resolve()),
                "video": contain_res["video"],
                "gif": contain_res["gif"],
                "video_fpv": contain_res["video_fpv"],
                "gif_fpv": contain_res["gif_fpv"],
                "start_png": contain_res["start_png"],
                "end_png": contain_res["end_png"],
                "start_fpv_png": contain_res["start_fpv_png"],
                "end_fpv_png": contain_res["end_fpv_png"],
            },
        },
    }
    _write_json(out_dir / "results_manifest.json", results_manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
