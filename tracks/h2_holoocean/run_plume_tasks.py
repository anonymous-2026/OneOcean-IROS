from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


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

    # task 1: plume localization (multi-agent)
    localize_seconds: float = 12.0
    plume_sigma_m: float = 5.0
    success_radius_m: float = 5.0
    grad_eps_m: float = 1.25
    grad_gain: float = 4.0
    explore_speed_mps: float = 1.0

    # task 2: plume containment+cleanup (multi-agent)
    contain_seconds: float = 10.0
    contain_radius_m: float = 10.0
    contain_tolerance_m: float = 2.0
    cleanup_fraction: float = 0.35  # fraction of agents assigned to cleanup role
    cleanup_radius_m: float = 4.0
    cleanup_mass_decay_per_s: float = 1.15
    cleanup_success_mass_frac: float = 0.35
    leakage_success_threshold: float = 0.15

    # controller gains (rough, but stable)
    kp_xy: float = 10.0
    kd_xy: float = 4.0
    kp_z: float = 10.0
    kd_z: float = 4.0
    max_planar_force: float = 30.0
    max_vertical_force: float = 25.0

    # camera follow
    cam_height_m: float = 10.0
    cam_back_m: float = 10.0

    # viewport visibility / lighting
    force_viewport_underwater: bool = True
    viewport_exposure: float = 1.35

    # visualization helpers
    debug_draw: bool = True
    debug_draw_thickness: float = 80.0
    debug_draw_lifetime_s: float = 0.4


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
        env.tick(publish=False)
    try:
        env.set_render_quality(int(cfg.render_quality))
    except Exception:
        pass


def _state_agent(state: dict, agent_name: str) -> dict:
    if agent_name in state:
        return state[agent_name]
    return state


def _gaussian_conc_xy(x: float, y: float, cx: float, cy: float, sigma: float) -> float:
    dx = x - cx
    dy = y - cy
    r2 = dx * dx + dy * dy
    s2 = max(1e-6, float(sigma) ** 2)
    return float(math.exp(-0.5 * r2 / s2))


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
        st = env.tick(publish=False)
        a0 = _state_agent(st, "auv0")
        if a0.get("PoseSensor") is not None and a0.get("ViewportCapture") is not None:
            break
    if st is None:
        raise RuntimeError("Failed to warm up environment.")

    p0 = _pose_to_position(_state_agent(st, "auv0").get("PoseSensor"))
    if p0 is None:
        raise RuntimeError("PoseSensor missing for auv0.")
    _apply_viewport_underwater_hack(env, cfg, first_agent_z=float(p0[2]))

    rng = np.random.default_rng(int(cfg.seed))
    source = np.array([p0[0] + float(rng.uniform(-10.0, 10.0)), p0[1] + float(rng.uniform(-10.0, 10.0)), p0[2]], dtype=np.float32)
    sigma = float(cfg.plume_sigma_m)

    steps = int(round(cfg.localize_seconds * cfg.fps))
    dt = 1.0 / float(cfg.fps)

    mp4_path = task_dir / "rollout.mp4"
    start_png = task_dir / "start.png"
    end_png = task_dir / "end.png"
    metrics_path = task_dir / "metrics.json"

    best_conc = -1.0
    best_xy = (float(p0[0]), float(p0[1]))
    success_step = None
    energy = 0.0
    collisions = 0

    headings = {f"auv{i}": float(rng.uniform(-math.pi, math.pi)) for i in range(cfg.num_agents)}
    eps = float(cfg.grad_eps_m)

    with _Mp4Writer(mp4_path, fps=cfg.fps) as vw:
        for t in range(steps):
            # Camera follows auv0 to keep at least one vehicle visible.
            p_cam = _pose_to_position(_state_agent(st, "auv0").get("PoseSensor")) or [p0[0], p0[1], p0[2]]
            cam_target = [p_cam[0], p_cam[1], float(source[2])]
            cam_pos = [p_cam[0] - cfg.cam_back_m, p_cam[1] - cfg.cam_back_m, float(source[2]) + cfg.cam_height_m]
            env.move_viewport(cam_pos, _look_at_rpy(cam_pos, cam_target))

            if t % 5 == 0:
                _debug_draw_point(env, cfg, xyz=[float(source[0]), float(source[1]), float(source[2])], color=(240, 60, 60))

            for i in range(cfg.num_agents):
                name = f"auv{i}"
                ai = _state_agent(st, name)
                pos = _pose_to_position(ai.get("PoseSensor"))
                vel = ai.get("VelocitySensor")
                vxyz = np.asarray(vel, dtype=np.float32).reshape(3) if vel is not None else np.zeros((3,), dtype=np.float32)
                if pos is None:
                    continue

                x, y, _z = pos
                conc = _gaussian_conc_xy(x, y, float(source[0]), float(source[1]), sigma)
                if conc > best_conc:
                    best_conc = conc
                    best_xy = (x, y)

                cxp = _gaussian_conc_xy(x + eps, y, float(source[0]), float(source[1]), sigma)
                cyp = _gaussian_conc_xy(x, y + eps, float(source[0]), float(source[1]), sigma)
                gx = (cxp - conc) / eps
                gy = (cyp - conc) / eps

                hd = headings[name]
                if (abs(gx) + abs(gy)) < 1e-3 and (t % 10 == 0):
                    headings[name] = float(rng.uniform(-math.pi, math.pi))
                    hd = headings[name]

                vx = cfg.grad_gain * gx + cfg.explore_speed_mps * math.cos(hd)
                vy = cfg.grad_gain * gy + cfg.explore_speed_mps * math.sin(hd)

                fx = cfg.kp_xy * (vx - float(vxyz[0])) - cfg.kd_xy * float(vxyz[0])
                fy = cfg.kp_xy * (vy - float(vxyz[1])) - cfg.kd_xy * float(vxyz[1])
                fz = cfg.kp_z * (0.0 - float(vxyz[2])) - cfg.kd_z * float(vxyz[2])

                u, v = current.velocity_xy_mps(x, y)
                fx += cfg.current_force_scale * cfg.current_scale * u
                fy += cfg.current_force_scale * cfg.current_scale * v

                act = _action_from_force(fx, fy, fz, cfg)
                energy += float(np.sum(act * act)) * dt
                env.act(name, act)

            st = env.tick(publish=False)

            # Collision metric: count per-tick "any collision" to avoid inflated counts.
            any_collision = False
            for i in range(cfg.num_agents):
                ai = _state_agent(st, f"auv{i}")
                if ai.get("CollisionSensor") is not None and bool(ai["CollisionSensor"][0]):
                    any_collision = True
            if any_collision:
                collisions += 1

            a0 = _state_agent(st, "auv0")
            frame = a0.get("ViewportCapture")
            if frame is not None:
                rgb = _maybe_expose(_ensure_uint8_rgb(frame), cfg.viewport_exposure)
                vw.append(rgb)
                if t == 10:
                    imageio.imwrite(start_png, rgb)

            if success_step is None:
                for i in range(cfg.num_agents):
                    ai = _state_agent(st, f"auv{i}")
                    pos = _pose_to_position(ai.get("PoseSensor"))
                    if pos is None:
                        continue
                    if math.hypot(pos[0] - float(source[0]), pos[1] - float(source[1])) <= cfg.success_radius_m:
                        success_step = t
                        break

        if frame is not None:
            imageio.imwrite(end_png, _maybe_expose(_ensure_uint8_rgb(frame), cfg.viewport_exposure))

    est = np.array([best_xy[0], best_xy[1]], dtype=np.float32)
    gt = np.array([float(source[0]), float(source[1])], dtype=np.float32)
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
        "gt_source_xy": [float(source[0]), float(source[1])],
        "est_source_xy": [float(best_xy[0]), float(best_xy[1])],
    }
    _write_json(metrics_path, metrics)
    return {"task_dir": str(task_dir), "metrics": metrics, "video": str(mp4_path), "start_png": str(start_png), "end_png": str(end_png)}


def run_contain_cleanup_task(env, cfg: RunnerCfg, current: DatasetCurrent, out_dir: Path) -> dict:
    import numpy as np
    import imageio.v2 as imageio

    task_dir = out_dir / "task_plume_contain_cleanup"
    task_dir.mkdir(parents=True, exist_ok=True)

    st = None
    for _ in range(200):
        st = env.tick(publish=False)
        a0 = _state_agent(st, "auv0")
        if a0.get("PoseSensor") is not None and a0.get("ViewportCapture") is not None:
            break
    if st is None:
        raise RuntimeError("Failed to warm up environment.")

    p0 = _pose_to_position(_state_agent(st, "auv0").get("PoseSensor"))
    if p0 is None:
        raise RuntimeError("PoseSensor missing for auv0.")
    _apply_viewport_underwater_hack(env, cfg, first_agent_z=float(p0[2]))

    center = [float(p0[0] + 6.0), float(p0[1] - 4.0), float(p0[2])]
    mass = 1.0

    steps = int(round(cfg.contain_seconds * cfg.fps))
    dt = 1.0 / float(cfg.fps)

    mp4_path = task_dir / "rollout.mp4"
    start_png = task_dir / "start.png"
    end_png = task_dir / "end.png"
    metrics_path = task_dir / "metrics.json"

    energy = 0.0
    collisions = 0
    success_step = None

    n_clean = max(2, int(round(cfg.cleanup_fraction * cfg.num_agents)))
    clean_agents = {f"auv{i}" for i in range(n_clean)}

    with _Mp4Writer(mp4_path, fps=cfg.fps) as vw:
        for t in range(steps):
            # Advect plume by dataset current.
            u, v = current.velocity_xy_mps(center[0], center[1])
            center[0] += cfg.current_scale * u * dt
            center[1] += cfg.current_scale * v * dt

            # Camera follows plume center to show the task region.
            cam_target = [float(center[0]), float(center[1]), float(center[2])]
            cam_pos = [float(center[0]) - cfg.cam_back_m, float(center[1]) - cfg.cam_back_m, float(center[2]) + cfg.cam_height_m]
            env.move_viewport(cam_pos, _look_at_rpy(cam_pos, cam_target))

            if t % 5 == 0:
                _debug_draw_point(env, cfg, xyz=[float(center[0]), float(center[1]), float(center[2])], color=(80, 200, 255))

            in_ring = 0
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

                if name in clean_agents and math.hypot(x - center[0], y - center[1]) <= cfg.cleanup_radius_m:
                    mass *= math.exp(-cfg.cleanup_mass_decay_per_s * dt)

            coverage = float(in_ring) / float(max(1, cfg.num_agents - len(clean_agents)))
            leakage = math.exp(-(cfg.contain_radius_m**2) / (2.0 * (cfg.plume_sigma_m**2))) * (1.0 - coverage)

            st = env.tick(publish=False)
            any_collision = False
            for i in range(cfg.num_agents):
                ai = _state_agent(st, f"auv{i}")
                if ai.get("CollisionSensor") is not None and bool(ai["CollisionSensor"][0]):
                    any_collision = True
            if any_collision:
                collisions += 1

            a0 = _state_agent(st, "auv0")
            frame = a0.get("ViewportCapture")
            if frame is not None:
                rgb = _maybe_expose(_ensure_uint8_rgb(frame), cfg.viewport_exposure)
                vw.append(rgb)
                if t == 10:
                    imageio.imwrite(start_png, rgb)

            if success_step is None and (mass <= cfg.cleanup_success_mass_frac) and (leakage <= cfg.leakage_success_threshold):
                success_step = t

        if frame is not None:
            imageio.imwrite(end_png, _maybe_expose(_ensure_uint8_rgb(frame), cfg.viewport_exposure))

    metrics = {
        "task": "plume_contain_cleanup",
        "seed": int(cfg.seed),
        "n_agents": int(cfg.num_agents),
        "steps": int(steps),
        "dt_s": float(dt),
        "success": success_step is not None,
        "time_to_success_s": (float(success_step) * dt) if success_step is not None else None,
        "remaining_mass_frac": float(mass),
        "collisions": int(collisions),
        "energy_proxy": float(energy),
        "cleanup_agents": sorted(clean_agents),
        "leakage_success_threshold": float(cfg.leakage_success_threshold),
    }
    _write_json(metrics_path, metrics)
    return {"task_dir": str(task_dir), "metrics": metrics, "video": str(mp4_path), "start_png": str(start_png), "end_png": str(end_png)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=os.environ.get("OUT_DIR", f"runs/h2_holoocean/plume_tasks_{_tag_now_local()}"))
    ap.add_argument("--scenario", type=str, default=os.environ.get("SCENARIO_NAME", RunnerCfg.scenario_name))
    ap.add_argument("--num-agents", type=int, default=int(os.environ.get("NUM_AGENTS", "10")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    ap.add_argument("--combined-nc", type=str, default=os.environ.get("COMBINED_NC", RunnerCfg.combined_nc))
    ap.add_argument("--time-index", type=int, default=int(os.environ.get("TIME_INDEX", "0")))
    ap.add_argument("--depth-index", type=int, default=int(os.environ.get("DEPTH_INDEX", "0")))
    args = ap.parse_args()

    cfg = RunnerCfg(
        scenario_name=args.scenario,
        num_agents=int(args.num_agents),
        seed=int(args.seed),
        combined_nc=str(args.combined_nc),
        time_index=int(args.time_index),
        depth_index=int(args.depth_index),
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
        "scenario_name": cfg.scenario_name,
        "scenario_cfg_note": "scenario loaded from installed package then patched in-memory (package_name + multi-agent + ViewportCapture + sensor Hz caps).",
        "outputs": {},
        "command_hint": f"cd oneocean(iros-2026-code) && {sys.executable} tracks/h2_holoocean/run_plume_tasks.py --scenario {cfg.scenario_name} --num-agents {cfg.num_agents} --seed {cfg.seed}",
        "note_on_gt": "For visualization/debug-draw in videos we mark the synthetic plume center; tasks/metrics are still computed from the field generator used in this runner.",
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
        env.tick(publish=False)

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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
