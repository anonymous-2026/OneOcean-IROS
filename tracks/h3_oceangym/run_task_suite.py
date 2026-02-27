from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

# Allow running as a script: `python tracks/h3_oceangym/run_task_suite.py`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ensure_ssl_cert_file() -> None:
    try:
        import certifi  # type: ignore

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass


def _tag_now_local() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_uint8_rgb(frame):
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
    def __init__(self, path: Path, fps: int, size_hw: tuple[int, int]):
        import cv2  # type: ignore

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        w, h = int(size_hw[1]), int(size_hw[0])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, float(fps), (w, h))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open mp4 writer for {self.path}")

    def append_rgb(self, rgb_u8) -> None:
        import cv2  # type: ignore

        self._writer.write(cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


@dataclass
class _Recorder:
    viewport_mp4: Path
    leftcamera_mp4: Path
    fps: int
    _vw: _Mp4Writer | None = None
    _fw: _Mp4Writer | None = None

    def start(self, viewport_frame, fp_frame) -> None:
        vp0 = _ensure_uint8_rgb(viewport_frame)
        fp0 = _ensure_uint8_rgb(fp_frame)
        self._vw = _Mp4Writer(self.viewport_mp4, fps=self.fps, size_hw=vp0.shape[:2])
        self._fw = _Mp4Writer(self.leftcamera_mp4, fps=self.fps, size_hw=fp0.shape[:2])
        self.append(viewport_frame, fp_frame)

    def append(self, viewport_frame, fp_frame) -> None:
        if self._vw is not None and viewport_frame is not None:
            self._vw.append_rgb(_ensure_uint8_rgb(viewport_frame))
        if self._fw is not None and fp_frame is not None:
            self._fw.append_rgb(_ensure_uint8_rgb(fp_frame))

    def close(self) -> None:
        if self._vw is not None:
            self._vw.close()
        if self._fw is not None:
            self._fw.close()


def _pose_xyz(pose) -> tuple[float, float, float]:
    import numpy as np

    arr = np.asarray(pose, dtype=np.float32)
    p = arr[:3, 3]
    return float(p[0]), float(p[1]), float(p[2])


def _look_at_rpy(camera_xyz: tuple[float, float, float], target_xyz: tuple[float, float, float]) -> list[float]:
    dx = target_xyz[0] - camera_xyz[0]
    dy = target_xyz[1] - camera_xyz[1]
    dz = target_xyz[2] - camera_xyz[2]
    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = math.hypot(dx, dy)
    pitch = math.degrees(math.atan2(dz, max(1e-6, dist_xy)))
    return [0.0, pitch, yaw]


@dataclass(frozen=True)
class SuiteCfg:
    preset: str = "ocean_worlds_camera"
    difficulty: str = "medium"
    ticks_per_sec: int = 20
    fps: int = 20
    max_steps: int = 300
    episodes: int = 3
    n_multiagent: int = 8
    render_quality: int = 3
    show_viewport: bool = False
    record_first_episode_only: bool = True

    # Task params (difficulty ladder lives in code; these are defaults).
    nav_goal_radius_m: float = 2.5
    nav_goal_dist_m: float = 40.0
    station_keep_seconds: float = 10.0

    plume_sigma_m: float = 35.0
    plume_success_radius_m: float = 15.0

    contain_leak_radius_m: float = 60.0
    contain_capture_radius_m: float = 6.0
    contain_spawn_per_step: int = 8

    current_u_mps: float = 0.25
    current_v_mps: float = 0.10

    # Optional data-grounded current series (exported from combined_environment.nc via export_current_series_npz.py).
    current_npz: str | None = None
    current_depth_m: float = 0.494025  # default: surface-ish depth from our combined_environment.nc
    dataset_days_per_sim_second: float = 0.0  # 0 => constant sample; >0 => advance time index as sim runs

    # Pollution model for plume-themed tasks.
    pollution_model: str = "analytic"  # analytic | ocpnet_3d
    ocpnet_grid: tuple[int, int, int] = (24, 24, 12)
    ocpnet_domain_m: tuple[float, float, float] = (240.0, 240.0, 60.0)
    ocpnet_sink_radius_m: float = 7.5
    ocpnet_sink_strength: float = 0.15


def _stable_seed(*parts: str | int) -> int:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    # Fit in 32-bit signed range.
    return int(h.hexdigest()[:8], 16)


def _cfg_with_difficulty(cfg: SuiteCfg, difficulty: str) -> SuiteCfg:
    d = str(difficulty).lower().strip()
    if d not in {"easy", "medium", "hard"}:
        raise ValueError(f"Unknown difficulty: {difficulty!r}")
    if d == "medium":
        return replace(cfg, difficulty="medium")
    if d == "easy":
        return replace(
            cfg,
            difficulty="easy",
            current_u_mps=cfg.current_u_mps * 0.6,
            current_v_mps=cfg.current_v_mps * 0.6,
            nav_goal_dist_m=max(25.0, cfg.nav_goal_dist_m * 0.75),
            station_keep_seconds=max(6.0, cfg.station_keep_seconds * 0.8),
            plume_sigma_m=cfg.plume_sigma_m * 1.25,
            plume_success_radius_m=cfg.plume_success_radius_m * 1.25,
            contain_spawn_per_step=max(3, int(round(cfg.contain_spawn_per_step * 0.7))),
        )
    return replace(
        cfg,
        difficulty="hard",
        current_u_mps=cfg.current_u_mps * 1.6,
        current_v_mps=cfg.current_v_mps * 1.6,
        nav_goal_dist_m=cfg.nav_goal_dist_m * 1.5,
        station_keep_seconds=cfg.station_keep_seconds * 1.25,
        plume_sigma_m=max(10.0, cfg.plume_sigma_m * 0.75),
        plume_success_radius_m=max(5.0, cfg.plume_success_radius_m * 0.7),
        contain_spawn_per_step=max(6, int(round(cfg.contain_spawn_per_step * 1.3))),
    )


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(sum(xs) / float(len(xs)))


def _summarize_task(task_name: str, episodes: list[dict]) -> dict:
    succ = [1.0 if bool(ep.get("success", False)) else 0.0 for ep in episodes]
    out: dict[str, object] = {
        "task": task_name,
        "episodes": int(len(episodes)),
        "success_rate": float(sum(succ) / float(len(succ))) if succ else 0.0,
    }
    for k in (
        "time_s",
        "steps",
        "collisions",
        "energy_proxy",
        "error_m",
        "leaked_particles",
        "removed_particles",
        "leaked_mass_kg",
        "removed_mass_kg",
    ):
        vals = []
        for ep in episodes:
            v = ep.get(k)
            if isinstance(v, (int, float)):
                vals.append(float(v))
        m = _mean(vals)
        if m is not None:
            out[f"mean_{k}"] = m
    return out


def _patch_for_suite(base: dict, *, cfg: SuiteCfg, add_viewport: bool, n_agents: int) -> dict:
    import copy

    from tracks.h3_oceangym.holoocean_patch import HoloCfg, add_hovering_auv_agents, patch_scenario_for_recording

    holo_cfg = HoloCfg(
        ticks_per_sec=cfg.ticks_per_sec,
        fps=cfg.fps,
        window_width=1280,
        window_height=720,
        render_quality=cfg.render_quality,
        show_viewport=cfg.show_viewport,
    )
    scenario = patch_scenario_for_recording(base, holo_cfg, add_viewport_capture=add_viewport)
    scenario = add_hovering_auv_agents(scenario, n_agents=n_agents)

    # Set PD control for all agents (HoveringAUV control_scheme=1 is PD controller).
    for a in scenario.get("agents", []):
        a["control_scheme"] = 1

    # Spread agents around the auv0 start pose.
    a0 = scenario["agents"][0]
    x0, y0, z0 = [float(v) for v in a0.get("location", [0.0, 0.0, -5.0])]
    r = 4.0
    for i, a in enumerate(scenario["agents"]):
        if i == 0:
            continue
        ang = 2.0 * math.pi * (i / float(max(2, n_agents)))
        a["location"] = [x0 + r * math.cos(ang), y0 + r * math.sin(ang), z0]
        a["rotation"] = list(copy.deepcopy(a0.get("rotation", [0.0, 0.0, 0.0])))

    return scenario


def _state_get(state: dict, agent: str, key: str, n_agents: int):
    if n_agents == 1:
        return state.get(key)
    return state.get(agent, {}).get(key)


def _apply_constant_current_drift(env, *, state: dict, agent_names: list[str], n_agents: int, u: float, v: float, dt: float):
    import numpy as np

    drift = np.array([u * dt, v * dt, 0.0], dtype=np.float32)
    for name in agent_names:
        pose = _state_get(state, name, "PoseSensor", n_agents)
        if pose is None:
            continue
        x, y, z = _pose_xyz(pose)
        env.agents[name].teleport(location=np.array([x, y, z], dtype=np.float32) + drift)


class _CurrentSeries:
    def __init__(self, npz_path: str, *, depth_m: float):
        import numpy as np

        self.path = str(Path(npz_path).expanduser().resolve())
        data = np.load(self.path, allow_pickle=True)
        self.time_ns = data["time_ns"].astype("int64")
        self.depth_m = data["depth_m"].astype("float32")
        self.uo = data["uo"].astype("float32")
        self.vo = data["vo"].astype("float32")

        if self.uo.ndim != 2 or self.vo.ndim != 2:
            raise ValueError(f"Expected uo/vo shape (T,D); got uo{self.uo.shape} vo{self.vo.shape}")

        self.t_len = int(self.uo.shape[0])
        self.d_len = int(self.uo.shape[1])
        if self.d_len != int(self.depth_m.shape[0]):
            raise ValueError("depth_m length mismatch with uo/vo depth dimension")

        self.depth_idx = int(np.argmin(np.abs(self.depth_m - float(depth_m))))
        self.depth_selected_m = float(self.depth_m[self.depth_idx])

        self.lat = float(data.get("latitude", float("nan")))
        self.lon = float(data.get("longitude", float("nan")))
        self.source_dataset = str(data.get("source_dataset", ""))

    def uv_at(self, *, time_idx: int) -> tuple[float, float]:
        i = int(time_idx) % self.t_len
        return float(self.uo[i, self.depth_idx]), float(self.vo[i, self.depth_idx])


def _episode_current_uv(
    series: _CurrentSeries | None,
    *,
    cfg: SuiteCfg,
    seed: int,
    sim_time_s: float,
) -> tuple[float, float]:
    if series is None:
        return float(cfg.current_u_mps), float(cfg.current_v_mps)

    t0 = seed % max(1, series.t_len)
    if float(cfg.dataset_days_per_sim_second) <= 0.0:
        return series.uv_at(time_idx=t0)

    day_offset = int(math.floor(sim_time_s * float(cfg.dataset_days_per_sim_second)))
    return series.uv_at(time_idx=t0 + day_offset)


def _run_nav_go_to_goal(env, *, cfg: SuiteCfg, seed: int, record: bool, out_dir: Path, n_agents: int, current: _CurrentSeries | None) -> dict:
    import numpy as np

    rng = np.random.default_rng(seed)
    agent_names = [f"auv{i}" for i in range(n_agents)]

    state = env.tick(num_ticks=1, publish=False)
    p0 = _pose_xyz(_state_get(state, "auv0", "PoseSensor", n_agents))
    ang = float(rng.uniform(0.0, 2.0 * math.pi))
    goal = (p0[0] + cfg.nav_goal_dist_m * math.cos(ang), p0[1] + cfg.nav_goal_dist_m * math.sin(ang), p0[2])

    dt = 1.0 / float(cfg.ticks_per_sec)
    energy = 0.0
    collisions = 0

    rec: _Recorder | None = None
    if record:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Warmup until we have frames.
        for _ in range(120):
            state = env.tick(num_ticks=1, publish=False)
            if _state_get(state, "auv0", "ViewportCapture", n_agents) is not None and _state_get(state, "auv0", "LeftCamera", n_agents) is not None:
                break
        rec = _Recorder(
            viewport_mp4=out_dir / "nav_viewport.mp4",
            leftcamera_mp4=out_dir / "nav_leftcamera.mp4",
            fps=cfg.fps,
        )
        rec.start(_state_get(state, "auv0", "ViewportCapture", n_agents), _state_get(state, "auv0", "LeftCamera", n_agents))

    success = False
    steps = 0
    for step in range(cfg.max_steps):
        steps = step + 1

        # Drift hook (data-driven current integration comes later; constant current is the default).
        u, v = _episode_current_uv(current, cfg=cfg, seed=seed, sim_time_s=float(step) * dt)
        _apply_constant_current_drift(
            env,
            state=state,
            agent_names=agent_names,
            n_agents=n_agents,
            u=u,
            v=v,
            dt=dt,
        )

        # Simple PD target: drive each agent toward the same goal.
        for name in agent_names:
            env.act(name, np.array([goal[0], goal[1], goal[2], 0.0, 0.0, 0.0], dtype=np.float32))

        state = env.tick(num_ticks=1, publish=False)

        vel = _state_get(state, "auv0", "VelocitySensor", n_agents)
        if vel is not None:
            v0 = np.asarray(vel, dtype=np.float32)
            energy += float(v0.dot(v0)) * dt

        col = _state_get(state, "auv0", "CollisionSensor", n_agents)
        if col is not None and bool(np.asarray(col).reshape(-1)[0]):
            collisions += 1

        pose = _state_get(state, "auv0", "PoseSensor", n_agents)
        if pose is not None:
            px, py, pz = _pose_xyz(pose)
            if math.dist((px, py, pz), goal) <= cfg.nav_goal_radius_m:
                success = True
                break

        if record and rec is not None:
            rec.append(_state_get(state, "auv0", "ViewportCapture", n_agents), _state_get(state, "auv0", "LeftCamera", n_agents))

            # Keep viewport behind auv0 if pose is available.
            if pose is not None:
                cx, cy, cz = px - 10.0, py - 10.0, pz + 3.5
                env.move_viewport([cx, cy, cz], _look_at_rpy((cx, cy, cz), (px, py, pz)))

    if rec is not None:
        rec.close()

    res = {
        "task": "go_to_goal_current",
        "seed": int(seed),
        "n_agents": int(n_agents),
        "goal_xyz": [float(goal[0]), float(goal[1]), float(goal[2])],
        "success": bool(success),
        "steps": int(steps),
        "time_s": float(steps * dt),
        "collisions": int(collisions),
        "energy_proxy": float(energy),
    }
    if record:
        res["media"] = {
            "viewport_mp4": str(out_dir / "nav_viewport.mp4"),
            "leftcamera_mp4": str(out_dir / "nav_leftcamera.mp4"),
        }
    return res


def _run_station_keeping(env, *, cfg: SuiteCfg, seed: int, record: bool, out_dir: Path, n_agents: int, current: _CurrentSeries | None) -> dict:
    import numpy as np

    dt = 1.0 / float(cfg.ticks_per_sec)
    steps = max(1, int(round(cfg.station_keep_seconds / dt)))
    agent_names = [f"auv{i}" for i in range(n_agents)]

    state = env.tick(num_ticks=1, publish=False)
    p0 = _pose_xyz(_state_get(state, "auv0", "PoseSensor", n_agents))

    sq_err = 0.0
    collisions = 0
    energy = 0.0

    rec: _Recorder | None = None
    if record:
        out_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(120):
            if _state_get(state, "auv0", "ViewportCapture", n_agents) is not None and _state_get(state, "auv0", "LeftCamera", n_agents) is not None:
                break
            state = env.tick(num_ticks=1, publish=False)
        rec = _Recorder(out_dir / "station_viewport.mp4", out_dir / "station_leftcamera.mp4", fps=cfg.fps)
        rec.start(_state_get(state, "auv0", "ViewportCapture", n_agents), _state_get(state, "auv0", "LeftCamera", n_agents))

    for step in range(steps):
        sim_time_s = float(step) * dt
        u, v = _episode_current_uv(current, cfg=cfg, seed=seed, sim_time_s=sim_time_s)
        _apply_constant_current_drift(
            env,
            state=state,
            agent_names=agent_names,
            n_agents=n_agents,
            u=u,
            v=v,
            dt=dt,
        )
        for name in agent_names:
            env.act(name, np.array([p0[0], p0[1], p0[2], 0.0, 0.0, 0.0], dtype=np.float32))
        state = env.tick(num_ticks=1, publish=False)
        pose = _state_get(state, "auv0", "PoseSensor", n_agents)
        if pose is not None:
            px, py, pz = _pose_xyz(pose)
            sq_err += float((px - p0[0]) ** 2 + (py - p0[1]) ** 2 + (pz - p0[2]) ** 2)
        col = _state_get(state, "auv0", "CollisionSensor", n_agents)
        if col is not None and bool(np.asarray(col).reshape(-1)[0]):
            collisions += 1
        vel = _state_get(state, "auv0", "VelocitySensor", n_agents)
        if vel is not None:
            v0 = np.asarray(vel, dtype=np.float32)
            energy += float(v0.dot(v0)) * dt

        if rec is not None:
            rec.append(_state_get(state, "auv0", "ViewportCapture", n_agents), _state_get(state, "auv0", "LeftCamera", n_agents))

    rms = math.sqrt(sq_err / float(steps))
    if rec is not None:
        rec.close()

    res = {
        "task": "station_keeping",
        "seed": int(seed),
        "n_agents": int(n_agents),
        "success": bool(rms <= 3.0),
        "steps": int(steps),
        "time_s": float(steps * dt),
        "rms_pos_error_m": float(rms),
        "collisions": int(collisions),
        "energy_proxy": float(energy),
    }
    if record:
        res["media"] = {
            "viewport_mp4": str(out_dir / "station_viewport.mp4"),
            "leftcamera_mp4": str(out_dir / "station_leftcamera.mp4"),
        }
    return res


def _run_plume_localization(env, *, cfg: SuiteCfg, seed: int, record: bool, out_dir: Path, n_agents: int, current: _CurrentSeries | None) -> dict:
    """
    Simple plume localization in world XY using an analytic Gaussian concentration field.
    Baseline policy: short spiral sweep around start; estimate source as argmax concentration.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    dt = 1.0 / float(cfg.ticks_per_sec)

    state = env.tick(num_ticks=1, publish=False)
    start = _pose_xyz(_state_get(state, "auv0", "PoseSensor", n_agents))

    # Hidden source location (near start, but not identical).
    src = (
        start[0] + float(rng.uniform(-50.0, 50.0)),
        start[1] + float(rng.uniform(-50.0, 50.0)),
        start[2],
    )

    best_c = -1.0
    best_xy = (start[0], start[1])

    plume = None
    if str(cfg.pollution_model).lower().strip() == "ocpnet_3d":
        from tracks.h3_oceangym.ocpnet_plume import OCPNetCfg, OCPNetPlume

        plume = OCPNetPlume(
            cfg=OCPNetCfg(
                domain_size_m=tuple(float(x) for x in cfg.ocpnet_domain_m),
                grid_resolution=tuple(int(x) for x in cfg.ocpnet_grid),
                time_step_s=dt,
            ),
            work_dir=out_dir / "_ocpnet_cache",
            world_center_xyz=(start[0], start[1], start[2]),
        )
        plume.set_source_world((src[0], src[1], src[2]))

    def conc(x: float, y: float, z: float) -> float:
        if plume is None:
            dx = x - src[0]
            dy = y - src[1]
            return math.exp(-0.5 * (dx * dx + dy * dy) / (cfg.plume_sigma_m * cfg.plume_sigma_m))
        return plume.concentration_at_world((x, y, z))

    rec: _Recorder | None = None
    if record:
        out_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(120):
            if _state_get(state, "auv0", "ViewportCapture", n_agents) is not None and _state_get(state, "auv0", "LeftCamera", n_agents) is not None:
                break
            state = env.tick(num_ticks=1, publish=False)
        rec = _Recorder(out_dir / "plume_viewport.mp4", out_dir / "plume_leftcamera.mp4", fps=cfg.fps)
        rec.start(_state_get(state, "auv0", "ViewportCapture", n_agents), _state_get(state, "auv0", "LeftCamera", n_agents))

    # Spiral sweep targets.
    steps = min(cfg.max_steps, 240)
    for k in range(steps):
        a = 2.0 * math.pi * (k / 60.0)
        r = min(55.0, 0.4 * k)
        tx = start[0] + r * math.cos(a)
        ty = start[1] + r * math.sin(a)

        u, v = _episode_current_uv(current, cfg=cfg, seed=seed, sim_time_s=float(k) * dt)
        _apply_constant_current_drift(
            env,
            state=state,
            agent_names=["auv0"],
            n_agents=n_agents,
            u=u,
            v=v,
            dt=dt,
        )
        if plume is not None:
            plume.step(u_mps=float(u), v_mps=float(v))

        env.act("auv0", np.array([tx, ty, start[2], 0.0, 0.0, 0.0], dtype=np.float32))
        state = env.tick(num_ticks=1, publish=False)

        pose = _state_get(state, "auv0", "PoseSensor", n_agents)
        if pose is None:
            continue
        px, py, _pz = _pose_xyz(pose)
        c = conc(px, py, start[2])
        if c > best_c:
            best_c = c
            best_xy = (px, py)

        if rec is not None:
            rec.append(_state_get(state, "auv0", "ViewportCapture", n_agents), _state_get(state, "auv0", "LeftCamera", n_agents))

    err = math.dist((best_xy[0], best_xy[1]), (src[0], src[1]))
    success = err <= cfg.plume_success_radius_m
    if rec is not None:
        rec.close()

    res = {
        "task": "plume_localization",
        "seed": int(seed),
        "n_agents": int(n_agents),
        "source_xy": [float(src[0]), float(src[1])],
        "estimate_xy": [float(best_xy[0]), float(best_xy[1])],
        "error_m": float(err),
        "success": bool(success),
        "steps": int(steps),
        "time_s": float(steps * dt),
    }
    if record:
        res["media"] = {
            "viewport_mp4": str(out_dir / "plume_viewport.mp4"),
            "leftcamera_mp4": str(out_dir / "plume_leftcamera.mp4"),
        }
    return res


def _run_plume_containment_multiagent(env, *, cfg: SuiteCfg, seed: int, record: bool, out_dir: Path, n_agents: int, current: _CurrentSeries | None) -> dict:
    """
    Multi-agent containment/cleanup using a lightweight particle advection model:
    - particles spawn near the source
    - particles advect with the configured constant current + diffusion
    - agents remove particles within capture radius
    - leakage counts particles that escape beyond leak radius
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    dt = 1.0 / float(cfg.ticks_per_sec)
    agent_names = [f"auv{i}" for i in range(n_agents)]

    state = env.tick(num_ticks=1, publish=False)
    p0 = _pose_xyz(_state_get(state, "auv0", "PoseSensor", n_agents))
    src = (p0[0] + float(rng.uniform(-30.0, 30.0)), p0[1] + float(rng.uniform(-30.0, 30.0)), p0[2])

    # Ring baseline: hold positions on a circle around source.
    ring_r = 25.0
    ring = []
    for i in range(n_agents):
        a = 2.0 * math.pi * (i / float(n_agents))
        ring.append((src[0] + ring_r * math.cos(a), src[1] + ring_r * math.sin(a), src[2]))

    use_ocpnet = str(cfg.pollution_model).lower().strip() == "ocpnet_3d"
    removed = 0
    leaked = 0
    removed_mass = 0.0
    leaked_mass = 0.0

    plume = None
    if use_ocpnet:
        from tracks.h3_oceangym.ocpnet_plume import OCPNetCfg, OCPNetPlume

        plume = OCPNetPlume(
            cfg=OCPNetCfg(
                domain_size_m=tuple(float(x) for x in cfg.ocpnet_domain_m),
                grid_resolution=tuple(int(x) for x in cfg.ocpnet_grid),
                time_step_s=dt,
            ),
            work_dir=out_dir / "_ocpnet_cache",
            world_center_xyz=(p0[0], p0[1], p0[2]),
        )
        plume.set_source_world((src[0], src[1], src[2]))
    else:
        particles = np.zeros((0, 3), dtype=np.float32)

    # Small diffusion to create spread.
    diff_sigma = 0.5

    # Viewport policy: orbit a camera around the (fixed) source to keep the multi-agent ring visible.
    # Without this, the default viewport in some worlds can end up in darkness (black MP4).
    cam_r = max(55.0, ring_r * 2.2)
    cam_h = 18.0

    rec_vp: _Mp4Writer | None = None
    rec_fp: _Mp4Writer | None = None
    if record:
        out_dir.mkdir(parents=True, exist_ok=True)
        # Wait for viewport capture to come online with non-black pixels.
        for _ in range(240):
            env.move_viewport(
                [src[0] + cam_r, src[1], src[2] + cam_h],
                _look_at_rpy((src[0] + cam_r, src[1], src[2] + cam_h), src),
            )
            state = env.tick(num_ticks=1, publish=False)
            vp = _state_get(state, "auv0", "ViewportCapture", n_agents)
            fp = _state_get(state, "auv0", "LeftCamera", n_agents)
            if vp is None or fp is None:
                continue
            vp0 = _ensure_uint8_rgb(vp)
            if float(vp0.mean()) < 2.0:
                continue
            rec_vp = _Mp4Writer(out_dir / "contain_viewport.mp4", fps=cfg.fps, size_hw=vp0.shape[:2])
            rec_vp.append_rgb(vp0)
            fp0 = _ensure_uint8_rgb(fp)
            rec_fp = _Mp4Writer(out_dir / "contain_leftcamera.mp4", fps=cfg.fps, size_hw=fp0.shape[:2])
            rec_fp.append_rgb(fp0)
            break

    steps = min(cfg.max_steps, 300)
    for step in range(steps):
        u, v = _episode_current_uv(current, cfg=cfg, seed=seed, sim_time_s=float(step) * dt)
        theta = 2.0 * math.pi * (step / float(max(1, steps)))
        cx = src[0] + cam_r * math.cos(theta)
        cy = src[1] + cam_r * math.sin(theta)
        cz = src[2] + cam_h
        env.move_viewport([cx, cy, cz], _look_at_rpy((cx, cy, cz), src))

        if plume is not None:
            plume.step(u_mps=float(u), v_mps=float(v))
        else:
            # Spawn
            spawn = rng.normal(0.0, 2.0, size=(cfg.contain_spawn_per_step, 3)).astype(np.float32)
            spawn[:, 2] = 0.0
            src_xyz = np.array([src[0], src[1], src[2]], dtype=np.float32)
            particles = np.concatenate([particles, src_xyz + spawn], axis=0)

            # Advect + diffuse
            if particles.size:
                particles[:, 0] += float(u) * dt
                particles[:, 1] += float(v) * dt
                particles += rng.normal(0.0, diff_sigma * math.sqrt(dt), size=particles.shape).astype(np.float32)

        # Agents act: stay on ring points.
        for i, name in enumerate(agent_names):
            tx, ty, tz = ring[i]
            env.act(name, np.array([tx, ty, tz, 0.0, 0.0, 0.0], dtype=np.float32))

        state = env.tick(num_ticks=1, publish=False)

        if rec_vp is not None:
            vp = _state_get(state, "auv0", "ViewportCapture", n_agents)
            if vp is not None:
                rec_vp.append_rgb(_ensure_uint8_rgb(vp))
        if rec_fp is not None:
            fp = _state_get(state, "auv0", "LeftCamera", n_agents)
            if fp is not None:
                rec_fp.append_rgb(_ensure_uint8_rgb(fp))

        if plume is not None:
            for name in agent_names:
                pose = _state_get(state, name, "PoseSensor", n_agents)
                if pose is None:
                    continue
                ax, ay, az = _pose_xyz(pose)
                removed_mass += float(
                    plume.apply_sink_at_world(
                        (ax, ay, az),
                        sink_radius_m=float(cfg.ocpnet_sink_radius_m),
                        sink_strength=float(cfg.ocpnet_sink_strength),
                    )
                )
            leaked_mass = float(plume.mass_leaked_world_xy(source_world_xy=(src[0], src[1]), leak_radius_m=float(cfg.contain_leak_radius_m)))
        else:
            # Remove captured particles.
            if particles.size:
                keep = np.ones((particles.shape[0],), dtype=bool)
                for name in agent_names:
                    pose = _state_get(state, name, "PoseSensor", n_agents)
                    if pose is None:
                        continue
                    ax, ay, az = _pose_xyz(pose)
                    d2 = (particles[:, 0] - ax) ** 2 + (particles[:, 1] - ay) ** 2 + (particles[:, 2] - az) ** 2
                    hit = d2 <= (cfg.contain_capture_radius_m ** 2)
                    if hit.any():
                        keep &= ~hit
                n_before = particles.shape[0]
                particles = particles[keep]
                removed += int(n_before - particles.shape[0])

            # Count leakage beyond radius.
            if particles.size:
                dx = particles[:, 0] - src[0]
                dy = particles[:, 1] - src[1]
                far = (dx * dx + dy * dy) >= (cfg.contain_leak_radius_m ** 2)
                if far.any():
                    leaked += int(far.sum())
                    particles = particles[~far]

    if plume is not None:
        removed_mass = max(0.0, float(removed_mass))
        leaked_mass = max(0.0, float(leaked_mass))
        success = (leaked_mass <= 0.25 * removed_mass) if removed_mass > 0 else (leaked_mass == 0.0)
    else:
        # Success: remove a meaningful amount while limiting leakage.
        success = (leaked <= 0.25 * removed) if removed > 0 else (leaked == 0)
    if rec_vp is not None:
        rec_vp.close()
    if rec_fp is not None:
        rec_fp.close()

    res = {
        "task": "plume_containment_multiagent",
        "seed": int(seed),
        "n_agents": int(n_agents),
        "source_xyz": [float(src[0]), float(src[1]), float(src[2])],
        "steps": int(steps),
        "time_s": float(steps * dt),
        "pollution_model": str(cfg.pollution_model),
        "success": bool(success),
    }
    if plume is not None:
        res["removed_mass_kg"] = float(removed_mass)
        res["leaked_mass_kg"] = float(leaked_mass)
    else:
        res["removed_particles"] = int(removed)
        res["leaked_particles"] = int(leaked)
    if record:
        res["media"] = {
            "viewport_mp4": str(out_dir / "contain_viewport.mp4"),
            "leftcamera_mp4": str(out_dir / "contain_leftcamera.mp4"),
        }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=SuiteCfg.preset)
    ap.add_argument("--difficulty", default=SuiteCfg.difficulty, choices=("easy", "medium", "hard"))
    ap.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Optional explicit scenario names (overrides --preset).",
    )
    ap.add_argument("--episodes", type=int, default=SuiteCfg.episodes)
    ap.add_argument("--n_multiagent", type=int, default=SuiteCfg.n_multiagent)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--show_viewport", action="store_true")
    ap.add_argument("--current_npz", default=None, help="Optional exported current series npz (data-grounded forcing).")
    ap.add_argument("--current_depth_m", type=float, default=SuiteCfg.current_depth_m)
    ap.add_argument("--dataset_days_per_sim_second", type=float, default=SuiteCfg.dataset_days_per_sim_second)
    ap.add_argument("--pollution_model", default=SuiteCfg.pollution_model, choices=("analytic", "ocpnet_3d"))
    args = ap.parse_args()

    _ensure_ssl_cert_file()

    from holoocean import packagemanager as pm  # type: ignore
    import holoocean  # type: ignore

    from tracks.h3_oceangym.scenarios import scenario_preset

    cfg = SuiteCfg(
        episodes=int(args.episodes),
        n_multiagent=int(args.n_multiagent),
        show_viewport=bool(args.show_viewport),
        current_npz=str(args.current_npz) if args.current_npz else None,
        current_depth_m=float(args.current_depth_m),
        dataset_days_per_sim_second=float(args.dataset_days_per_sim_second),
        pollution_model=str(args.pollution_model),
    )
    cfg = _cfg_with_difficulty(cfg, str(args.difficulty))
    scenarios = list(args.scenarios) if args.scenarios else scenario_preset(args.preset)

    out_root = Path(args.out_dir) if args.out_dir else Path("runs") / "oceangym_h3" / f"task_suite_{_tag_now_local()}"
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    suite_manifest = {
        "track": "h3_oceangym",
        "script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "out_dir": str(out_root),
        "cfg": asdict(cfg),
        "scenarios": {},
    }
    suite_manifest["task_models"] = {"current": "constant_or_npz_series", "pollution_model": str(cfg.pollution_model)}
    current_series = _CurrentSeries(cfg.current_npz, depth_m=cfg.current_depth_m) if cfg.current_npz else None
    if current_series is not None:
        suite_manifest["data_grounding"] = {
            "current_npz": current_series.path,
            "source_dataset": current_series.source_dataset,
            "lat": current_series.lat,
            "lon": current_series.lon,
            "depth_selected_m": current_series.depth_selected_m,
            "dataset_days_per_sim_second": float(cfg.dataset_days_per_sim_second),
        }

    tasks = [
        ("go_to_goal_current", 1),
        ("station_keeping", 1),
        ("plume_localization", 1),
        ("plume_containment_multiagent", cfg.n_multiagent),
    ]

    for scenario_name in scenarios:
        base = pm.get_scenario(scenario_name)
        per = {
            "scenario_name": scenario_name,
            "episodes": [],
            "media": {},
        }

        for task_name, n_agents in tasks:
            scenario = _patch_for_suite(base, cfg=cfg, add_viewport=True, n_agents=n_agents)
            task_dir = out_root / scenario_name.replace("/", "_") / task_name
            task_dir.mkdir(parents=True, exist_ok=True)

            per_task = {"task": task_name, "n_agents": int(n_agents), "episodes": []}
            media_manifest: dict[str, str] = {}

            record_this = True
            for ep in range(cfg.episodes):
                seed = _stable_seed(scenario_name, task_name, ep)
                record = bool(not cfg.record_first_episode_only or ep == 0)
                record_this = False

                with holoocean.make(
                    scenario_cfg=scenario,
                    show_viewport=cfg.show_viewport,
                    ticks_per_sec=cfg.ticks_per_sec,
                    frames_per_sec=cfg.fps,
                    verbose=False,
                ) as env:
                    env.set_render_quality(int(cfg.render_quality))
                    env.should_render_viewport(True)

                    # Warmup for sensors.
                    state = env.tick(num_ticks=1, publish=False)
                    for _ in range(120):
                        if n_agents == 1:
                            ok = ("PoseSensor" in state)
                        else:
                            ok = ("auv0" in state and "PoseSensor" in state["auv0"])
                        if ok:
                            break
                        state = env.tick(num_ticks=1, publish=False)

                    if task_name == "go_to_goal_current":
                        res = _run_nav_go_to_goal(
                            env,
                            cfg=cfg,
                            seed=seed,
                            record=record,
                            out_dir=task_dir / f"ep{ep:03d}",
                            n_agents=n_agents,
                            current=current_series,
                        )
                    elif task_name == "station_keeping":
                        res = _run_station_keeping(
                            env,
                            cfg=cfg,
                            seed=seed,
                            record=record,
                            out_dir=task_dir / f"ep{ep:03d}",
                            n_agents=n_agents,
                            current=current_series,
                        )
                    elif task_name == "plume_localization":
                        res = _run_plume_localization(
                            env,
                            cfg=cfg,
                            seed=seed,
                            record=record,
                            out_dir=task_dir / f"ep{ep:03d}",
                            n_agents=n_agents,
                            current=current_series,
                        )
                    elif task_name == "plume_containment_multiagent":
                        res = _run_plume_containment_multiagent(
                            env,
                            cfg=cfg,
                            seed=seed,
                            record=record,
                            out_dir=task_dir / f"ep{ep:03d}",
                            n_agents=n_agents,
                            current=current_series,
                        )
                    else:
                        raise ValueError(task_name)

                per_task["episodes"].append(res)
                if "media" in res:
                    for k, v in dict(res["media"]).items():
                        media_manifest[f"ep{ep:03d}_{k}"] = str(v)

            per_task["summary"] = _summarize_task(task_name, list(per_task["episodes"]))

            # Write per-task manifest.
            (task_dir / "media_manifest.json").write_text(json.dumps(media_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (task_dir / "results_manifest.json").write_text(json.dumps(per_task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (task_dir / "metrics.json").write_text(json.dumps(per_task["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
            per["episodes"].append(per_task)

        suite_manifest["scenarios"][scenario_name] = per

    (out_root / "results_manifest.json").write_text(json.dumps(suite_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[h3] wrote:", out_root / "results_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
