from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
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


def _ensure_nofile_limit(min_soft: int = 4096) -> None:
    """
    HoloOcean spawns many short-lived Unreal clients during suites. Some systems default to `ulimit -n 1024`,
    which can trigger `OSError: [Errno 24] Too many open files` mid-run.
    """
    try:
        import resource  # type: ignore

        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft >= min_soft:
            return
        target = min(int(hard), int(min_soft))
        if target > int(soft):
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, int(hard)))
    except Exception:
        return


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
    plume_success_radius_m: float = 25.0

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
    ocpnet_warmup_steps: int = 40
    ocpnet_freeze_source_after_warmup: bool = True
    ocpnet_emission_rate: float = 0.05

    plume_localization_steps: int = 900
    plume_containment_steps: int = 300


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
            nav_goal_dist_m=max(12.0, cfg.nav_goal_dist_m * 0.4),
            nav_goal_radius_m=max(6.0, cfg.nav_goal_radius_m * 3.0),
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
    # Auto-summarize any numeric per-episode fields so new tasks don't require manual schema updates.
    exclude = {
        "task",
        "task_id",
        "task_alias",
        "task_variant",
        "seed",
        "goal_xyz",
        "source_xy",
        "source_xyz",
        "estimate_xy",
        "argmax_xy",
        "media",
        "difficulty",
        "pollution_model",
    }
    keys: set[str] = set()
    for ep in episodes:
        if isinstance(ep, dict):
            keys.update(str(k) for k in ep.keys())
    for k in sorted(keys):
        if k in exclude:
            continue
        vals: list[float] = []
        for ep in episodes:
            if not isinstance(ep, dict):
                continue
            v = ep.get(k)
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                vals.append(float(v))
        m = _mean(vals)
        if m is not None:
            out[f"mean_{k}"] = m
    return out


def _git_info(repo_root: Path) -> dict[str, object]:
    """
    Best-effort git metadata for reproducibility (works even if git is unavailable).
    """
    repo_root = Path(repo_root).resolve()
    info: dict[str, object] = {"repo_root": str(repo_root)}
    try:
        sha = (
            subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        dirty = (
            subprocess.check_output(["git", "-C", str(repo_root), "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
        info["git_sha"] = sha
        info["git_dirty"] = bool(dirty)
    except Exception:
        pass
    return info


def _write_summary_csv(out_root: Path, suite_manifest: dict) -> Path:
    """
    Flatten per-episode results into a table-ready CSV (row = task×difficulty×scenario×N×seed).
    """
    out_root = Path(out_root).resolve()
    rows: list[dict[str, object]] = []
    cfg = suite_manifest.get("cfg", {})
    track = suite_manifest.get("track", "h3_oceangym")
    data_grounding = suite_manifest.get("data_grounding", {}) if isinstance(suite_manifest, dict) else {}
    dg_src = data_grounding.get("source_dataset") if isinstance(data_grounding, dict) else None
    dg_lat = data_grounding.get("lat") if isinstance(data_grounding, dict) else None
    dg_lon = data_grounding.get("lon") if isinstance(data_grounding, dict) else None
    dg_depth = data_grounding.get("depth_selected_m") if isinstance(data_grounding, dict) else None

    scenarios = suite_manifest.get("scenarios", {})
    if not isinstance(scenarios, dict):
        scenarios = {}
    for scenario_name, per in scenarios.items():
        if not isinstance(per, dict):
            continue
        for per_task in per.get("episodes", []):
            if not isinstance(per_task, dict):
                continue
            for ep in per_task.get("episodes", []):
                if not isinstance(ep, dict):
                    continue
                row: dict[str, object] = {
                    "track": track,
                    "scenario": scenario_name,
                    "task_id": ep.get("task_id", per_task.get("task_id", per_task.get("task"))),
                    "task_variant": ep.get("task_variant", per_task.get("task_variant", "")),
                    "task_alias": ep.get("task_alias", per_task.get("task_alias", per_task.get("task"))),
                    "difficulty": ep.get("difficulty", cfg.get("difficulty", "")),
                    "n_agents": ep.get("n_agents", per_task.get("n_agents", "")),
                    "seed": ep.get("seed", ""),
                    "success": ep.get("success", False),
                    "time_s": ep.get("time_s", ""),
                    "steps": ep.get("steps", ""),
                    "collisions": ep.get("collisions", ""),
                    "energy_proxy": ep.get("energy_proxy", ""),
                    "data_source_dataset": dg_src or "",
                    "data_lat": dg_lat if dg_lat is not None else "",
                    "data_lon": dg_lon if dg_lon is not None else "",
                    "data_depth_selected_m": dg_depth if dg_depth is not None else "",
                }
                # Append all numeric per-episode keys so we don't lose task-specific metrics.
                for k, v in ep.items():
                    if k in row:
                        continue
                    if isinstance(v, bool):
                        continue
                    if isinstance(v, (int, float, str)):
                        row[k] = v
                rows.append(row)

    # Stable column order: core columns first, then lexicographic remainder.
    core_cols = [
        "track",
        "scenario",
        "task_id",
        "task_variant",
        "task_alias",
        "difficulty",
        "n_agents",
        "seed",
        "success",
        "time_s",
        "steps",
        "collisions",
        "energy_proxy",
        "data_source_dataset",
        "data_lat",
        "data_lon",
        "data_depth_selected_m",
    ]
    extra_cols: list[str] = []
    seen = set(core_cols)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                extra_cols.append(k)
                seen.add(k)
    cols = core_cols + sorted(extra_cols)

    out_path = out_root / "summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in cols})
    return out_path


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

    # Keep the scenario's original HoveringAUV control scheme (OceanGym worlds ship with a working default).
    # Overriding this can silently break motion control (e.g., go-to-goal never converges).
    cs = int(scenario.get("agents", [{}])[0].get("control_scheme", 0))
    for a in scenario.get("agents", []):
        a["control_scheme"] = cs

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
    prev_collision = False

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
    min_dist = float("inf")
    final_dist = float("inf")
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
        if col is not None:
            now = bool(np.asarray(col).reshape(-1)[0])
            if now and not prev_collision:
                collisions += 1
            prev_collision = now

        pose = _state_get(state, "auv0", "PoseSensor", n_agents)
        if pose is not None:
            px, py, pz = _pose_xyz(pose)
            d_goal = float(math.dist((px, py, pz), goal))
            min_dist = min(min_dist, d_goal)
            final_dist = d_goal
            if d_goal <= cfg.nav_goal_radius_m:
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
        "goal_radius_m": float(cfg.nav_goal_radius_m),
        "success": bool(success),
        "steps": int(steps),
        "time_s": float(steps * dt),
        "min_dist_to_goal_m": float(min_dist),
        "final_dist_to_goal_m": float(final_dist),
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
    prev_collision = False
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
        if col is not None:
            now = bool(np.asarray(col).reshape(-1)[0])
            if now and not prev_collision:
                collisions += 1
            prev_collision = now
        vel = _state_get(state, "auv0", "VelocitySensor", n_agents)
        if vel is not None:
            v0 = np.asarray(vel, dtype=np.float32)
            energy += float(v0.dot(v0)) * dt

        if rec is not None:
            rec.append(_state_get(state, "auv0", "ViewportCapture", n_agents), _state_get(state, "auv0", "LeftCamera", n_agents))

    rms = math.sqrt(sq_err / float(steps))
    if rec is not None:
        rec.close()

    d = str(cfg.difficulty).lower().strip()
    if d == "easy":
        rms_thresh = 8.0
    elif d == "hard":
        rms_thresh = 3.0
    else:
        rms_thresh = 5.0

    res = {
        "task": "station_keeping",
        "seed": int(seed),
        "n_agents": int(n_agents),
        "difficulty": str(cfg.difficulty),
        "success": bool(rms <= rms_thresh),
        "steps": int(steps),
        "time_s": float(steps * dt),
        "rms_pos_error_m": float(rms),
        "rms_success_threshold_m": float(rms_thresh),
        "collisions": int(collisions),
        "energy_proxy": float(energy),
    }
    if record:
        res["media"] = {
            "viewport_mp4": str(out_dir / "station_viewport.mp4"),
            "leftcamera_mp4": str(out_dir / "station_leftcamera.mp4"),
        }
    return res


def _dist_point_to_segment_xy(px: float, py: float, ax: float, ay: float, bx: float, by: float) -> float:
    vx = bx - ax
    vy = by - ay
    wx = px - ax
    wy = py - ay
    vv = vx * vx + vy * vy
    if vv <= 1e-12:
        return math.hypot(px - ax, py - ay)
    t = (wx * vx + wy * vy) / vv
    t = 0.0 if t < 0.0 else 1.0 if t > 1.0 else t
    cx = ax + t * vx
    cy = ay + t * vy
    return math.hypot(px - cx, py - cy)


def _run_route_following_waypoints(env, *, cfg: SuiteCfg, seed: int, record: bool, out_dir: Path, n_agents: int, current: _CurrentSeries | None) -> dict:
    """
    Single-agent route following: follow a set of waypoints under drift; report cross-track error and completion.
    """
    import numpy as np

    dt = 1.0 / float(cfg.ticks_per_sec)
    state = env.tick(num_ticks=1, publish=False)
    start = _pose_xyz(_state_get(state, "auv0", "PoseSensor", n_agents))

    d = str(cfg.difficulty).lower().strip()
    if d == "easy":
        r = 25.0
        wp_radius = 5.0
    elif d == "hard":
        r = 60.0
        wp_radius = 3.0
    else:
        r = 40.0
        wp_radius = 4.0

    wps = [
        (start[0] + r, start[1] + 0.0),
        (start[0] + r, start[1] + r),
        (start[0] + 0.0, start[1] + r),
        (start[0] - r, start[1] + r),
        (start[0] - r, start[1] + 0.0),
    ]
    seg_prev = (start[0], start[1])
    wp_idx = 0

    energy = 0.0
    collisions = 0
    prev_collision = False
    xte_sum = 0.0
    xte_n = 0

    rec: _Recorder | None = None
    if record:
        out_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(120):
            if _state_get(state, "auv0", "ViewportCapture", n_agents) is not None and _state_get(state, "auv0", "LeftCamera", n_agents) is not None:
                break
            state = env.tick(num_ticks=1, publish=False)
        rec = _Recorder(out_dir / "route_viewport.mp4", out_dir / "route_leftcamera.mp4", fps=cfg.fps)
        rec.start(_state_get(state, "auv0", "ViewportCapture", n_agents), _state_get(state, "auv0", "LeftCamera", n_agents))

    success = False
    steps = 0
    for step in range(int(cfg.max_steps)):
        steps = step + 1
        u, v = _episode_current_uv(current, cfg=cfg, seed=seed, sim_time_s=float(step) * dt)
        _apply_constant_current_drift(env, state=state, agent_names=["auv0"], n_agents=n_agents, u=u, v=v, dt=dt)

        tx, ty = wps[min(wp_idx, len(wps) - 1)]
        env.act("auv0", np.array([tx, ty, start[2], 0.0, 0.0, 0.0], dtype=np.float32))
        state = env.tick(num_ticks=1, publish=False)

        pose = _state_get(state, "auv0", "PoseSensor", n_agents)
        if pose is not None:
            px, py, _pz = _pose_xyz(pose)
            xte_sum += _dist_point_to_segment_xy(px, py, seg_prev[0], seg_prev[1], tx, ty)
            xte_n += 1
            if math.dist((px, py), (tx, ty)) <= float(wp_radius):
                seg_prev = (tx, ty)
                wp_idx += 1
                if wp_idx >= len(wps):
                    success = True
                    break

        col = _state_get(state, "auv0", "CollisionSensor", n_agents)
        if col is not None:
            now = bool(np.asarray(col).reshape(-1)[0])
            if now and not prev_collision:
                collisions += 1
            prev_collision = now
        vel = _state_get(state, "auv0", "VelocitySensor", n_agents)
        if vel is not None:
            v0 = np.asarray(vel, dtype=np.float32)
            energy += float(v0.dot(v0)) * dt

        if rec is not None:
            rec.append(_state_get(state, "auv0", "ViewportCapture", n_agents), _state_get(state, "auv0", "LeftCamera", n_agents))

    if rec is not None:
        rec.close()

    res = {
        "task": "route_following_waypoints",
        "seed": int(seed),
        "n_agents": int(n_agents),
        "difficulty": str(cfg.difficulty),
        "success": bool(success),
        "steps": int(steps),
        "time_s": float(steps * dt),
        "waypoints": int(len(wps)),
        "waypoints_reached": int(min(wp_idx, len(wps))),
        "mean_cross_track_error_m": float(xte_sum / float(max(1, xte_n))),
        "collisions": int(collisions),
        "energy_proxy": float(energy),
    }
    if record:
        res["media"] = {"viewport_mp4": str(out_dir / "route_viewport.mp4"), "leftcamera_mp4": str(out_dir / "route_leftcamera.mp4")}
    return res


def _run_depth_profile_tracking(env, *, cfg: SuiteCfg, seed: int, record: bool, out_dir: Path, n_agents: int, current: _CurrentSeries | None) -> dict:
    """
    Single-agent depth profile tracking: follow a piecewise depth schedule under drift.
    """
    import numpy as np

    dt = 1.0 / float(cfg.ticks_per_sec)
    steps = max(1, int(round(cfg.station_keep_seconds / dt)))
    state = env.tick(num_ticks=1, publish=False)
    start = _pose_xyz(_state_get(state, "auv0", "PoseSensor", n_agents))

    d = str(cfg.difficulty).lower().strip()
    if d == "easy":
        amp = 6.0
        rms_thresh = 4.0
    elif d == "hard":
        amp = 18.0
        rms_thresh = 2.0
    else:
        amp = 12.0
        rms_thresh = 3.0

    z0 = float(start[2])
    schedule = [z0, z0 - amp, z0 - 2.0 * amp, z0 - amp, z0]
    seg_len = max(1, steps // max(1, len(schedule) - 1))

    sq_depth = 0.0
    collisions = 0
    prev_collision = False
    energy = 0.0

    rec: _Recorder | None = None
    if record:
        out_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(120):
            if _state_get(state, "auv0", "ViewportCapture", n_agents) is not None and _state_get(state, "auv0", "LeftCamera", n_agents) is not None:
                break
            state = env.tick(num_ticks=1, publish=False)
        rec = _Recorder(out_dir / "depth_viewport.mp4", out_dir / "depth_leftcamera.mp4", fps=cfg.fps)
        rec.start(_state_get(state, "auv0", "ViewportCapture", n_agents), _state_get(state, "auv0", "LeftCamera", n_agents))

    for step in range(steps):
        u, v = _episode_current_uv(current, cfg=cfg, seed=seed, sim_time_s=float(step) * dt)
        _apply_constant_current_drift(env, state=state, agent_names=["auv0"], n_agents=n_agents, u=u, v=v, dt=dt)

        seg = min(len(schedule) - 2, step // seg_len)
        t = (step - seg * seg_len) / float(max(1, seg_len))
        z_t = float(schedule[seg] * (1.0 - t) + schedule[seg + 1] * t)
        env.act("auv0", np.array([start[0], start[1], z_t, 0.0, 0.0, 0.0], dtype=np.float32))
        state = env.tick(num_ticks=1, publish=False)

        pose = _state_get(state, "auv0", "PoseSensor", n_agents)
        if pose is not None:
            _x, _y, pz = _pose_xyz(pose)
            sq_depth += float((pz - z_t) ** 2)

        col = _state_get(state, "auv0", "CollisionSensor", n_agents)
        if col is not None:
            now = bool(np.asarray(col).reshape(-1)[0])
            if now and not prev_collision:
                collisions += 1
            prev_collision = now
        vel = _state_get(state, "auv0", "VelocitySensor", n_agents)
        if vel is not None:
            v0 = np.asarray(vel, dtype=np.float32)
            energy += float(v0.dot(v0)) * dt

        if rec is not None:
            rec.append(_state_get(state, "auv0", "ViewportCapture", n_agents), _state_get(state, "auv0", "LeftCamera", n_agents))

    if rec is not None:
        rec.close()

    rms = math.sqrt(sq_depth / float(max(1, steps)))
    res = {
        "task": "depth_profile_tracking",
        "seed": int(seed),
        "n_agents": int(n_agents),
        "difficulty": str(cfg.difficulty),
        "success": bool(rms <= float(rms_thresh)),
        "steps": int(steps),
        "time_s": float(steps * dt),
        "rms_depth_error_m": float(rms),
        "rms_success_threshold_m": float(rms_thresh),
        "collisions": int(collisions),
        "energy_proxy": float(energy),
    }
    if record:
        res["media"] = {"viewport_mp4": str(out_dir / "depth_viewport.mp4"), "leftcamera_mp4": str(out_dir / "depth_leftcamera.mp4")}
    return res


def _run_formation_transit_multiagent(env, *, cfg: SuiteCfg, seed: int, record: bool, out_dir: Path, n_agents: int, current: _CurrentSeries | None) -> dict:
    """
    Multi-agent formation transit: a leader navigates to a goal while followers maintain a circular formation.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    dt = 1.0 / float(cfg.ticks_per_sec)
    agent_names = [f"auv{i}" for i in range(n_agents)]

    state = env.tick(num_ticks=1, publish=False)
    p0 = _pose_xyz(_state_get(state, "auv0", "PoseSensor", n_agents))
    ang = float(rng.uniform(0.0, 2.0 * math.pi))
    goal = (p0[0] + cfg.nav_goal_dist_m * math.cos(ang), p0[1] + cfg.nav_goal_dist_m * math.sin(ang), p0[2])

    # Formation offsets in leader frame.
    form_r = 10.0 if str(cfg.difficulty).lower().strip() != "hard" else 14.0
    offsets = []
    for i in range(n_agents):
        if i == 0:
            offsets.append((0.0, 0.0, 0.0))
            continue
        a = 2.0 * math.pi * ((i - 1) / float(max(1, n_agents - 1)))
        offsets.append((form_r * math.cos(a), form_r * math.sin(a), 0.0))

    energy = 0.0
    collisions = 0
    prev_collision = False
    form_err_sq = 0.0
    form_err_n = 0

    rec: _Recorder | None = None
    if record:
        out_dir.mkdir(parents=True, exist_ok=True)
        for _ in range(240):
            if _state_get(state, "auv0", "ViewportCapture", n_agents) is not None and _state_get(state, "auv0", "LeftCamera", n_agents) is not None:
                break
            state = env.tick(num_ticks=1, publish=False)
        rec = _Recorder(out_dir / "formation_viewport.mp4", out_dir / "formation_leftcamera.mp4", fps=cfg.fps)
        rec.start(_state_get(state, "auv0", "ViewportCapture", n_agents), _state_get(state, "auv0", "LeftCamera", n_agents))

    success = False
    steps = 0
    for step in range(int(cfg.max_steps)):
        steps = step + 1
        u, v = _episode_current_uv(current, cfg=cfg, seed=seed, sim_time_s=float(step) * dt)
        _apply_constant_current_drift(env, state=state, agent_names=agent_names, n_agents=n_agents, u=u, v=v, dt=dt)

        # Leader drives to goal.
        env.act("auv0", np.array([goal[0], goal[1], goal[2], 0.0, 0.0, 0.0], dtype=np.float32))

        # Followers maintain formation around leader's *current* position.
        leader_pose = _state_get(state, "auv0", "PoseSensor", n_agents)
        if leader_pose is not None:
            lx, ly, lz = _pose_xyz(leader_pose)
            for i, name in enumerate(agent_names[1:], start=1):
                ox, oy, oz = offsets[i]
                env.act(name, np.array([lx + ox, ly + oy, lz + oz, 0.0, 0.0, 0.0], dtype=np.float32))

        state = env.tick(num_ticks=1, publish=False)

        # Formation error (RMS follower position error vs desired offsets in XY).
        leader_pose2 = _state_get(state, "auv0", "PoseSensor", n_agents)
        if leader_pose2 is not None:
            lx, ly, lz = _pose_xyz(leader_pose2)
            for i, name in enumerate(agent_names[1:], start=1):
                pose_i = _state_get(state, name, "PoseSensor", n_agents)
                if pose_i is None:
                    continue
                px, py, pz = _pose_xyz(pose_i)
                ox, oy, oz = offsets[i]
                dx = (px - (lx + ox))
                dy = (py - (ly + oy))
                dz = (pz - (lz + oz))
                form_err_sq += float(dx * dx + dy * dy + dz * dz)
                form_err_n += 1

            d_goal = float(math.dist((lx, ly, lz), goal))
            if d_goal <= max(6.0, float(cfg.nav_goal_radius_m)):
                success = True
                break

        col = _state_get(state, "auv0", "CollisionSensor", n_agents)
        if col is not None:
            now = bool(np.asarray(col).reshape(-1)[0])
            if now and not prev_collision:
                collisions += 1
            prev_collision = now
        vel = _state_get(state, "auv0", "VelocitySensor", n_agents)
        if vel is not None:
            v0 = np.asarray(vel, dtype=np.float32)
            energy += float(v0.dot(v0)) * dt

        if rec is not None:
            rec.append(_state_get(state, "auv0", "ViewportCapture", n_agents), _state_get(state, "auv0", "LeftCamera", n_agents))

    if rec is not None:
        rec.close()

    rms_form = math.sqrt(form_err_sq / float(max(1, form_err_n)))
    d = str(cfg.difficulty).lower().strip()
    if d == "easy":
        rms_thresh = 6.0
    elif d == "hard":
        rms_thresh = 3.0
    else:
        rms_thresh = 4.5

    res = {
        "task": "formation_transit_multiagent",
        "seed": int(seed),
        "n_agents": int(n_agents),
        "difficulty": str(cfg.difficulty),
        "goal_xyz": [float(goal[0]), float(goal[1]), float(goal[2])],
        "success": bool(success and rms_form <= rms_thresh),
        "steps": int(steps),
        "time_s": float(steps * dt),
        "rms_formation_error_m": float(rms_form),
        "rms_success_threshold_m": float(rms_thresh),
        "collisions": int(collisions),
        "energy_proxy": float(energy),
    }
    if record:
        res["media"] = {"viewport_mp4": str(out_dir / "formation_viewport.mp4"), "leftcamera_mp4": str(out_dir / "formation_leftcamera.mp4")}
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
    agent_names = [f"auv{i}" for i in range(n_agents)]
    start = _pose_xyz(_state_get(state, "auv0", "PoseSensor", n_agents))

    # Hidden source location (near start, but not identical).
    d = str(cfg.difficulty).lower().strip()
    if d == "easy":
        src_range = 30.0
    elif d == "hard":
        src_range = 70.0
    else:
        src_range = 50.0
    src = (
        start[0] + float(rng.uniform(-src_range, src_range)),
        start[1] + float(rng.uniform(-src_range, src_range)),
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
                initial_concentration=0.0,
                emission_rate=float(cfg.ocpnet_emission_rate),
            ),
            work_dir=out_dir / "_ocpnet_cache",
            world_center_xyz=(start[0], start[1], start[2]),
        )
        plume.set_source_world((src[0], src[1], src[2]))

        # Warmup plume before agent starts sweeping.
        for i in range(max(0, int(cfg.ocpnet_warmup_steps))):
            u0, v0 = _episode_current_uv(current, cfg=cfg, seed=seed, sim_time_s=float(i) * dt)
            plume.step(u_mps=float(u0), v_mps=float(v0))

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

    # Waypoint sweep targets (hold each target long enough for the PD controller to actually travel).
    steps = min(max(1, int(cfg.plume_localization_steps)), max(1, int(cfg.max_steps)) * 10)

    grid_span = float(src_range)
    if src_range <= 35.0:
        grid_step = 20.0
    elif src_range <= 60.0:
        grid_step = 30.0
    else:
        grid_step = 40.0

    dxs: list[float] = []
    dys: list[float] = []
    x = -grid_span
    while x <= grid_span + 1e-6:
        dxs.append(float(x))
        x += grid_step
    y = -grid_span
    while y <= grid_span + 1e-6:
        dys.append(float(y))
        y += grid_step

    waypoints: list[tuple[float, float]] = []
    flip = False
    for dy in dys:
        row = list(dxs)
        if flip:
            row.reverse()
        for dx in row:
            waypoints.append((start[0] + dx, start[1] + dy))
        flip = not flip
    if not waypoints:
        waypoints = [(start[0], start[1])]

    max_hold_ticks = int(max(int(cfg.ticks_per_sec) * 3, math.ceil(steps / float(max(1, len(waypoints))))))
    # Per-agent waypoint cursors: interleave coverage.
    wp_idx = [i % max(1, len(waypoints)) for i in range(n_agents)]
    hold_ticks = [0 for _ in range(n_agents)]
    prev_xy = [(start[0], start[1]) for _ in range(n_agents)]
    traveled = [0.0 for _ in range(n_agents)]
    max_radius_from_start_m = 0.0
    min_dist_to_source_m = float("inf")
    w_sum = 0.0
    wx_sum = 0.0
    wy_sum = 0.0

    for k in range(steps):
        u, v = _episode_current_uv(current, cfg=cfg, seed=seed, sim_time_s=float(k) * dt)
        _apply_constant_current_drift(
            env,
            state=state,
            agent_names=agent_names,
            n_agents=n_agents,
            u=u,
            v=v,
            dt=dt,
        )
        if plume is not None:
            plume.step(u_mps=float(u), v_mps=float(v))

        # Each agent sweeps a different waypoint stream.
        for i, name in enumerate(agent_names):
            tx, ty = waypoints[min(wp_idx[i], len(waypoints) - 1)]
            env.act(name, np.array([tx, ty, start[2], 0.0, 0.0, 0.0], dtype=np.float32))

        state = env.tick(num_ticks=1, publish=False)

        for i, name in enumerate(agent_names):
            pose = _state_get(state, name, "PoseSensor", n_agents)
            if pose is None:
                continue
            px, py, _pz = _pose_xyz(pose)

            traveled[i] += math.dist(prev_xy[i], (px, py))
            prev_xy[i] = (px, py)
            max_radius_from_start_m = max(max_radius_from_start_m, math.dist((px, py), (start[0], start[1])))
            min_dist_to_source_m = min(min_dist_to_source_m, math.dist((px, py), (src[0], src[1])))

            c = conc(px, py, start[2])
            if c > best_c:
                best_c = c
                best_xy = (px, py)
            w = float(max(0.0, c)) ** 3
            w_sum += w
            wx_sum += w * float(px)
            wy_sum += w * float(py)

            tx, ty = waypoints[min(wp_idx[i], len(waypoints) - 1)]
            if wp_idx[i] < len(waypoints) - 1:
                if math.dist((px, py), (tx, ty)) <= 2.5:
                    wp_idx[i] += n_agents
                    hold_ticks[i] = 0
                else:
                    hold_ticks[i] += 1
                    if hold_ticks[i] >= max_hold_ticks:
                        wp_idx[i] += n_agents
                        hold_ticks[i] = 0

        if rec is not None:
            rec.append(_state_get(state, "auv0", "ViewportCapture", n_agents), _state_get(state, "auv0", "LeftCamera", n_agents))

    if w_sum > 0.0:
        est_xy = (wx_sum / w_sum, wy_sum / w_sum)
    else:
        est_xy = (best_xy[0], best_xy[1])

    err = math.dist((est_xy[0], est_xy[1]), (src[0], src[1]))
    success = err <= cfg.plume_success_radius_m
    if rec is not None:
        rec.close()

    res = {
        "task": "plume_localization",
        "seed": int(seed),
        "n_agents": int(n_agents),
        "difficulty": str(cfg.difficulty),
        "pollution_model": str(cfg.pollution_model),
        "source_range_m": float(src_range),
        "source_xy": [float(src[0]), float(src[1])],
        "estimate_xy": [float(est_xy[0]), float(est_xy[1])],
        "argmax_xy": [float(best_xy[0]), float(best_xy[1])],
        "best_concentration": float(best_c),
        "error_m": float(err),
        "mean_traveled_m": float(sum(traveled) / float(max(1, len(traveled)))),
        "max_radius_from_start_m": float(max_radius_from_start_m),
        "min_dist_to_source_m": float(min_dist_to_source_m),
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
                initial_concentration=0.0,
                emission_rate=float(cfg.ocpnet_emission_rate),
            ),
            work_dir=out_dir / "_ocpnet_cache",
            world_center_xyz=(p0[0], p0[1], p0[2]),
        )
        plume.set_source_world((src[0], src[1], src[2]))
    else:
        particles = np.zeros((0, 3), dtype=np.float32)

    # Warmup plume to create an initial field; optionally freeze the source so containment is meaningful.
    warmup = max(0, int(cfg.ocpnet_warmup_steps)) if plume is not None else 0
    for i in range(warmup):
        u0, v0 = _episode_current_uv(current, cfg=cfg, seed=seed, sim_time_s=float(i) * dt)
        plume.step(u_mps=float(u0), v_mps=float(v0))  # type: ignore[union-attr]
    if plume is not None and bool(cfg.ocpnet_freeze_source_after_warmup):
        plume.freeze_source()

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

    steps = min(max(1, int(cfg.plume_containment_steps)), max(1, int(cfg.max_steps)) * 10)
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
        "ocpnet_freeze_source_after_warmup": bool(cfg.ocpnet_freeze_source_after_warmup),
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
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Reuse existing per-task results under --out_dir and run only missing tasks.",
    )
    ap.add_argument(
        "--tasks",
        nargs="*",
        default=None,
        help=(
            "Optional subset of tasks to run. Match by canonical task_id (e.g., formation_transit_multiagent), "
            "by backend alias (e.g., plume_containment_multiagent), or by dir name (e.g., surface_pollution_cleanup_multiagent__containment)."
        ),
    )
    ap.add_argument("--show_viewport", action="store_true")
    ap.add_argument("--current_npz", default=None, help="Optional exported current series npz (data-grounded forcing).")
    ap.add_argument("--current_depth_m", type=float, default=SuiteCfg.current_depth_m)
    ap.add_argument("--dataset_days_per_sim_second", type=float, default=SuiteCfg.dataset_days_per_sim_second)
    ap.add_argument("--pollution_model", default=SuiteCfg.pollution_model, choices=("analytic", "ocpnet_3d"))
    ap.add_argument("--ocpnet_warmup_steps", type=int, default=SuiteCfg.ocpnet_warmup_steps)
    ap.add_argument("--ocpnet_emission_rate", type=float, default=SuiteCfg.ocpnet_emission_rate)
    ap.add_argument(
        "--ocpnet_freeze_source_after_warmup",
        action=argparse.BooleanOptionalAction,
        default=SuiteCfg.ocpnet_freeze_source_after_warmup,
    )
    ap.add_argument("--plume_localization_steps", type=int, default=SuiteCfg.plume_localization_steps)
    ap.add_argument("--plume_containment_steps", type=int, default=SuiteCfg.plume_containment_steps)
    args = ap.parse_args()

    _ensure_ssl_cert_file()
    _ensure_nofile_limit()

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
        ocpnet_warmup_steps=int(args.ocpnet_warmup_steps),
        ocpnet_emission_rate=float(args.ocpnet_emission_rate),
        ocpnet_freeze_source_after_warmup=bool(args.ocpnet_freeze_source_after_warmup),
        plume_localization_steps=int(args.plume_localization_steps),
        plume_containment_steps=int(args.plume_containment_steps),
    )
    cfg = _cfg_with_difficulty(cfg, str(args.difficulty))
    scenarios = list(args.scenarios) if args.scenarios else scenario_preset(args.preset)

    out_root = Path(args.out_dir) if args.out_dir else Path("runs") / "oceangym_h3" / f"task_suite_{_tag_now_local()}"
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    git = _git_info(_REPO_ROOT)
    suite_manifest = {
        "track": "h3_oceangym",
        "script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "command": list(sys.argv),
        "git": git,
        "out_dir": str(out_root),
        "cfg": asdict(cfg),
        "scenarios": {},
    }
    root_media_manifest: dict[str, object] = {
        "track": "h3_oceangym",
        "script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "command": list(sys.argv),
        "git": git,
        "out_dir": str(out_root),
        "cfg": asdict(cfg),
        "scenarios": {},
    }
    suite_manifest["task_models"] = {"current": "constant_or_npz_series", "pollution_model": str(cfg.pollution_model)}
    current_series = _CurrentSeries(cfg.current_npz, depth_m=cfg.current_depth_m) if cfg.current_npz else None
    if current_series is not None:
        data_grounding = {
            "current_npz": current_series.path,
            "source_dataset": current_series.source_dataset,
            "lat": current_series.lat,
            "lon": current_series.lon,
            "depth_selected_m": current_series.depth_selected_m,
            "dataset_days_per_sim_second": float(cfg.dataset_days_per_sim_second),
        }
        suite_manifest["data_grounding"] = data_grounding
        root_media_manifest["data_grounding"] = data_grounding

    # Canonical task ids (see project/h_track_requirements.md).
    tasks: list[dict[str, object]] = [
        {"task_id": "go_to_goal_current", "task_alias": "go_to_goal_current", "task_variant": "", "n_agents": 1},
        {"task_id": "station_keeping", "task_alias": "station_keeping", "task_variant": "", "n_agents": 1},
        {"task_id": "route_following_waypoints", "task_alias": "route_following_waypoints", "task_variant": "", "n_agents": 1},
        {"task_id": "depth_profile_tracking", "task_alias": "depth_profile_tracking", "task_variant": "", "n_agents": 1},
        {
            "task_id": "formation_transit_multiagent",
            "task_alias": "formation_transit_multiagent",
            "task_variant": "",
            "n_agents": int(cfg.n_multiagent),
        },
        # Pollution family: we normalize to the canonical demo Task 1 id, and record the backend alias as metadata.
        {
            "task_id": "surface_pollution_cleanup_multiagent",
            "task_alias": "plume_localization",
            "task_variant": "localization",
            "n_agents": int(cfg.n_multiagent),
        },
        {
            "task_id": "surface_pollution_cleanup_multiagent",
            "task_alias": "plume_containment_multiagent",
            "task_variant": "containment",
            "n_agents": int(cfg.n_multiagent),
        },
    ]
    if args.tasks:
        wanted = {str(x).strip() for x in list(args.tasks) if str(x).strip()}

        def _match(t: dict[str, object]) -> bool:
            tid = str(t.get("task_id", ""))
            alias = str(t.get("task_alias", ""))
            var = str(t.get("task_variant", "") or "")
            dn = tid if not var else f"{tid}__{var}"
            return tid in wanted or alias in wanted or dn in wanted

        tasks = [t for t in tasks if _match(t)]
        if not tasks:
            raise ValueError(f"--tasks did not match any known tasks: {sorted(wanted)!r}")

    for scenario_name in scenarios:
        base = pm.get_scenario(scenario_name)
        per = {
            "scenario_name": scenario_name,
            "episodes": [],
            "media": {},
        }
        per_media: dict[str, object] = {}

        for t in tasks:
            task_id = str(t["task_id"])
            task_alias = str(t["task_alias"])
            task_variant = str(t.get("task_variant", "") or "")
            n_agents = int(t["n_agents"])

            scenario = _patch_for_suite(base, cfg=cfg, add_viewport=True, n_agents=n_agents)
            dir_name = task_id if not task_variant else f"{task_id}__{task_variant}"
            task_dir = out_root / scenario_name.replace("/", "_") / dir_name
            task_dir.mkdir(parents=True, exist_ok=True)

            existing_results = task_dir / "results_manifest.json"
            if bool(args.resume) and existing_results.exists():
                try:
                    loaded = json.loads(existing_results.read_text(encoding="utf-8"))
                except Exception:
                    loaded = None
                if isinstance(loaded, dict) and isinstance(loaded.get("episodes"), list) and len(loaded["episodes"]) == int(cfg.episodes):
                    per["episodes"].append(loaded)
                    media_task: dict[str, object] = {
                        "task": task_alias,
                        "task_id": task_id,
                        "task_alias": task_alias,
                        "task_variant": task_variant,
                        "n_agents": int(n_agents),
                        "episodes": {},
                    }
                    episodes_dict = media_task["episodes"]
                    if isinstance(episodes_dict, dict):
                        for i, ep in enumerate(loaded.get("episodes", [])):
                            m = ep.get("media") if isinstance(ep, dict) else None
                            if isinstance(m, dict) and m:
                                episodes_dict[f"ep{i:03d}"] = dict(m)
                    per_media[dir_name] = media_task
                    continue

            per_task = {
                "task": task_alias,
                "task_id": task_id,
                "task_alias": task_alias,
                "task_variant": task_variant,
                "n_agents": int(n_agents),
                "episodes": [],
            }
            per_task_media_manifest: dict[str, str] = {}
            media_task: dict[str, object] = {
                "task": task_alias,
                "task_id": task_id,
                "task_alias": task_alias,
                "task_variant": task_variant,
                "n_agents": int(n_agents),
                "episodes": {},
            }

            record_this = True
            with holoocean.make(
                scenario_cfg=scenario,
                show_viewport=cfg.show_viewport,
                ticks_per_sec=cfg.ticks_per_sec,
                frames_per_sec=cfg.fps,
                verbose=False,
            ) as env:
                env.set_render_quality(int(cfg.render_quality))
                env.should_render_viewport(True)

                for ep in range(cfg.episodes):
                    seed = _stable_seed(scenario_name, task_id, task_variant, task_alias, ep)
                    record = bool(not cfg.record_first_episode_only or ep == 0)
                    record_this = False

                    # Reset between episodes to avoid cumulative drift and to reduce resource churn.
                    state = env.reset()

                    # Warmup for sensors.
                    for _ in range(120):
                        if n_agents == 1:
                            ok = ("PoseSensor" in state)
                        else:
                            ok = ("auv0" in state and "PoseSensor" in state["auv0"])
                        if ok:
                            break
                        state = env.tick(num_ticks=1, publish=False)

                    if task_alias == "go_to_goal_current":
                        res = _run_nav_go_to_goal(
                            env,
                            cfg=cfg,
                            seed=seed,
                            record=record,
                            out_dir=task_dir / f"ep{ep:03d}",
                            n_agents=n_agents,
                            current=current_series,
                        )
                    elif task_alias == "station_keeping":
                        res = _run_station_keeping(
                            env,
                            cfg=cfg,
                            seed=seed,
                            record=record,
                            out_dir=task_dir / f"ep{ep:03d}",
                            n_agents=n_agents,
                            current=current_series,
                        )
                    elif task_alias == "route_following_waypoints":
                        res = _run_route_following_waypoints(
                            env,
                            cfg=cfg,
                            seed=seed,
                            record=record,
                            out_dir=task_dir / f"ep{ep:03d}",
                            n_agents=n_agents,
                            current=current_series,
                        )
                    elif task_alias == "depth_profile_tracking":
                        res = _run_depth_profile_tracking(
                            env,
                            cfg=cfg,
                            seed=seed,
                            record=record,
                            out_dir=task_dir / f"ep{ep:03d}",
                            n_agents=n_agents,
                            current=current_series,
                        )
                    elif task_alias == "formation_transit_multiagent":
                        res = _run_formation_transit_multiagent(
                            env,
                            cfg=cfg,
                            seed=seed,
                            record=record,
                            out_dir=task_dir / f"ep{ep:03d}",
                            n_agents=n_agents,
                            current=current_series,
                        )
                    elif task_alias == "plume_localization":
                        res = _run_plume_localization(
                            env,
                            cfg=cfg,
                            seed=seed,
                            record=record,
                            out_dir=task_dir / f"ep{ep:03d}",
                            n_agents=n_agents,
                            current=current_series,
                        )
                    elif task_alias == "plume_containment_multiagent":
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
                        raise ValueError(task_alias)

                    # Normalize ids for unified paper tables.
                    res["task_id"] = task_id
                    res["task_alias"] = task_alias
                    res["task_variant"] = task_variant

                    per_task["episodes"].append(res)
                    if "media" in res:
                        for k, v in dict(res["media"]).items():
                            per_task_media_manifest[f"ep{ep:03d}_{k}"] = str(v)
                        episodes_dict = media_task["episodes"]
                        if isinstance(episodes_dict, dict):
                            episodes_dict[f"ep{ep:03d}"] = dict(res["media"])

            per_task["summary"] = _summarize_task(task_alias, list(per_task["episodes"]))

            # Write per-task manifest.
            (task_dir / "media_manifest.json").write_text(
                json.dumps(per_task_media_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (task_dir / "results_manifest.json").write_text(json.dumps(per_task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            (task_dir / "metrics.json").write_text(json.dumps(per_task["summary"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
            per["episodes"].append(per_task)
            per_media[dir_name] = media_task

        suite_manifest["scenarios"][scenario_name] = per
        per_media["scenario_name"] = scenario_name
        root_media_manifest["scenarios"][scenario_name] = per_media

    (out_root / "results_manifest.json").write_text(json.dumps(suite_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_root / "media_manifest.json").write_text(
        json.dumps(root_media_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary_path = _write_summary_csv(out_root, suite_manifest)
    print("[h3] wrote:", summary_path)
    print("[h3] wrote:", out_root / "results_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
