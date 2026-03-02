from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict
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


def _git_sha(repo_root: Path) -> str | None:
    try:
        out = subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
        return out or None
    except Exception:
        return None


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_summary_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _task_set() -> list[str]:
    # H2 must-do set per project/h_track_requirements.md
    return [
        "go_to_goal_current",
        "station_keeping",
        "surface_pollution_cleanup_multiagent",
        "route_following_waypoints",
        "depth_profile_tracking",
        "formation_transit_multiagent",
    ]


def _difficulty_presets(task_id: str, difficulty: str) -> dict:
    d = str(difficulty).lower().strip()
    if d not in {"easy", "medium", "hard"}:
        raise ValueError(f"Unknown difficulty='{difficulty}'. Expected easy|medium|hard.")

    base = {
        "fps": 20,
        "render_quality": 1,
        "current_scale": 1.0,
        "current_force_scale": 6.0,
        "kp_xy": 6.0,
        "kd_xy": 3.0,
        "kp_z": 10.0,
        "kd_z": 4.0,
        "max_planar_force": 18.0,
        "max_vertical_force": 25.0,
    }

    if task_id in {"go_to_goal_current", "station_keeping", "route_following_waypoints", "depth_profile_tracking", "formation_transit_multiagent"}:
        if d == "easy":
            # Keep current injection non-zero but small so easy runs succeed.
            base |= {"current_scale": 0.15, "current_force_scale": 2.0}
        elif d == "medium":
            base |= {"current_scale": 1.0, "current_force_scale": 7.0}
        else:
            base |= {"current_scale": 1.3, "current_force_scale": 9.0}

    if task_id == "surface_pollution_cleanup_multiagent":
        if d == "easy":
            base |= {
                "current_scale": 0.30,
                "current_force_scale": 3.0,
                "success_radius_m": 30.0,
                "localize_seconds": 35.0,
                "contain_seconds": 70.0,
                "contain_radius_m": 5.0,
                "contain_tolerance_m": 12.0,
                "cleanup_success_mass_frac": 0.70,
                "coverage_success_min": 0.6,
                "kp_xy": 10.0,
                "kd_xy": 4.5,
                "max_planar_force": 90.0,
                "max_vertical_force": 60.0,
            }
        elif d == "medium":
            base |= {
                "current_scale": 0.8,
                "current_force_scale": 5.5,
                "success_radius_m": 22.0,
                "localize_seconds": 45.0,
                "contain_seconds": 90.0,
                "contain_radius_m": 10.0,
                "contain_tolerance_m": 7.0,
                "cleanup_success_mass_frac": 0.65,
                "coverage_success_min": 0.65,
            }
        else:
            base |= {
                "current_scale": 1.1,
                "current_force_scale": 7.5,
                "success_radius_m": 16.0,
                "localize_seconds": 55.0,
                "contain_seconds": 110.0,
                "contain_radius_m": 12.0,
                "contain_tolerance_m": 6.0,
                "cleanup_success_mass_frac": 0.60,
                "coverage_success_min": 0.7,
            }

    if d == "easy" and task_id in {"go_to_goal_current", "route_following_waypoints"}:
        # Easy should succeed reliably; raise thrust limits for faster motion.
        base |= {"kp_xy": 12.0, "kd_xy": 5.0, "max_planar_force": 90.0, "max_vertical_force": 60.0}

    if d == "easy" and task_id == "formation_transit_multiagent":
        # Formation is sensitive to leader speed; keep thrust moderate so followers can maintain spacing.
        base |= {"kp_xy": 12.0, "kd_xy": 5.0, "max_planar_force": 55.0, "max_vertical_force": 60.0}

    return dict(base)


def _goal_distance_m(difficulty: str) -> float:
    return {"easy": 10.0, "medium": 35.0, "hard": 55.0}[difficulty]


def _goal_radius_m(difficulty: str) -> float:
    return {"easy": 10.0, "medium": 5.0, "hard": 3.5}[difficulty]


def _station_radius_m(difficulty: str) -> float:
    return {"easy": 6.0, "medium": 4.0, "hard": 3.0}[difficulty]


def _waypoint_spacing_m(difficulty: str) -> float:
    return {"easy": 8.0, "medium": 25.0, "hard": 35.0}[difficulty]


def _formation_radius_m(num_agents: int, difficulty: str) -> float:
    base = 6.0 if num_agents <= 4 else 8.0
    if difficulty == "easy":
        return base
    if difficulty == "medium":
        return base * 1.2
    return base * 1.4


def _formation_tol_m(difficulty: str) -> float:
    return {"easy": 8.0, "medium": 2.8, "hard": 2.0}[difficulty]


def _episode_seconds(task_id: str, difficulty: str) -> float:
    if task_id == "go_to_goal_current":
        return {"easy": 25.0, "medium": 16.0, "hard": 20.0}[difficulty]
    if task_id == "station_keeping":
        return {"easy": 16.0, "medium": 20.0, "hard": 26.0}[difficulty]
    if task_id == "route_following_waypoints":
        return {"easy": 30.0, "medium": 24.0, "hard": 30.0}[difficulty]
    if task_id == "depth_profile_tracking":
        return {"easy": 18.0, "medium": 22.0, "hard": 28.0}[difficulty]
    if task_id == "formation_transit_multiagent":
        return {"easy": 30.0, "medium": 22.0, "hard": 28.0}[difficulty]
    if task_id == "surface_pollution_cleanup_multiagent":
        return {"easy": 70.0, "medium": 90.0, "hard": 110.0}[difficulty]
    raise ValueError(f"Unknown task_id='{task_id}'.")


def _warmup_and_get_state(env, rp, *, num_agents: int) -> tuple[dict, list[float]]:
    st = None
    for _ in range(220):
        st = rp._safe_tick(env, publish=False)
        a0 = rp._state_agent(st, "auv0")
        if a0.get("PoseSensor") is not None and a0.get("ViewportCapture") is not None and (rp._get_fpv_frame(a0) is not None):
            break
    if st is None:
        raise RuntimeError("Failed to warm up environment.")
    p0 = rp._pose_to_position(rp._state_agent(st, "auv0").get("PoseSensor"))
    if p0 is None:
        raise RuntimeError("PoseSensor missing for auv0.")
    return st, p0


def _record_common(*, env, rp, cfg, task_dir: Path, steps: int, per_step_fn, state_init: dict, first_agent_z: float) -> dict:
    import imageio.v2 as imageio

    rp._apply_viewport_underwater_hack(env, cfg, first_agent_z=float(first_agent_z))
    mp4_path = task_dir / "rollout.mp4"
    gif_path = task_dir / "rollout.gif"
    fpv_mp4_path = task_dir / "rollout_fpv.mp4"
    fpv_gif_path = task_dir / "rollout_fpv.gif"
    start_png = task_dir / "start.png"
    end_png = task_dir / "end.png"
    start_fpv_png = task_dir / "start_fpv.png"
    end_fpv_png = task_dir / "end_fpv.png"

    gif_frames = []
    fpv_gif_frames = []
    gif_stride = max(1, int(round(float(cfg.fps) / 8.0)))
    last_fpv = None

    st = state_init
    with rp._Mp4Writer(mp4_path, fps=cfg.fps) as vw, rp._Mp4Writer(fpv_mp4_path, fps=cfg.fps) as fw:
        for t in range(int(steps)):
            st, _step = per_step_fn(int(t), st)
            a0 = rp._state_agent(st, "auv0")
            frame = a0.get("ViewportCapture")
            last_fpv = rp._get_fpv_frame(a0)
            if frame is not None:
                rgb = rp._maybe_expose(rp._ensure_uint8_rgb(frame), cfg.viewport_exposure)
                vw.append(rgb)
                if t == 10:
                    imageio.imwrite(start_png, rgb)
                if (t % gif_stride) == 0:
                    gif_frames.append(rp._downscale_for_gif(rgb, target_width=480))
            if last_fpv is not None:
                rgb = rp._maybe_expose(rp._ensure_uint8_rgb(last_fpv), cfg.fpv_exposure)
                fw.append(rgb)
                if t == 10:
                    imageio.imwrite(start_fpv_png, rgb)
                if (t % gif_stride) == 0:
                    fpv_gif_frames.append(rp._downscale_for_gif(rgb, target_width=480))
        if frame is not None:
            imageio.imwrite(end_png, rp._maybe_expose(rp._ensure_uint8_rgb(frame), cfg.viewport_exposure))
        if last_fpv is not None:
            imageio.imwrite(end_fpv_png, rp._maybe_expose(rp._ensure_uint8_rgb(last_fpv), cfg.fpv_exposure))

    rp._write_gif(gif_path, gif_frames, fps=8)
    rp._write_gif(fpv_gif_path, fpv_gif_frames, fps=8)
    return {
        "video": str(mp4_path),
        "gif": str(gif_path),
        "video_fpv": str(fpv_mp4_path),
        "gif_fpv": str(fpv_gif_path),
        "start_png": str(start_png),
        "end_png": str(end_png),
        "start_fpv_png": str(start_fpv_png),
        "end_fpv_png": str(end_fpv_png),
    }


def _symlink_or_copy(dst: Path, src: Path) -> None:
    try:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    except Exception:
        try:
            import shutil

            shutil.copy2(src, dst)
        except Exception:
            pass


def _run_go_to_goal_current(*, env, rp, cfg, current, out_dir: Path, difficulty: str) -> dict:
    import numpy as np

    task_id = "go_to_goal_current"
    task_dir = out_dir / f"task_{task_id}_{difficulty}"
    task_dir.mkdir(parents=True, exist_ok=True)
    st, p0 = _warmup_and_get_state(env, rp, num_agents=cfg.num_agents)
    first_z = float(p0[2])
    dt = 1.0 / float(cfg.fps)
    steps = int(round(_episode_seconds(task_id, difficulty) * float(cfg.fps)))
    d_goal = float(_goal_distance_m(difficulty))
    goal_xy = (float(p0[0] + d_goal), float(p0[1] + 0.35 * d_goal))
    goal_z = float(p0[2])
    offsets = {}
    for i in range(int(cfg.num_agents)):
        pi = rp._pose_to_position(rp._state_agent(st, f"auv{i}").get("PoseSensor")) or p0
        offsets[f"auv{i}"] = (float(pi[0] - p0[0]), float(pi[1] - p0[1]))
    energy = 0.0
    collisions = 0
    prev_colliding: dict[str, bool] = {}
    success_step = None
    goal_err_sum = 0.0
    goal_err_max = 0.0

    def per_step_fn(t: int, st_in: dict) -> tuple[dict, dict]:
        nonlocal energy, collisions, success_step, goal_err_sum, goal_err_max
        for i in range(int(cfg.num_agents)):
            name = f"auv{i}"
            ai = rp._state_agent(st_in, name)
            pos = rp._pose_to_position(ai.get("PoseSensor"))
            vel = ai.get("VelocitySensor")
            vxyz = np.asarray(vel, dtype=np.float32).reshape(3) if vel is not None else np.zeros((3,), dtype=np.float32)
            if pos is None:
                continue
            off = offsets[name]
            target = [goal_xy[0] + off[0], goal_xy[1] + off[1], goal_z]
            ex = float(target[0]) - float(pos[0])
            ey = float(target[1]) - float(pos[1])
            ez = float(target[2]) - float(pos[2])
            fx = cfg.kp_xy * ex - cfg.kd_xy * float(vxyz[0])
            fy = cfg.kp_xy * ey - cfg.kd_xy * float(vxyz[1])
            fz = cfg.kp_z * ez - cfg.kd_z * float(vxyz[2])
            u, v = current.velocity_xy_mps(float(pos[0]), float(pos[1]))
            fx += cfg.current_force_scale * cfg.current_scale * u
            fy += cfg.current_force_scale * cfg.current_scale * v
            bx, by, bz = rp._world_force_to_body(ai.get("PoseSensor"), fx, fy, fz)
            act = rp._action_from_force(bx, by, bz, cfg)
            energy += float(np.sum(act * act)) * dt
            env.act(name, act)
        st_out = rp._safe_tick(env, publish=False)
        collisions += rp._count_collision_events(st_out, num_agents=cfg.num_agents, prev=prev_colliding)
        p = rp._pose_to_position(rp._state_agent(st_out, "auv0").get("PoseSensor"))
        if p is not None:
            e = float(np.hypot(float(p[0]) - float(goal_xy[0]), float(p[1]) - float(goal_xy[1])))
            goal_err_sum += e
            goal_err_max = max(goal_err_max, e)
            if success_step is None and e <= float(_goal_radius_m(difficulty)):
                success_step = int(t)
        return st_out, {}

    media = _record_common(env=env, rp=rp, cfg=cfg, task_dir=task_dir, steps=steps, per_step_fn=per_step_fn, state_init=st, first_agent_z=first_z)
    metrics = {
        "task_id": task_id,
        "difficulty": difficulty,
        "seed": int(cfg.seed),
        "n_agents": int(cfg.num_agents),
        "steps": int(steps),
        "dt_s": float(dt),
        "success": success_step is not None,
        "success_step": int(success_step) if success_step is not None else None,
        "time_to_success_s": (float(success_step + 1) * dt) if success_step is not None else None,
        "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
        "goal_radius_m": float(_goal_radius_m(difficulty)),
        "mean_goal_error_m": float(goal_err_sum / float(max(1, steps))),
        "max_goal_error_m": float(goal_err_max),
        "collisions": int(collisions),
        "energy_proxy": float(energy),
    }
    _write_json(task_dir / "metrics.json", metrics)
    return {"task_dir": str(task_dir), "metrics": metrics, **media}


def _run_station_keeping(*, env, rp, cfg, current, out_dir: Path, difficulty: str) -> dict:
    import numpy as np

    task_id = "station_keeping"
    task_dir = out_dir / f"task_{task_id}_{difficulty}"
    task_dir.mkdir(parents=True, exist_ok=True)
    st, p0 = _warmup_and_get_state(env, rp, num_agents=cfg.num_agents)
    first_z = float(p0[2])
    dt = 1.0 / float(cfg.fps)
    steps = int(round(_episode_seconds(task_id, difficulty) * float(cfg.fps)))
    radius = float(_station_radius_m(difficulty))
    setpoints = {}
    for i in range(int(cfg.num_agents)):
        pi = rp._pose_to_position(rp._state_agent(st, f"auv{i}").get("PoseSensor")) or p0
        setpoints[f"auv{i}"] = [float(pi[0]), float(pi[1]), float(pi[2])]
    energy = 0.0
    collisions = 0
    prev_colliding: dict[str, bool] = {}
    success_step = None
    err_sum = 0.0
    err_max = 0.0
    hold_s = 6.0 if difficulty == "easy" else 8.0
    hold_steps = int(round(hold_s * float(cfg.fps)))
    in_radius_hist = []

    def per_step_fn(t: int, st_in: dict) -> tuple[dict, dict]:
        nonlocal energy, collisions, success_step, err_sum, err_max
        all_in = True
        for i in range(int(cfg.num_agents)):
            name = f"auv{i}"
            ai = rp._state_agent(st_in, name)
            pos = rp._pose_to_position(ai.get("PoseSensor"))
            vel = ai.get("VelocitySensor")
            vxyz = np.asarray(vel, dtype=np.float32).reshape(3) if vel is not None else np.zeros((3,), dtype=np.float32)
            if pos is None:
                all_in = False
                continue
            sp = setpoints[name]
            ex = float(sp[0]) - float(pos[0])
            ey = float(sp[1]) - float(pos[1])
            ez = float(sp[2]) - float(pos[2])
            fx = cfg.kp_xy * ex - cfg.kd_xy * float(vxyz[0])
            fy = cfg.kp_xy * ey - cfg.kd_xy * float(vxyz[1])
            fz = cfg.kp_z * ez - cfg.kd_z * float(vxyz[2])
            u, v = current.velocity_xy_mps(float(pos[0]), float(pos[1]))
            fx += cfg.current_force_scale * cfg.current_scale * u
            fy += cfg.current_force_scale * cfg.current_scale * v
            bx, by, bz = rp._world_force_to_body(ai.get("PoseSensor"), fx, fy, fz)
            act = rp._action_from_force(bx, by, bz, cfg)
            energy += float(np.sum(act * act)) * dt
            env.act(name, act)
            e = float(np.sqrt(ex * ex + ey * ey + ez * ez))
            err_sum += e
            err_max = max(err_max, e)
            if e > radius:
                all_in = False
        in_radius_hist.append(bool(all_in))
        if len(in_radius_hist) > hold_steps:
            in_radius_hist.pop(0)
        if success_step is None and len(in_radius_hist) == hold_steps and all(in_radius_hist):
            success_step = int(t)
        st_out = rp._safe_tick(env, publish=False)
        collisions += rp._count_collision_events(st_out, num_agents=cfg.num_agents, prev=prev_colliding)
        return st_out, {}

    media = _record_common(env=env, rp=rp, cfg=cfg, task_dir=task_dir, steps=steps, per_step_fn=per_step_fn, state_init=st, first_agent_z=first_z)
    metrics = {
        "task_id": task_id,
        "difficulty": difficulty,
        "seed": int(cfg.seed),
        "n_agents": int(cfg.num_agents),
        "steps": int(steps),
        "dt_s": float(dt),
        "success": success_step is not None,
        "success_step": int(success_step) if success_step is not None else None,
        "time_to_success_s": (float(success_step + 1) * dt) if success_step is not None else None,
        "station_radius_m": float(radius),
        "station_error_mean_m": float(err_sum / float(max(1, steps * max(1, int(cfg.num_agents))))),
        "station_error_max_m": float(err_max),
        "collisions": int(collisions),
        "energy_proxy": float(energy),
    }
    _write_json(task_dir / "metrics.json", metrics)
    return {"task_dir": str(task_dir), "metrics": metrics, **media}


def _run_route_following_waypoints(*, env, rp, cfg, current, out_dir: Path, difficulty: str) -> dict:
    import numpy as np

    task_id = "route_following_waypoints"
    task_dir = out_dir / f"task_{task_id}_{difficulty}"
    task_dir.mkdir(parents=True, exist_ok=True)
    st, p0 = _warmup_and_get_state(env, rp, num_agents=cfg.num_agents)
    first_z = float(p0[2])
    dt = 1.0 / float(cfg.fps)
    steps = int(round(_episode_seconds(task_id, difficulty) * float(cfg.fps)))
    spacing = float(_waypoint_spacing_m(difficulty))
    waypoints = [
        [float(p0[0] + spacing), float(p0[1]), float(p0[2])],
        [float(p0[0] + spacing), float(p0[1] + spacing), float(p0[2])],
        [float(p0[0]), float(p0[1] + spacing), float(p0[2])],
    ]
    wp_tol = float(_goal_radius_m(difficulty))
    wp_idx = 0
    success_step = None
    last_p0 = None
    offsets = {}
    for i in range(int(cfg.num_agents)):
        pi = rp._pose_to_position(rp._state_agent(st, f"auv{i}").get("PoseSensor")) or p0
        offsets[f"auv{i}"] = (float(pi[0] - p0[0]), float(pi[1] - p0[1]))
    energy = 0.0
    collisions = 0
    prev_colliding: dict[str, bool] = {}

    def per_step_fn(t: int, st_in: dict) -> tuple[dict, dict]:
        nonlocal energy, collisions, wp_idx, success_step, last_p0
        target0 = waypoints[min(int(wp_idx), len(waypoints) - 1)]
        for i in range(int(cfg.num_agents)):
            name = f"auv{i}"
            ai = rp._state_agent(st_in, name)
            pos = rp._pose_to_position(ai.get("PoseSensor"))
            vel = ai.get("VelocitySensor")
            vxyz = np.asarray(vel, dtype=np.float32).reshape(3) if vel is not None else np.zeros((3,), dtype=np.float32)
            if pos is None:
                continue
            off = offsets[name]
            target = [float(target0[0] + off[0]), float(target0[1] + off[1]), float(target0[2])]
            ex = float(target[0]) - float(pos[0])
            ey = float(target[1]) - float(pos[1])
            ez = float(target[2]) - float(pos[2])
            fx = cfg.kp_xy * ex - cfg.kd_xy * float(vxyz[0])
            fy = cfg.kp_xy * ey - cfg.kd_xy * float(vxyz[1])
            fz = cfg.kp_z * ez - cfg.kd_z * float(vxyz[2])
            u, v = current.velocity_xy_mps(float(pos[0]), float(pos[1]))
            fx += cfg.current_force_scale * cfg.current_scale * u
            fy += cfg.current_force_scale * cfg.current_scale * v
            bx, by, bz = rp._world_force_to_body(ai.get("PoseSensor"), fx, fy, fz)
            act = rp._action_from_force(bx, by, bz, cfg)
            energy += float(np.sum(act * act)) * dt
            env.act(name, act)
        st_out = rp._safe_tick(env, publish=False)
        collisions += rp._count_collision_events(st_out, num_agents=cfg.num_agents, prev=prev_colliding)
        p = rp._pose_to_position(rp._state_agent(st_out, "auv0").get("PoseSensor"))
        if p is not None:
            last_p0 = [float(p[0]), float(p[1]), float(p[2])]
        if p is not None and wp_idx < len(waypoints):
            e = float(np.hypot(float(p[0]) - float(target0[0]), float(p[1]) - float(target0[1])))
            if e <= wp_tol:
                wp_idx += 1
                if wp_idx >= len(waypoints) and success_step is None:
                    success_step = int(t)
        return st_out, {}

    media = _record_common(env=env, rp=rp, cfg=cfg, task_dir=task_dir, steps=steps, per_step_fn=per_step_fn, state_init=st, first_agent_z=first_z)
    p = last_p0 or p0
    last_wp = waypoints[-1]
    final_err = float(np.hypot(float(p[0]) - float(last_wp[0]), float(p[1]) - float(last_wp[1])))
    metrics = {
        "task_id": task_id,
        "difficulty": difficulty,
        "seed": int(cfg.seed),
        "n_agents": int(cfg.num_agents),
        "steps": int(steps),
        "dt_s": float(dt),
        "success": success_step is not None,
        "success_step": int(success_step) if success_step is not None else None,
        "time_to_success_s": (float(success_step + 1) * dt) if success_step is not None else None,
        "waypoint_count": int(len(waypoints)),
        "waypoint_index": int(min(wp_idx, len(waypoints) - 1)),
        "waypoint_tolerance_m": float(wp_tol),
        "final_waypoint_error_m": float(final_err),
        "collisions": int(collisions),
        "energy_proxy": float(energy),
    }
    _write_json(task_dir / "metrics.json", metrics)
    return {"task_dir": str(task_dir), "metrics": metrics, **media}


def _run_depth_profile_tracking(*, env, rp, cfg, current, out_dir: Path, difficulty: str) -> dict:
    import numpy as np

    task_id = "depth_profile_tracking"
    task_dir = out_dir / f"task_{task_id}_{difficulty}"
    task_dir.mkdir(parents=True, exist_ok=True)
    st, p0 = _warmup_and_get_state(env, rp, num_agents=cfg.num_agents)
    first_z = float(p0[2])
    dt = 1.0 / float(cfg.fps)
    steps = int(round(_episode_seconds(task_id, difficulty) * float(cfg.fps)))
    target_xy = (float(p0[0]), float(p0[1]))
    depth0 = float(max(3.0, min(20.0, -float(p0[2]))))
    amp = {"easy": 2.0, "medium": 3.5, "hard": 5.0}[difficulty]
    period_s = {"easy": 10.0, "medium": 8.0, "hard": 6.5}[difficulty]
    tol = {"easy": 3.0, "medium": 2.0, "hard": 1.5}[difficulty]
    energy = 0.0
    collisions = 0
    prev_colliding: dict[str, bool] = {}
    success_step = None
    z_errs = []

    def target_depth_abs(t: int) -> float:
        tt = float(t) * dt
        return float(depth0 + amp * np.sin(2.0 * np.pi * (tt / float(period_s))))

    def per_step_fn(t: int, st_in: dict) -> tuple[dict, dict]:
        nonlocal energy, collisions
        depth_abs = target_depth_abs(t)
        target_z = -float(depth_abs)
        for i in range(int(cfg.num_agents)):
            name = f"auv{i}"
            ai = rp._state_agent(st_in, name)
            pos = rp._pose_to_position(ai.get("PoseSensor"))
            vel = ai.get("VelocitySensor")
            vxyz = np.asarray(vel, dtype=np.float32).reshape(3) if vel is not None else np.zeros((3,), dtype=np.float32)
            if pos is None:
                continue
            dx = 2.0 * (i % 5) - 4.0
            dy = 2.0 * (i // 5) - 1.0
            target = [target_xy[0] + dx, target_xy[1] + dy, target_z]
            ex = float(target[0]) - float(pos[0])
            ey = float(target[1]) - float(pos[1])
            ez = float(target[2]) - float(pos[2])
            fx = cfg.kp_xy * ex - cfg.kd_xy * float(vxyz[0])
            fy = cfg.kp_xy * ey - cfg.kd_xy * float(vxyz[1])
            fz = cfg.kp_z * ez - cfg.kd_z * float(vxyz[2])
            u, v = current.velocity_xy_mps(float(pos[0]), float(pos[1]))
            fx += cfg.current_force_scale * cfg.current_scale * u
            fy += cfg.current_force_scale * cfg.current_scale * v
            bx, by, bz = rp._world_force_to_body(ai.get("PoseSensor"), fx, fy, fz)
            act = rp._action_from_force(bx, by, bz, cfg)
            energy += float(np.sum(act * act)) * dt
            env.act(name, act)
        st_out = rp._safe_tick(env, publish=False)
        collisions += rp._count_collision_events(st_out, num_agents=cfg.num_agents, prev=prev_colliding)
        p = rp._pose_to_position(rp._state_agent(st_out, "auv0").get("PoseSensor"))
        if p is not None:
            z_errs.append(abs(float(p[2]) - float(target_z)))
        return st_out, {}

    media = _record_common(env=env, rp=rp, cfg=cfg, task_dir=task_dir, steps=steps, per_step_fn=per_step_fn, state_init=st, first_agent_z=first_z)
    if z_errs:
        rmse = float((sum(e * e for e in z_errs) / float(len(z_errs))) ** 0.5)
        max_err = float(max(z_errs))
    else:
        rmse, max_err = float("inf"), float("inf")
    if rmse <= tol and max_err <= (1.5 * tol):
        success_step = int(steps - 1)
    metrics = {
        "task_id": task_id,
        "difficulty": difficulty,
        "seed": int(cfg.seed),
        "n_agents": int(cfg.num_agents),
        "steps": int(steps),
        "dt_s": float(dt),
        "success": bool(success_step is not None),
        "success_step": int(success_step) if success_step is not None else None,
        "time_to_success_s": (float(success_step + 1) * dt) if success_step is not None else None,
        "depth_rmse_m": float(rmse),
        "max_depth_error_m": float(max_err),
        "depth_tol_m": float(tol),
        "collisions": int(collisions),
        "energy_proxy": float(energy),
    }
    _write_json(task_dir / "metrics.json", metrics)
    return {"task_dir": str(task_dir), "metrics": metrics, **media}


def _run_formation_transit_multiagent(*, env, rp, cfg, current, out_dir: Path, difficulty: str) -> dict:
    import numpy as np

    task_id = "formation_transit_multiagent"
    task_dir = out_dir / f"task_{task_id}_{difficulty}"
    task_dir.mkdir(parents=True, exist_ok=True)
    st, p0 = _warmup_and_get_state(env, rp, num_agents=cfg.num_agents)
    first_z = float(p0[2])
    dt = 1.0 / float(cfg.fps)
    steps = int(round(_episode_seconds(task_id, difficulty) * float(cfg.fps)))
    d_goal = float(_goal_distance_m(difficulty)) * 1.1
    goal_xy = (float(p0[0] + d_goal), float(p0[1] - 0.2 * d_goal))
    goal_radius = float(_goal_radius_m(difficulty))
    r_form = float(_formation_radius_m(int(cfg.num_agents), difficulty))
    tol_form = float(_formation_tol_m(difficulty))
    offsets = {"auv0": (0.0, 0.0)}
    if int(cfg.num_agents) > 1:
        for i in range(1, int(cfg.num_agents)):
            ang = 2.0 * np.pi * (float(i - 1) / float(max(1, int(cfg.num_agents) - 1)))
            offsets[f"auv{i}"] = (float(r_form * np.cos(ang)), float(r_form * np.sin(ang)))
    energy = 0.0
    collisions = 0
    prev_colliding: dict[str, bool] = {}
    success_step = None
    form_err_sum = 0.0
    form_err_max = 0.0
    leader_goal_err_min = float("inf")

    def per_step_fn(t: int, st_in: dict) -> tuple[dict, dict]:
        nonlocal energy, collisions, success_step, form_err_sum, form_err_max, leader_goal_err_min
        pL = rp._pose_to_position(rp._state_agent(st_in, "auv0").get("PoseSensor")) or p0
        vL_raw = rp._state_agent(st_in, "auv0").get("VelocitySensor")
        vL = np.asarray(vL_raw, dtype=np.float32).reshape(3) if vL_raw is not None else np.zeros((3,), dtype=np.float32)
        ex = float(goal_xy[0]) - float(pL[0])
        ey = float(goal_xy[1]) - float(pL[1])
        ez = float(p0[2]) - float(pL[2])
        fx = cfg.kp_xy * ex - cfg.kd_xy * float(vL[0])
        fy = cfg.kp_xy * ey - cfg.kd_xy * float(vL[1])
        fz = cfg.kp_z * ez - cfg.kd_z * float(vL[2])
        u, v = current.velocity_xy_mps(float(pL[0]), float(pL[1]))
        fx += cfg.current_force_scale * cfg.current_scale * u
        fy += cfg.current_force_scale * cfg.current_scale * v
        aL = rp._state_agent(st_in, "auv0")
        bx0, by0, bz0 = rp._world_force_to_body(aL.get("PoseSensor"), fx, fy, fz)
        act0 = rp._action_from_force(bx0, by0, bz0, cfg)
        energy += float(np.sum(act0 * act0)) * dt
        env.act("auv0", act0)
        ferrs_this = []
        for i in range(1, int(cfg.num_agents)):
            name = f"auv{i}"
            ai = rp._state_agent(st_in, name)
            pos = rp._pose_to_position(ai.get("PoseSensor"))
            vel = ai.get("VelocitySensor")
            vxyz = np.asarray(vel, dtype=np.float32).reshape(3) if vel is not None else np.zeros((3,), dtype=np.float32)
            if pos is None:
                continue
            off = offsets[name]
            target = [float(pL[0] + off[0]), float(pL[1] + off[1]), float(p0[2])]
            ex = float(target[0]) - float(pos[0])
            ey = float(target[1]) - float(pos[1])
            ez = float(target[2]) - float(pos[2])
            fx = cfg.kp_xy * ex - cfg.kd_xy * float(vxyz[0])
            fy = cfg.kp_xy * ey - cfg.kd_xy * float(vxyz[1])
            fz = cfg.kp_z * ez - cfg.kd_z * float(vxyz[2])
            u, v = current.velocity_xy_mps(float(pos[0]), float(pos[1]))
            fx += cfg.current_force_scale * cfg.current_scale * u
            fy += cfg.current_force_scale * cfg.current_scale * v
            bx, by, bz = rp._world_force_to_body(ai.get("PoseSensor"), fx, fy, fz)
            act = rp._action_from_force(bx, by, bz, cfg)
            energy += float(np.sum(act * act)) * dt
            env.act(name, act)
            relx = float(pos[0] - pL[0]) - float(off[0])
            rely = float(pos[1] - pL[1]) - float(off[1])
            ferr = float(np.hypot(relx, rely))
            ferrs_this.append(float(ferr))
            form_err_sum += ferr
            form_err_max = max(form_err_max, ferr)
        st_out = rp._safe_tick(env, publish=False)
        collisions += rp._count_collision_events(st_out, num_agents=cfg.num_agents, prev=prev_colliding)
        pL2 = rp._pose_to_position(rp._state_agent(st_out, "auv0").get("PoseSensor"))
        if pL2 is not None:
            gerr = float(np.hypot(float(pL2[0]) - float(goal_xy[0]), float(pL2[1]) - float(goal_xy[1])))
            leader_goal_err_min = min(float(leader_goal_err_min), float(gerr))
            if success_step is None and gerr <= goal_radius:
                # Use *current* formation error near goal; cumulative error is too strict for a transit task.
                if ferrs_this:
                    mean_f = float(sum(ferrs_this) / float(len(ferrs_this)))
                    max_f = float(max(ferrs_this))
                else:
                    mean_f, max_f = 0.0, 0.0
                if mean_f <= tol_form and max_f <= (1.5 * tol_form):
                    success_step = int(t)
        return st_out, {}

    media = _record_common(env=env, rp=rp, cfg=cfg, task_dir=task_dir, steps=steps, per_step_fn=per_step_fn, state_init=st, first_agent_z=first_z)
    denom = float(max(1, steps * max(1, int(cfg.num_agents) - 1)))
    metrics = {
        "task_id": task_id,
        "difficulty": difficulty,
        "seed": int(cfg.seed),
        "n_agents": int(cfg.num_agents),
        "steps": int(steps),
        "dt_s": float(dt),
        "success": success_step is not None,
        "success_step": int(success_step) if success_step is not None else None,
        "time_to_success_s": (float(success_step + 1) * dt) if success_step is not None else None,
        "goal_xy": [float(goal_xy[0]), float(goal_xy[1])],
        "goal_radius_m": float(goal_radius),
        "leader_goal_error_min_m": None if not np.isfinite(leader_goal_err_min) else float(leader_goal_err_min),
        "formation_radius_m": float(r_form),
        "formation_tol_m": float(tol_form),
        "formation_error_mean_m": float(form_err_sum / denom),
        "formation_error_max_m": float(form_err_max),
        "collisions": int(collisions),
        "energy_proxy": float(energy),
    }
    _write_json(task_dir / "metrics.json", metrics)
    return {"task_dir": str(task_dir), "metrics": metrics, **media}


def _run_surface_pollution_cleanup_multiagent(*, env, rp, cfg, current, out_dir: Path, difficulty: str) -> dict:
    task_id = "surface_pollution_cleanup_multiagent"
    task_dir = out_dir / f"task_{task_id}_{difficulty}"
    task_dir.mkdir(parents=True, exist_ok=True)
    localize_res = rp.run_localize_task(env, cfg, current, task_dir)
    env.reset()
    env.set_render_quality(int(cfg.render_quality))
    env.should_render_viewport(True)
    rp._safe_tick(env, publish=False)
    contain_res = rp.run_contain_cleanup_task(env, cfg, current, task_dir)
    loc_m = dict(localize_res["metrics"])
    cc_m = dict(contain_res["metrics"])
    success = bool(loc_m.get("success")) and bool(cc_m.get("success"))
    t_s = None
    if success:
        t1, t2 = loc_m.get("time_to_success_s"), cc_m.get("time_to_success_s")
        if isinstance(t1, (int, float)) and isinstance(t2, (int, float)):
            t_s = float(t1) + float(t2)
    cc_dir = Path(contain_res["task_dir"])
    for name in ["rollout.mp4", "rollout_fpv.mp4", "rollout.gif", "rollout_fpv.gif", "start.png", "start_fpv.png", "end.png", "end_fpv.png"]:
        src = cc_dir / name
        if src.exists():
            _symlink_or_copy(task_dir / name, src)
    metrics = {
        "task_id": task_id,
        "difficulty": difficulty,
        "seed": int(cfg.seed),
        "n_agents": int(cfg.num_agents),
        "steps": int(cc_m.get("steps") or 0),
        "dt_s": float(cc_m.get("dt_s") or (1.0 / float(cfg.fps))),
        "success": success,
        "time_to_success_s": t_s,
        "localization_error_m": loc_m.get("localization_error_m"),
        "remaining_mass_frac_raw": cc_m.get("remaining_mass_frac_raw"),
        "remaining_mass_frac": cc_m.get("remaining_mass_frac"),
        "mean_contain_coverage_frac": cc_m.get("mean_contain_coverage_frac"),
        "max_leakage_proxy": cc_m.get("max_leakage_proxy"),
        "mean_leakage_proxy": cc_m.get("mean_leakage_proxy"),
        "cleanup_success_mass_frac": cc_m.get("cleanup_success_mass_frac"),
        "coverage_success_min": cc_m.get("coverage_success_min"),
        "leakage_success_threshold": cc_m.get("leakage_success_threshold"),
        "collisions": int((loc_m.get("collisions") or 0) + (cc_m.get("collisions") or 0)),
        "energy_proxy": float((loc_m.get("energy_proxy") or 0.0) + (cc_m.get("energy_proxy") or 0.0)),
        "phases": {"localize": localize_res, "contain_cleanup": contain_res},
    }
    _write_json(task_dir / "metrics.json", metrics)
    return {
        "task_dir": str(task_dir),
        "metrics": metrics,
        "phases": {"localize": localize_res, "contain_cleanup": contain_res},
        "canonical_metrics_json": str((task_dir / "metrics.json").resolve()),
        "video": str((task_dir / "rollout.mp4").resolve()),
        "video_fpv": str((task_dir / "rollout_fpv.mp4").resolve()),
        "gif": str((task_dir / "rollout.gif").resolve()),
        "gif_fpv": str((task_dir / "rollout_fpv.gif").resolve()),
        "start_png": str((task_dir / "start.png").resolve()),
        "start_fpv_png": str((task_dir / "start_fpv.png").resolve()),
        "end_png": str((task_dir / "end.png").resolve()),
        "end_fpv_png": str((task_dir / "end_fpv.png").resolve()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=os.environ.get("OUT_DIR", f"runs/h2_holoocean/h2_suite_{_tag_now_local()}"))
    ap.add_argument("--scenario", type=str, default=os.environ.get("SCENARIO_NAME", "PierHarbor-HoveringCamera"))
    ap.add_argument("--num-agents", type=int, default=int(os.environ.get("NUM_AGENTS", "8")))
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    ap.add_argument("--fps", type=int, default=int(os.environ.get("FPS", "20")))
    ap.add_argument("--render-quality", type=int, default=int(os.environ.get("RENDER_QUALITY", "1")))
    ap.add_argument("--window-width", type=int, default=int(os.environ.get("WINDOW_WIDTH", "1280")))
    ap.add_argument("--window-height", type=int, default=int(os.environ.get("WINDOW_HEIGHT", "720")))
    ap.add_argument("--difficulties", type=str, default=os.environ.get("DIFFICULTIES", "easy"), help="Comma-separated: easy,medium,hard")
    ap.add_argument("--tasks", type=str, default=os.environ.get("TASKS", ",".join(_task_set())), help="Comma-separated canonical task ids.")
    ap.add_argument("--dataset-variant", type=str, default=os.environ.get("DATASET_VARIANT", "public"))
    ap.add_argument("--combined-nc", type=str, default=os.environ.get("COMBINED_NC", ""))
    ap.add_argument("--time-index", type=int, default=int(os.environ.get("TIME_INDEX", "0")))
    ap.add_argument("--depth-index", type=int, default=int(os.environ.get("DEPTH_INDEX", "0")))
    args = ap.parse_args()

    from tracks.h2_holoocean import run_plume_tasks as rp

    dataset_variant = str(args.dataset_variant).strip() or None
    combined_nc = rp._resolve_combined_nc(str(args.combined_nc), dataset_variant)
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = _REPO_ROOT / out_dir
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
    diffs = [d.strip() for d in str(args.difficulties).split(",") if d.strip()]
    allowed = set(_task_set())
    for t in tasks:
        if t not in allowed:
            raise ValueError(f"Unsupported task_id='{t}' for H2 suite. Allowed: {', '.join(_task_set())}")
    for d in diffs:
        if d not in {"easy", "medium", "hard"}:
            raise ValueError(f"Unsupported difficulty='{d}'. Allowed: easy,medium,hard")

    base_cfg = rp.RunnerCfg(
        scenario_name=str(args.scenario),
        num_agents=int(args.num_agents),
        seed=int(args.seed),
        fps=int(args.fps),
        window_width=int(args.window_width),
        window_height=int(args.window_height),
        render_quality=int(args.render_quality),
        combined_nc=str(combined_nc),
        time_index=int(args.time_index),
        depth_index=int(args.depth_index),
        pollution_warmup_s=10.0,
        pollution_domain_xy_m=160.0,
        localize_seconds=25.0,
        contain_seconds=45.0,
        cleanup_success_mass_frac=0.65,
        coverage_success_min=0.6,
        contain_tolerance_m=6.0,
        current_scale=1.0,
        current_force_scale=6.0,
    )

    import holoocean
    from holoocean.packagemanager import get_scenario

    current = rp.DatasetCurrent(str(combined_nc), int(args.time_index), int(args.depth_index))
    base_scenario = get_scenario(str(args.scenario))

    suite_outputs: dict[str, dict] = {}
    summary_rows: list[dict] = []

    media_manifest = {
        "created_at_utc": _utc_now(),
        "track": "h2_holoocean",
        "suite": "h2_suite",
        "script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "dataset_variant": dataset_variant,
        "combined_nc": str(combined_nc),
        "scenario_name": str(args.scenario),
        "git_sha": _git_sha(_REPO_ROOT),
        "worlds_roots_on_disk": rp._world_roots_on_disk(base_cfg.package_name),
        "provenance_note": "See tracks/h2_holoocean/scene_provenance.md for world package provenance/licensing notes.",
        "outputs": {},
    }

    for difficulty in diffs:
        for task_id in tasks:
            preset = _difficulty_presets(task_id, difficulty)
            merged = asdict(base_cfg) | preset
            merged["seed"] = int(base_cfg.seed)
            merged["num_agents"] = int(base_cfg.num_agents)
            merged["scenario_name"] = str(base_cfg.scenario_name)
            merged["combined_nc"] = str(base_cfg.combined_nc)
            merged["time_index"] = int(base_cfg.time_index)
            merged["depth_index"] = int(base_cfg.depth_index)
            cfg = rp.RunnerCfg(**merged)
            scenario = rp._patch_scenario_for_runner(base_scenario, cfg)
            key = f"{task_id}/{difficulty}"

            with holoocean.make(
                scenario_cfg=scenario,
                show_viewport=False,
                ticks_per_sec=int(cfg.fps),
                frames_per_sec=int(cfg.fps),
                verbose=False,
                copy_state=True,
            ) as env:
                env.set_render_quality(int(cfg.render_quality))
                env.should_render_viewport(True)
                rp._safe_tick(env, publish=False)

                if task_id == "go_to_goal_current":
                    res = _run_go_to_goal_current(env=env, rp=rp, cfg=cfg, current=current, out_dir=out_dir, difficulty=difficulty)
                elif task_id == "station_keeping":
                    res = _run_station_keeping(env=env, rp=rp, cfg=cfg, current=current, out_dir=out_dir, difficulty=difficulty)
                elif task_id == "route_following_waypoints":
                    res = _run_route_following_waypoints(env=env, rp=rp, cfg=cfg, current=current, out_dir=out_dir, difficulty=difficulty)
                elif task_id == "depth_profile_tracking":
                    res = _run_depth_profile_tracking(env=env, rp=rp, cfg=cfg, current=current, out_dir=out_dir, difficulty=difficulty)
                elif task_id == "formation_transit_multiagent":
                    res = _run_formation_transit_multiagent(env=env, rp=rp, cfg=cfg, current=current, out_dir=out_dir, difficulty=difficulty)
                elif task_id == "surface_pollution_cleanup_multiagent":
                    res = _run_surface_pollution_cleanup_multiagent(env=env, rp=rp, cfg=cfg, current=current, out_dir=out_dir, difficulty=difficulty)
                else:
                    raise AssertionError(task_id)

            suite_outputs[key] = {
                "task_id": task_id,
                "difficulty": difficulty,
                "seed": int(cfg.seed),
                "num_agents": int(cfg.num_agents),
                "out_dir": str(Path(res["task_dir"]).resolve()),
                "result": res,
            }
            media_manifest["outputs"][key] = res
            m = dict(res["metrics"])
            row = {
                "task_id": task_id,
                "difficulty": difficulty,
                "scenario": str(args.scenario),
                "seed": int(cfg.seed),
                "num_agents": int(cfg.num_agents),
                "dataset_variant": dataset_variant or "",
                "combined_nc": str(combined_nc),
            }
            for k, v in m.items():
                if k in {"phases"}:
                    continue
                row[k] = v
            summary_rows.append(row)

    _write_json(out_dir / "media_manifest.json", media_manifest)
    results_manifest = {
        "created_at_utc": _utc_now(),
        "track": "h2_holoocean",
        "suite": "h2_suite",
        "git_sha": _git_sha(_REPO_ROOT),
        "dataset_variant": dataset_variant,
        "combined_nc": str(combined_nc),
        "scenario": str(args.scenario),
        "seed": int(args.seed),
        "num_agents": int(args.num_agents),
        "difficulties": diffs,
        "tasks": tasks,
        "metrics_json": str((out_dir / "metrics.json").resolve()),
        "summary_csv": str((out_dir / "summary.csv").resolve()),
        "media_manifest_json": str((out_dir / "media_manifest.json").resolve()),
        "outputs": suite_outputs,
        "command_hint": (
            f"cd '{_REPO_ROOT}' && {sys.executable} tracks/h2_holoocean/run_h2_suite.py "
            f"--scenario {args.scenario} --dataset-variant {(dataset_variant or '')} "
            f"--num-agents {int(args.num_agents)} --seed {int(args.seed)} --difficulties {','.join(diffs)}"
        ),
    }
    _write_json(out_dir / "results_manifest.json", results_manifest)
    _write_json(out_dir / "metrics.json", {"created_at_utc": _utc_now(), "track": "h2_holoocean", "suite": "h2_suite", "outputs": suite_outputs})
    _write_summary_csv(out_dir / "summary.csv", summary_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
