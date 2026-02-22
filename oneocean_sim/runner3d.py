from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import json
from math import atan2, cos, sin, sqrt
from pathlib import Path
from time import perf_counter
from typing import Optional

import mujoco
import numpy as np
from PIL import Image

from .data import CurrentSampler, DatasetContext, open_dataset, resolve_dataset_path, validate_time_depth_indices
from .mujoco3d_env import Mujoco3DConfig, OceanMujoco3DEnv
from .mujoco3d_scene import ObstacleSpec
from .tasks.nav3d import Nav3DController, Nav3DTaskConfig, count_collisions, sample_obstacles
from .tasks.plume3d import (
    CastAndSurgeController,
    PlumeMultiAgentTaskConfig,
    plume_concentration,
    sample_source_xy,
)


SUPPORTED_TASKS_3D = ("nav_obstacles_3d", "plume_source_localization_3d")
SUPPORTED_CONTROLLERS_3D = ("auto", "compensated", "naive")


@dataclass
class Run3DConfig:
    task: str = "nav_obstacles_3d"
    controller: str = "auto"  # auto -> compensated
    variant: str = "scene"
    dataset_path: Optional[str] = None
    episodes: int = 3
    seed: int = 42
    time_index: int = 0
    depth_index: int = 0
    include_tides: bool = True

    # Scene/sim knobs
    dt_sec: float = 0.05
    max_steps: int = 900
    target_domain_size_m: float = 1000.0
    meters_per_sim_meter: Optional[float] = None
    current_scale: float = 80.0
    record_media: bool = True
    record_all_episodes: bool = False
    render_width: int = 960
    render_height: int = 544
    fps: int = 20
    camera: str = "cam_main"

    # Task knobs
    nav_goal_distance_m: float = 60.0
    nav_goal_tolerance_m: float = 12.0
    nav_obstacle_count: int = 14
    plume_detection_threshold: float = 0.16
    plume_source_tolerance_m: float = 10.0


def _default_output_dir(task: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"oneocean_{task}_3d_{stamp}"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _distance2(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return sqrt(dx * dx + dy * dy + dz * dz)


def _summarize(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    keys = [k for k in rows[0].keys() if isinstance(rows[0][k], (int, float))]
    out: dict[str, float] = {"episodes": float(len(rows))}
    for k in keys:
        values = np.asarray([float(r[k]) for r in rows], dtype=float)
        out[f"{k}_mean"] = float(values.mean())
        out[f"{k}_std"] = float(values.std())
    if "success" in rows[0]:
        out["success_rate"] = float(np.mean([r["success"] for r in rows]))
    return out


def _save_png(img: np.ndarray, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(img).save(path)
    return str(path)


def _record_episode_video(
    env: OceanMujoco3DEnv,
    *,
    out_dir: Path,
    episode: int,
    fps: int,
    width: int,
    height: int,
    camera: str,
    frames: list[np.ndarray],
) -> dict[str, str]:
    media_dir = out_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)
    start_png = media_dir / f"episode_{episode:03d}_start.png"
    end_png = media_dir / f"episode_{episode:03d}_end.png"
    mp4_path = media_dir / f"episode_{episode:03d}.mp4"

    if frames:
        _save_png(frames[0], start_png)
        _save_png(frames[-1], end_png)

        import imageio.v2 as imageio

        with imageio.get_writer(mp4_path, fps=int(fps)) as writer:
            for frame in frames:
                writer.append_data(frame)

    return {
        "start_png": str(start_png),
        "end_png": str(end_png),
        "mp4": str(mp4_path),
        "camera": str(camera),
        "fps": str(int(fps)),
        "width": str(int(width)),
        "height": str(int(height)),
    }


def _resolve_controller_mode(task: str, controller: str) -> str:
    if controller == "auto":
        return "compensated"
    if controller not in SUPPORTED_CONTROLLERS_3D:
        raise ValueError(f"Unsupported controller: {controller}")
    if controller == "compensated":
        return "compensated"
    if controller == "naive":
        return "naive"
    raise ValueError(f"Unsupported controller: {controller}")


def run_task_3d(
    *,
    config: Run3DConfig,
    output_dir: Optional[str],
    command: str,
) -> dict[str, str]:
    if config.task not in SUPPORTED_TASKS_3D:
        raise ValueError(f"Unsupported task: {config.task}")

    out_dir = Path(output_dir) if output_dir else _default_output_dir(config.task)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_path(config.dataset_path, config.variant)
    ds = open_dataset(dataset_path)
    time_index, depth_index = validate_time_depth_indices(ds, config.time_index, config.depth_index)
    context = DatasetContext(
        dataset_path=dataset_path,
        variant=config.variant,
        time_index=time_index,
        depth_index=depth_index,
    )
    sampler = CurrentSampler(ds, include_tides=config.include_tides)

    ctrl_mode = _resolve_controller_mode(config.task, config.controller)
    compensated = ctrl_mode == "compensated"

    sim_cfg = Mujoco3DConfig(
        dt_sec=float(config.dt_sec),
        max_steps=int(config.max_steps),
        target_domain_size_m=float(config.target_domain_size_m),
        meters_per_sim_meter=config.meters_per_sim_meter,
        current_speed_scale=float(config.current_scale),
    )

    per_episode_metrics: list[dict[str, float]] = []
    media_entries: list[dict[str, object]] = []

    trajectories_dir = out_dir / "trajectories"
    trajectories_dir.mkdir(exist_ok=True)

    for episode in range(int(config.episodes)):
        rng = np.random.default_rng(int(config.seed + episode))

        if config.task == "nav_obstacles_3d":
            nav_cfg = Nav3DTaskConfig(
                goal_distance_m=float(config.nav_goal_distance_m),
                goal_tolerance_m=float(config.nav_goal_tolerance_m),
                desired_depth_z_m=-4.0,
                max_steps=int(config.max_steps),
            )
            # Build a temporary env once to know x/y bounds for obstacle sampling.
            temp_env = OceanMujoco3DEnv(
                ds,
                sampler,
                variant=config.variant,
                time_index=time_index,
                depth_index=depth_index,
                seed=int(config.seed + episode),
                agent_count=1,
                obstacles=(),
                config=sim_cfg,
            )
            obstacles = sample_obstacles(
                rng,
                count=int(config.nav_obstacle_count),
                x_half_m=temp_env.x_half_m,
                y_half_m=temp_env.y_half_m,
                z_center_m=nav_cfg.desired_depth_z_m - 1.6,
            )
            env = OceanMujoco3DEnv(
                ds,
                sampler,
                variant=config.variant,
                time_index=time_index,
                depth_index=depth_index,
                seed=int(config.seed + episode),
                agent_count=1,
                obstacles=obstacles,
                config=sim_cfg,
            )

            # Start near one corner; goal at a random bearing with fixed distance.
            start = (
                float(rng.uniform(-0.7 * env.x_half_m, -0.2 * env.x_half_m)),
                float(rng.uniform(-0.7 * env.y_half_m, -0.2 * env.y_half_m)),
                float(nav_cfg.desired_depth_z_m),
            )
            bearing = float(rng.uniform(0.0, 2.0 * np.pi))
            goal = (
                float(np.clip(start[0] + nav_cfg.goal_distance_m * cos(bearing), -0.85 * env.x_half_m, 0.85 * env.x_half_m)),
                float(np.clip(start[1] + nav_cfg.goal_distance_m * sin(bearing), -0.85 * env.y_half_m, 0.85 * env.y_half_m)),
                float(nav_cfg.desired_depth_z_m),
            )
            env.reset(agent_xyz_m=[start], goal_xyz_m=goal, source_xyz_m=goal)

            controller = Nav3DController(
                max_speed_mps=float(nav_cfg.cruise_speed_mps),
                slowdown_radius_m=float(nav_cfg.slowdown_radius_m),
                compensate_current=bool(compensated),
                obstacle_influence_m=float(nav_cfg.obstacle_influence_m),
                obstacle_repulsion_gain=float(nav_cfg.obstacle_repulsion_gain),
            )

            renderer = None
            frames: list[np.ndarray] = []
            if config.record_media and (episode == 0 or config.record_all_episodes):
                renderer = mujoco.Renderer(env.model, height=int(config.render_height), width=int(config.render_width))

            trajectory: list[dict[str, object]] = []
            collisions = 0
            energy = 0.0
            t0 = perf_counter()
            success = False
            invalid_terminated = False

            for step in range(int(nav_cfg.max_steps)):
                obs = env.agent_state(0)
                action = controller.act(obs=obs, goal_xyz_m=goal, obstacles=obstacles, desired_depth_z_m=nav_cfg.desired_depth_z_m)
                env.step([action])

                post = env.agent_state(0)
                dist_goal = _distance2((post["x_m"], post["y_m"], post["z_m"]), goal)
                collisions_step = count_collisions(
                    agent_xyz_m=(post["x_m"], post["y_m"], post["z_m"]),
                    agent_radius_m=0.45,
                    obstacles=obstacles,
                )
                collisions += collisions_step
                speed_cmd = sqrt(float(action["vx_mps"]) ** 2 + float(action["vy_mps"]) ** 2)
                energy += (speed_cmd * speed_cmd) * float(sim_cfg.dt_sec)
                invalid = post["invalid_region"] > 0.5
                if invalid:
                    invalid_terminated = True

                if renderer is not None:
                    frames.append(env.render(renderer, camera=config.camera))

                trajectory.append(
                    {
                        "step": float(step),
                        "x_m": float(post["x_m"]),
                        "y_m": float(post["y_m"]),
                        "z_m": float(post["z_m"]),
                        "vx_mps": float(post["vx_mps"]),
                        "vy_mps": float(post["vy_mps"]),
                        "current_u_mps": float(post["current_u_mps"]),
                        "current_v_mps": float(post["current_v_mps"]),
                        "goal_x_m": float(goal[0]),
                        "goal_y_m": float(goal[1]),
                        "goal_z_m": float(goal[2]),
                        "distance_to_goal_m": float(dist_goal),
                        "cmd_vx_mps": float(action["vx_mps"]),
                        "cmd_vy_mps": float(action["vy_mps"]),
                        "collisions_step": float(collisions_step),
                        "invalid_region": float(invalid),
                    }
                )

                if dist_goal <= nav_cfg.goal_tolerance_m:
                    success = True
                    break
                if collisions_step > 0:
                    break
                if invalid_terminated:
                    break

            wall = float(perf_counter() - t0)
            steps = float(len(trajectory))
            time_sec = steps * float(sim_cfg.dt_sec)
            end = (trajectory[-1]["x_m"], trajectory[-1]["y_m"], trajectory[-1]["z_m"]) if trajectory else start
            path_len = 0.0
            if len(trajectory) >= 2:
                xs = np.asarray([r["x_m"] for r in trajectory], dtype=float)
                ys = np.asarray([r["y_m"] for r in trajectory], dtype=float)
                zs = np.asarray([r["z_m"] for r in trajectory], dtype=float)
                path_len = float(np.sum(np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2 + np.diff(zs) ** 2)))

            ep_metrics = {
                "task": 0.0,
                "task_name": config.task,
                "episode": float(episode),
                "success": float(success),
                "timeout": float(not success and steps >= nav_cfg.max_steps),
                "invalid_terminated": float(invalid_terminated),
                "collisions": float(collisions),
                "steps": float(steps),
                "time_sec": float(time_sec),
                "path_length_m": float(path_len),
                "energy_proxy": float(energy),
                "final_distance_to_goal_m": float(_distance2((float(end[0]), float(end[1]), float(end[2])), goal)),
                "wall_clock_sec": float(wall),
                "sim_steps_per_sec": float(steps / max(1e-9, wall)),
                "controller_compensates_current": float(compensated),
                "variant": config.variant,
                "time_index": float(time_index),
                "depth_index": float(depth_index),
            }
            per_episode_metrics.append(ep_metrics)

            traj_path = trajectories_dir / f"episode_{episode:03d}_agent0.csv"
            _write_csv(traj_path, trajectory)

            if renderer is not None:
                media = _record_episode_video(
                    env,
                    out_dir=out_dir,
                    episode=episode,
                    fps=int(config.fps),
                    width=int(config.render_width),
                    height=int(config.render_height),
                    camera=str(config.camera),
                    frames=frames,
                )
                media_entries.append(
                    {
                        "task": config.task,
                        "episode": int(episode),
                        "controller": ctrl_mode,
                        "files": media,
                        "generation_command": command,
                    }
                )
                renderer.close()

        else:
            plume_cfg = PlumeMultiAgentTaskConfig(
                detection_threshold=float(config.plume_detection_threshold),
                source_tolerance_m=float(config.plume_source_tolerance_m),
                desired_depth_z_m=-4.0,
                max_steps=int(config.max_steps),
            )
            plume_field_note = (
                "Plume concentration is a synthetic analytic field (anisotropic Gaussian aligned with local current). "
                "Currents are sampled from our dataset; do not interpret plume as measured pollution data."
            )

            env = OceanMujoco3DEnv(
                ds,
                sampler,
                variant=config.variant,
                time_index=time_index,
                depth_index=depth_index,
                seed=int(config.seed + episode),
                agent_count=2,
                obstacles=(),
                config=sim_cfg,
            )
            source_xy = sample_source_xy(rng, x_half_m=env.x_half_m, y_half_m=env.y_half_m)
            source = (float(source_xy[0]), float(source_xy[1]), float(plume_cfg.desired_depth_z_m))

            # Place the team downstream of the source so detection is feasible within the episode horizon.
            lat_s, lon_s = env.geo.xy_to_latlon(source[0], source[1])
            cu, cv = env.sim_uv_mps(lat_s, lon_s)
            c_norm = sqrt(cu * cu + cv * cv)
            if c_norm <= 1e-9:
                du, dv = 1.0, 0.0
            else:
                du, dv = cu / c_norm, cv / c_norm
            cross_u, cross_v = -dv, du
            downstream_offset = 32.0
            cross_offset = 7.0
            start0_xy = (
                float(source[0] + du * downstream_offset + cross_u * cross_offset),
                float(source[1] + dv * downstream_offset + cross_v * cross_offset),
            )
            start1_xy = (
                float(source[0] + du * downstream_offset - cross_u * cross_offset),
                float(source[1] + dv * downstream_offset - cross_v * cross_offset),
            )
            start0 = (
                float(np.clip(start0_xy[0], -0.85 * env.x_half_m, 0.85 * env.x_half_m)),
                float(np.clip(start0_xy[1], -0.85 * env.y_half_m, 0.85 * env.y_half_m)),
                float(plume_cfg.desired_depth_z_m),
            )
            start1 = (
                float(np.clip(start1_xy[0], -0.85 * env.x_half_m, 0.85 * env.x_half_m)),
                float(np.clip(start1_xy[1], -0.85 * env.y_half_m, 0.85 * env.y_half_m)),
                float(plume_cfg.desired_depth_z_m),
            )
            env.reset(agent_xyz_m=[start0, start1], goal_xyz_m=source, source_xyz_m=source)

            controllers = [
                CastAndSurgeController(
                    cruise_speed_mps=float(plume_cfg.cruise_speed_mps),
                    cast_speed_mps=float(plume_cfg.cast_speed_mps),
                    cast_period_steps=int(plume_cfg.cast_period_steps),
                    cast_sign=+1.0,
                    detection_threshold=float(plume_cfg.detection_threshold),
                    compensate_current=bool(compensated),
                ),
                CastAndSurgeController(
                    cruise_speed_mps=float(plume_cfg.cruise_speed_mps),
                    cast_speed_mps=float(plume_cfg.cast_speed_mps),
                    cast_period_steps=int(plume_cfg.cast_period_steps),
                    cast_sign=-1.0,
                    detection_threshold=float(plume_cfg.detection_threshold),
                    compensate_current=bool(compensated),
                ),
            ]

            renderer = None
            frames: list[np.ndarray] = []
            if config.record_media and (episode == 0 or config.record_all_episodes):
                renderer = mujoco.Renderer(env.model, height=int(config.render_height), width=int(config.render_width))

            traj0: list[dict[str, object]] = []
            traj1: list[dict[str, object]] = []
            energy = 0.0
            success = False
            first_detection_step: Optional[int] = None
            t0 = perf_counter()

            for step in range(int(plume_cfg.max_steps)):
                obs0 = env.agent_state(0)
                obs1 = env.agent_state(1)
                conc0 = plume_concentration(
                    xy_m=(obs0["x_m"], obs0["y_m"]),
                    source_xy_m=(source[0], source[1]),
                    current_uv_mps=(obs0["current_u_mps"], obs0["current_v_mps"]),
                    sigma_cross_m=float(plume_cfg.sigma_cross_m),
                    sigma_down_m=float(plume_cfg.sigma_down_m),
                    sigma_up_m=float(plume_cfg.sigma_up_m),
                    upstream_scale=float(plume_cfg.upstream_scale_m),
                )
                conc1 = plume_concentration(
                    xy_m=(obs1["x_m"], obs1["y_m"]),
                    source_xy_m=(source[0], source[1]),
                    current_uv_mps=(obs1["current_u_mps"], obs1["current_v_mps"]),
                    sigma_cross_m=float(plume_cfg.sigma_cross_m),
                    sigma_down_m=float(plume_cfg.sigma_down_m),
                    sigma_up_m=float(plume_cfg.sigma_up_m),
                    upstream_scale=float(plume_cfg.upstream_scale_m),
                )
                conc0 = float(max(0.0, conc0 + rng.normal(0.0, float(plume_cfg.sensor_noise_std))))
                conc1 = float(max(0.0, conc1 + rng.normal(0.0, float(plume_cfg.sensor_noise_std))))
                detected = (conc0 >= plume_cfg.detection_threshold) or (conc1 >= plume_cfg.detection_threshold)
                if detected and first_detection_step is None:
                    first_detection_step = int(step)

                act0 = controllers[0].act(step_index=step, obs=obs0, concentration=conc0, global_detected=bool(detected))
                act1 = controllers[1].act(step_index=step, obs=obs1, concentration=conc1, global_detected=bool(detected))
                act0["z_m"] = float(plume_cfg.desired_depth_z_m)
                act1["z_m"] = float(plume_cfg.desired_depth_z_m)
                env.step([act0, act1])

                post0 = env.agent_state(0)
                post1 = env.agent_state(1)
                d0 = _distance2((post0["x_m"], post0["y_m"], post0["z_m"]), source)
                d1 = _distance2((post1["x_m"], post1["y_m"], post1["z_m"]), source)
                min_d = min(d0, d1)

                speed0 = sqrt(float(act0["vx_mps"]) ** 2 + float(act0["vy_mps"]) ** 2)
                speed1 = sqrt(float(act1["vx_mps"]) ** 2 + float(act1["vy_mps"]) ** 2)
                energy += ((speed0 * speed0) + (speed1 * speed1)) * float(sim_cfg.dt_sec)

                if renderer is not None:
                    frames.append(env.render(renderer, camera=config.camera))

                traj0.append(
                    {
                        "step": float(step),
                        "x_m": float(post0["x_m"]),
                        "y_m": float(post0["y_m"]),
                        "z_m": float(post0["z_m"]),
                        "current_u_mps": float(post0["current_u_mps"]),
                        "current_v_mps": float(post0["current_v_mps"]),
                        "cmd_vx_mps": float(act0["vx_mps"]),
                        "cmd_vy_mps": float(act0["vy_mps"]),
                        "concentration": float(conc0),
                        "source_x_m": float(source[0]),
                        "source_y_m": float(source[1]),
                        "distance_to_source_m": float(d0),
                    }
                )
                traj1.append(
                    {
                        "step": float(step),
                        "x_m": float(post1["x_m"]),
                        "y_m": float(post1["y_m"]),
                        "z_m": float(post1["z_m"]),
                        "current_u_mps": float(post1["current_u_mps"]),
                        "current_v_mps": float(post1["current_v_mps"]),
                        "cmd_vx_mps": float(act1["vx_mps"]),
                        "cmd_vy_mps": float(act1["vy_mps"]),
                        "concentration": float(conc1),
                        "source_x_m": float(source[0]),
                        "source_y_m": float(source[1]),
                        "distance_to_source_m": float(d1),
                    }
                )

                if min_d <= plume_cfg.source_tolerance_m:
                    success = True
                    break

            wall = float(perf_counter() - t0)
            steps = float(max(len(traj0), len(traj1)))
            time_sec = steps * float(sim_cfg.dt_sec)
            final_d0 = float(traj0[-1]["distance_to_source_m"]) if traj0 else float("nan")
            final_d1 = float(traj1[-1]["distance_to_source_m"]) if traj1 else float("nan")
            ep_metrics = {
                "task": 1.0,
                "task_name": config.task,
                "episode": float(episode),
                "success": float(success),
                "timeout": float(not success and steps >= plume_cfg.max_steps),
                "steps": float(steps),
                "time_sec": float(time_sec),
                "energy_proxy": float(energy),
                "final_distance_to_source_min_m": float(min(final_d0, final_d1)),
                "first_detection_time_sec": float(first_detection_step * sim_cfg.dt_sec) if first_detection_step is not None else float("nan"),
                "controller_compensates_current": float(compensated),
                "wall_clock_sec": float(wall),
                "sim_steps_per_sec": float(steps / max(1e-9, wall)),
                "variant": config.variant,
                "time_index": float(time_index),
                "depth_index": float(depth_index),
            }
            per_episode_metrics.append(ep_metrics)
            _write_csv(trajectories_dir / f"episode_{episode:03d}_agent0.csv", traj0)
            _write_csv(trajectories_dir / f"episode_{episode:03d}_agent1.csv", traj1)

            if renderer is not None:
                media = _record_episode_video(
                    env,
                    out_dir=out_dir,
                    episode=episode,
                    fps=int(config.fps),
                    width=int(config.render_width),
                    height=int(config.render_height),
                    camera=str(config.camera),
                    frames=frames,
                )
                media_entries.append(
                    {
                        "task": config.task,
                        "episode": int(episode),
                        "controller": ctrl_mode,
                        "files": media,
                        "notes": plume_field_note,
                        "generation_command": command,
                    }
                )
                renderer.close()

    summary = _summarize(per_episode_metrics)
    summary["task"] = config.task
    summary["controller_mode"] = ctrl_mode

    metrics_csv_path = out_dir / "metrics.csv"
    _write_csv(metrics_csv_path, [dict(r) for r in per_episode_metrics])

    run_config_path = out_dir / "run_config.json"
    with run_config_path.open("w", encoding="utf-8") as file:
        task_notes = None
        if config.task == "plume_source_localization_3d":
            task_notes = "Includes a synthetic plume field (see media_manifest notes)."
        json.dump(
            {
                "run_config": asdict(config),
                "dataset_context": {
                    "dataset_path": str(context.dataset_path),
                    "variant": context.variant,
                    "time_index": context.time_index,
                    "depth_index": context.depth_index,
                },
                "resolved_controller_mode": ctrl_mode,
                "task_notes": task_notes,
                "generation_command": command,
            },
            file,
            indent=2,
        )

    metrics_json_path = out_dir / "metrics.json"
    with metrics_json_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "summary": summary,
                "episodes": per_episode_metrics,
                "run_config_path": str(run_config_path),
            },
            file,
            indent=2,
        )

    media_manifest_path = out_dir / "media_manifest.json"
    with media_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "task": config.task,
                "controller_mode": ctrl_mode,
                "entries": media_entries,
                "dataset_context": {
                    "dataset_path": str(context.dataset_path),
                    "variant": context.variant,
                    "time_index": context.time_index,
                    "depth_index": context.depth_index,
                },
                "generation_command": command,
            },
            file,
            indent=2,
        )

    ds.close()
    return {
        "output_dir": str(out_dir),
        "metrics_csv": str(metrics_csv_path),
        "metrics_json": str(metrics_json_path),
        "run_config": str(run_config_path),
        "media_manifest": str(media_manifest_path),
    }
