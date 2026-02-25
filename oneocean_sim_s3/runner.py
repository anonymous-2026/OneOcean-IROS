from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import csv
import json
from math import atan2, cos, sin, sqrt
from pathlib import Path
from typing import Optional

import cv2
import imageio.v2 as imageio
import numpy as np

from oneocean_sim.data import (
    DatasetContext,
    CurrentSampler,
    open_dataset,
    resolve_dataset_path,
    validate_time_depth_indices,
)

from .env_3d import OceanWorldS3_3D, S3WorldConfig3D
from .software_renderer import CameraConfig, CameraPose


@dataclass(frozen=True)
class RunConfigS3:
    task: str = "reef_navigation"  # reef_navigation | formation_navigation
    variant: str = "scene"
    dataset_path: Optional[str] = None
    episodes: int = 2
    seed: int = 120
    time_index: int = 0
    depth_index: int = 0
    include_tides: bool = True

    # external scene (prefer third-party assets as the base)
    external_scene: Optional[str] = "polyhaven:dutch_ship_large_01"
    external_scene_resolution: str = "1k"
    external_scene_max_faces: int = 12000

    # world config
    dt_sec: float = 0.12
    max_steps: int = 360
    max_rel_speed_mps: float = 1.6
    velocity_tau_sec: float = 0.6
    terrain_grid_size: int = 33
    terrain_z_min_m: float = -30.0
    terrain_z_max_m: float = -5.0
    terrain_xy_scale: float = 0.02
    obstacle_count: int = 10
    depth_clearance_m: float = 4.0
    current_scale: float = 1.0

    # task config
    goal_distance_m: float = 40.0
    goal_bearing_deg: Optional[float] = None
    goal_tolerance_m: float = 8.0
    formation_offset_y_m: float = 10.0
    formation_tolerance_m: float = 6.0

    # rendering
    render: bool = True
    render_width: int = 640
    render_height: int = 480
    render_fps: int = 12
    render_frame_stride: int = 2
    camera_mode: str = "follow"  # follow | orbit
    render_episode_index: int = 0


def _default_output_dir(task: str) -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"s3_3d_{task}_{stamp}"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _norm2(x: float, y: float) -> float:
    return sqrt(x * x + y * y)


def _underwater_self_check(scene_png: Path) -> dict[str, object]:
    img = cv2.imread(str(scene_png), cv2.IMREAD_COLOR)
    if img is None:
        return {"passed": False, "reason": "cv2.imread failed", "scene_png": str(scene_png)}

    mean_bgr = img.reshape(-1, 3).mean(axis=0).astype(float)
    std_all = float(img.reshape(-1, 3).std())
    height = img.shape[0]
    top_mean = float(img[: max(1, height // 4)].mean())
    bottom_mean = float(img[max(0, 3 * height // 4) :].mean())

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bright_thr = 180
    bright_px = int((gray > bright_thr).sum())
    bright_ratio = float(bright_px) / float(gray.size)

    mean_b, mean_g, mean_r = float(mean_bgr[0]), float(mean_bgr[1]), float(mean_bgr[2])
    checks = {
        "blue_dominant": bool(mean_b > mean_r + 5.0),
        "non_flat": bool(std_all > 10.0),
        "attenuation_gradient": bool(top_mean > bottom_mean + 10.0),
        "particles_detected": bool(bright_ratio > 1e-4),
    }
    passed = bool(all(checks.values()))
    reason = (
        "blue-dominant palette + top-to-bottom attenuation + non-flat texture + particle specks detected"
        if passed
        else "failed one or more heuristics"
    )
    return {
        "passed": passed,
        "reason": reason,
        "checks": checks,
        "mean_bgr": [mean_b, mean_g, mean_r],
        "std_all": std_all,
        "top_mean": top_mean,
        "bottom_mean": bottom_mean,
        "bright_thr": int(bright_thr),
        "bright_ratio": bright_ratio,
        "scene_png": str(scene_png),
    }


def _compute_cmd_reef_navigation(
    *,
    x: float,
    y: float,
    z: float,
    goal_xy: tuple[float, float],
    obstacles: list[tuple[float, float, float]],
    max_rel_speed_mps: float,
    floor_z: float,
    depth_clearance_m: float,
) -> tuple[float, float, float]:
    dx = float(goal_xy[0] - x)
    dy = float(goal_xy[1] - y)
    dist = max(1e-6, _norm2(dx, dy))
    vx = max_rel_speed_mps * (dx / dist)
    vy = max_rel_speed_mps * (dy / dist)

    repulse = np.zeros(2, dtype=np.float64)
    influence = 9.5
    for ox, oy, radius in obstacles:
        ex = float(x - ox)
        ey = float(y - oy)
        d = max(1e-6, _norm2(ex, ey) - float(radius))
        if d > influence:
            continue
        mag = 0.8 * (1.0 / max(1e-6, d) - 1.0 / influence) / max(1e-6, d * d)
        repulse += mag * np.array([ex, ey], dtype=np.float64)
    vx += float(np.clip(repulse[0], -0.9, 0.9))
    vy += float(np.clip(repulse[1], -0.9, 0.9))

    speed = float(_norm2(vx, vy))
    if speed > max_rel_speed_mps:
        scale = max_rel_speed_mps / max(1e-9, speed)
        vx *= scale
        vy *= scale

    z_target = float(floor_z + depth_clearance_m)
    vz = float(np.clip(0.65 * (z_target - z), -0.8, 0.8))
    return float(vx), float(vy), float(vz)


def _compute_cmd_formation(
    *,
    self_xy: tuple[float, float],
    goal_xy: tuple[float, float],
    desired_offset_xy: tuple[float, float],
    other_xy: tuple[float, float],
    max_rel_speed_mps: float,
    floor_z: float,
    z: float,
    depth_clearance_m: float,
) -> tuple[float, float, float]:
    tx = float(other_xy[0] + desired_offset_xy[0])
    ty = float(other_xy[1] + desired_offset_xy[1])

    mix = 0.55
    gx = float((1.0 - mix) * tx + mix * goal_xy[0])
    gy = float((1.0 - mix) * ty + mix * goal_xy[1])

    dx = float(gx - self_xy[0])
    dy = float(gy - self_xy[1])
    dist = max(1e-6, _norm2(dx, dy))
    vx = max_rel_speed_mps * (dx / dist)
    vy = max_rel_speed_mps * (dy / dist)

    z_target = float(floor_z + depth_clearance_m)
    vz = float(np.clip(0.65 * (z_target - z), -0.8, 0.8))
    return float(vx), float(vy), float(vz)


def _camera_pose(
    *,
    mode: str,
    centroid: np.ndarray,
    heading_xy: np.ndarray,
    step: int,
) -> CameraPose:
    target = (float(centroid[0]), float(centroid[1]), float(centroid[2]))
    if mode == "orbit":
        angle = 0.015 * float(step)
        radius = 26.0
        eye = (
            float(centroid[0] + radius * cos(angle)),
            float(centroid[1] + radius * sin(angle)),
            float(centroid[2] + 18.0),
        )
        return CameraPose(eye_m=eye, target_m=target, up=(0.0, 0.0, 1.0))

    vx, vy = float(heading_xy[0]), float(heading_xy[1])
    yaw = atan2(vy, vx) if abs(vx) + abs(vy) > 1e-6 else 0.0
    back = np.array([-cos(yaw), -sin(yaw), 0.0], dtype=np.float64)
    eye_vec = 18.0 * back + np.array([0.0, 0.0, 12.5], dtype=np.float64)
    eye = (float(centroid[0] + eye_vec[0]), float(centroid[1] + eye_vec[1]), float(centroid[2] + eye_vec[2]))
    return CameraPose(eye_m=eye, target_m=target, up=(0.0, 0.0, 1.0))


def run_task_s3(config: RunConfigS3, output_dir: Optional[str]) -> dict[str, str]:
    if config.task not in {"reef_navigation", "formation_navigation"}:
        raise ValueError(f"Unsupported task: {config.task}")

    agents = 1 if config.task == "reef_navigation" else 2

    out_dir = Path(output_dir) if output_dir else _default_output_dir(config.task)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trajectories").mkdir(exist_ok=True)
    (out_dir / "media").mkdir(exist_ok=True)
    (out_dir / "assets").mkdir(exist_ok=True)

    dataset_path = resolve_dataset_path(config.dataset_path, config.variant)
    ds = open_dataset(dataset_path)
    try:
        time_index, depth_index = validate_time_depth_indices(ds, config.time_index, config.depth_index)
        context = DatasetContext(
            dataset_path=dataset_path,
            variant=config.variant,
            time_index=time_index,
            depth_index=depth_index,
        )
        sampler = CurrentSampler(ds, include_tides=config.include_tides)

        world_cfg = S3WorldConfig3D(
            dt_sec=float(config.dt_sec),
            max_steps=int(config.max_steps),
            max_rel_speed_mps=float(config.max_rel_speed_mps),
            velocity_tau_sec=float(config.velocity_tau_sec),
            terrain_grid_size=int(config.terrain_grid_size),
            terrain_z_min_m=float(config.terrain_z_min_m),
            terrain_z_max_m=float(config.terrain_z_max_m),
            terrain_xy_scale=float(config.terrain_xy_scale),
            obstacle_count=int(config.obstacle_count),
            depth_clearance_m=float(config.depth_clearance_m),
            current_scale=float(config.current_scale),
        )

        camera_cfg = CameraConfig(
            width=int(config.render_width),
            height=int(config.render_height),
            fovy_deg=60.0,
            near=0.2,
            far=5000.0,
        )

        world = OceanWorldS3_3D(
            sampler=sampler,
            dataset_path=dataset_path,
            time_index=time_index,
            depth_index=depth_index,
            include_tides=config.include_tides,
            seed=int(config.seed),
            agents=agents,
            config=world_cfg,
            output_dir=out_dir,
            external_scene=config.external_scene,
            external_scene_resolution=str(config.external_scene_resolution),
            external_scene_max_faces=int(config.external_scene_max_faces),
        )

        per_episode_metrics: list[dict[str, object]] = []
        media_records: list[dict[str, object]] = []

        for episode in range(int(config.episodes)):
            ep_seed = int(config.seed + episode)
            world.reset_task(
                task=config.task,
                seed=ep_seed,
                goal_distance_m=float(config.goal_distance_m),
                goal_bearing_deg=config.goal_bearing_deg,
                goal_tolerance_m=float(config.goal_tolerance_m),
                formation_offset_m=(0.0, float(config.formation_offset_y_m)),
                formation_tolerance_m=float(config.formation_tolerance_m),
            )

            obstacles_xy = [
                (float(o.center_m[0]), float(o.center_m[1]), float(o.radius_m)) for o in world.obstacles
            ]

            frames: list[np.ndarray] = []
            step_rows: list[dict[str, object]] = []

            success = False
            collision = False
            invalid_region = False
            energy = 0.0
            final_dist0 = float("nan")

            for step in range(int(world_cfg.max_steps)):
                obs = world.observe()
                agents_obs = obs["agents"]

                cmds: list[tuple[float, float, float]] = []
                if config.task == "reef_navigation":
                    a0 = agents_obs[0]
                    cmd = _compute_cmd_reef_navigation(
                        x=float(a0["x_m"]),
                        y=float(a0["y_m"]),
                        z=float(a0["z_m"]),
                        goal_xy=world.goal_xy_m,
                        obstacles=obstacles_xy,
                        max_rel_speed_mps=float(world_cfg.max_rel_speed_mps),
                        floor_z=float(a0["floor_z_m"]),
                        depth_clearance_m=float(world_cfg.depth_clearance_m),
                    )
                    cmds = [cmd]
                else:
                    a0 = agents_obs[0]
                    a1 = agents_obs[1]
                    leader_cmd = _compute_cmd_reef_navigation(
                        x=float(a0["x_m"]),
                        y=float(a0["y_m"]),
                        z=float(a0["z_m"]),
                        goal_xy=world.goal_xy_m,
                        obstacles=obstacles_xy,
                        max_rel_speed_mps=float(world_cfg.max_rel_speed_mps),
                        floor_z=float(a0["floor_z_m"]),
                        depth_clearance_m=float(world_cfg.depth_clearance_m),
                    )
                    follower_cmd = _compute_cmd_reef_navigation(
                        x=float(a1["x_m"]),
                        y=float(a1["y_m"]),
                        z=float(a1["z_m"]),
                        goal_xy=(
                            float((1.0 - 0.45) * (a0["x_m"]) + 0.45 * world.goal_xy_m[0]),
                            float((1.0 - 0.45) * (a0["y_m"] + config.formation_offset_y_m) + 0.45 * world.goal_xy_m[1]),
                        ),
                        obstacles=obstacles_xy,
                        max_rel_speed_mps=float(world_cfg.max_rel_speed_mps),
                        floor_z=float(a1["floor_z_m"]),
                        depth_clearance_m=float(world_cfg.depth_clearance_m),
                    )
                    cmds = [leader_cmd, follower_cmd]

                info = world.step(cmds)
                collision = bool(info["collided"])
                invalid_region = bool(info["invalid_region"])

                energy += float(sum(float(c[0] * c[0] + c[1] * c[1] + c[2] * c[2]) for c in cmds)) * float(
                    world_cfg.dt_sec
                )

                obs_next = world.observe()
                agents_next = obs_next["agents"]
                goal_x = float(obs_next["goal_x_m"])
                goal_y = float(obs_next["goal_y_m"])

                dist0 = _norm2(float(agents_next[0]["x_m"] - goal_x), float(agents_next[0]["y_m"] - goal_y))
                final_dist0 = float(dist0)
                if config.task == "reef_navigation":
                    success = (dist0 <= float(world.goal_tolerance_m)) and (not collision) and (not invalid_region)
                else:
                    desired_dx = 0.0
                    desired_dy = float(config.formation_offset_y_m)
                    form_dx = float(agents_next[1]["x_m"] - agents_next[0]["x_m"]) - desired_dx
                    form_dy = float(agents_next[1]["y_m"] - agents_next[0]["y_m"]) - desired_dy
                    formation_error = _norm2(form_dx, form_dy)
                    success = (
                        (dist0 <= float(world.goal_tolerance_m))
                        and (formation_error <= float(world.formation_tolerance_m))
                        and (not collision)
                        and (not invalid_region)
                    )

                for a in agents_next:
                    step_rows.append(
                        {
                            "episode": int(episode),
                            "step": int(step),
                            "time_sec": float((step + 1) * world_cfg.dt_sec),
                            "agent": int(a["agent"]),
                            "x_m": float(a["x_m"]),
                            "y_m": float(a["y_m"]),
                            "z_m": float(a["z_m"]),
                            "vx_mps": float(a["vx_mps"]),
                            "vy_mps": float(a["vy_mps"]),
                            "vz_mps": float(a["vz_mps"]),
                            "current_u_mps": float(a["current_u_mps"]),
                            "current_v_mps": float(a["current_v_mps"]),
                            "goal_x_m": float(goal_x),
                            "goal_y_m": float(goal_y),
                            "collided": int(collision),
                            "invalid_region": int(invalid_region),
                        }
                    )

                if config.render and episode == int(config.render_episode_index):
                    if (step % max(1, int(config.render_frame_stride))) == 0:
                        centroid = np.mean(
                            np.asarray([[float(a["x_m"]), float(a["y_m"]), float(a["z_m"])] for a in agents_next], dtype=np.float64),
                            axis=0,
                        )
                        heading = np.asarray([float(agents_next[0]["vx_mps"]), float(agents_next[0]["vy_mps"])], dtype=np.float64)
                        cam_pose = _camera_pose(
                            mode=str(config.camera_mode),
                            centroid=centroid,
                            heading_xy=heading,
                            step=step,
                        )
                        frame = world.render(camera=camera_cfg, pose=cam_pose)
                        frames.append(frame)

                if success or collision or invalid_region:
                    break

            traj_path = out_dir / "trajectories" / f"episode_{episode:03d}.csv"
            _write_csv(traj_path, step_rows)

            per_episode_metrics.append(
                {
                    "episode": int(episode),
                    "seed": int(ep_seed),
                    "success": int(success),
                    "collided": int(collision),
                    "invalid_region": int(invalid_region),
                    "steps": int(step + 1),
                    "time_sec": float((step + 1) * world_cfg.dt_sec),
                    "final_distance_to_goal_m": float(final_dist0),
                    "energy_proxy": float(energy),
                }
            )

            if config.render and episode == int(config.render_episode_index) and frames:
                media_dir = out_dir / "media"
                media_dir.mkdir(exist_ok=True)
                scene_png = media_dir / "scene.png"
                final_png = media_dir / "final.png"
                rollout_gif = media_dir / "rollout.gif"
                cv2.imwrite(str(scene_png), frames[0])
                cv2.imwrite(str(final_png), frames[-1])
                imageio.mimsave(str(rollout_gif), frames, fps=int(config.render_fps))
                self_check = _underwater_self_check(scene_png)
                media_records.append(
                    {
                        "episode": int(episode),
                        "scene_png": str(scene_png),
                        "final_png": str(final_png),
                        "rollout_gif": str(rollout_gif),
                        "width": int(camera_cfg.width),
                        "height": int(camera_cfg.height),
                        "fps": int(config.render_fps),
                        "frame_count": int(len(frames)),
                        "self_check": self_check,
                    }
                )

        # summary
        success_rate = float(np.mean([float(r["success"]) for r in per_episode_metrics])) if per_episode_metrics else 0.0
        summary = {
            "task": str(config.task),
            "episodes": int(config.episodes),
            "success_rate": float(success_rate),
            "energy_proxy_mean": float(np.mean([float(r["energy_proxy"]) for r in per_episode_metrics]))
            if per_episode_metrics
            else None,
        }

        metrics_csv_path = out_dir / "metrics.csv"
        _write_csv(metrics_csv_path, per_episode_metrics)

        run_config_path = out_dir / "run_config.json"
        with run_config_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "run_config": asdict(config),
                    "dataset_context": {
                        "dataset_path": str(context.dataset_path),
                        "variant": context.variant,
                        "time_index": context.time_index,
                        "depth_index": context.depth_index,
                    },
                    "world_metadata": world.dump_world_metadata(),
                    "backend": "s3_sapien_no_vulkan_renderer",
                    "notes": "S3 track uses SAPIEN physics but renders with a CPU software renderer due to missing Vulkan device on this machine.",
                },
                file,
                indent=2,
            )

        metrics_json_path = out_dir / "metrics.json"
        with metrics_json_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "summary": summary,
                    "episodes": per_episode_metrics,
                    "media": media_records,
                    "run_config_path": str(run_config_path),
                },
                file,
                indent=2,
            )

        if media_records:
            command_args = [
                f"--task {config.task}",
                f"--variant {config.variant}",
                f"--external-scene {config.external_scene or 'none'}",
                f"--external-scene-resolution {str(config.external_scene_resolution)}",
                f"--external-scene-max-faces {int(config.external_scene_max_faces)}",
                f"--episodes {config.episodes}",
                f"--seed {config.seed}",
            ]
            media_manifest = {
                "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                "task": str(config.task),
                "output_dir": str(out_dir),
                "media": media_records,
                "command": "python -m oneocean_sim_s3.cli.run_navigation_task_s3 "
                + " ".join(command_args),
            }
            media_manifest_path = out_dir / "media" / "media_manifest.json"
            with media_manifest_path.open("w", encoding="utf-8") as file:
                json.dump(media_manifest, file, indent=2)

    finally:
        ds.close()

    return {
        "output_dir": str(out_dir),
        "metrics_csv": str(metrics_csv_path),
        "metrics_json": str(metrics_json_path),
        "run_config": str(run_config_path),
        "backend": "s3_sapien_no_vulkan_renderer",
        "task": str(config.task),
    }
