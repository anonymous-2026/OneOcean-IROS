from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import csv
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from .drift import CachedDriftField, DriftConfig, sample_drift_xz


@dataclass
class RunConfig:
    episodes: int = 1
    max_steps: int = 160
    seed: int = 42
    output_dir: Optional[str] = None
    preset: str = "default"
    screenshot_interval: int = 25
    topdown_interval: int = 1
    max_screenshots_per_episode: int = 200
    max_topdown_per_episode: int = 1000
    write_video: bool = True
    video_fps: float = 12.0
    stop_distance_m: float = 0.45
    turn_threshold_rad: float = 0.22
    drift_compensation_gain: float = 0.75
    drift: DriftConfig = field(default_factory=DriftConfig)
    drift_cache_path: Optional[str] = None
    drift_origin_lat: Optional[float] = None
    drift_origin_lon: Optional[float] = None
    obstacle_proxy_mode: str = "off"
    obstacle_land_mask_threshold: float = 0.5
    obstacle_elevation_threshold: Optional[float] = None


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"oneocean_habitat_s2_{stamp}"


def _resolve_habitat_setup_root() -> Path:
    root = Path("/data/private/user2/workspace/habitat-setup").resolve()
    override = Path(os.environ.get("ONEOCEAN_HABITAT_SETUP", str(root))).expanduser().resolve()
    if not override.exists():
        raise FileNotFoundError(f"Habitat setup root not found: {override}")
    return override


def _import_habitat_wrapper():
    habitat_setup_root = _resolve_habitat_setup_root()
    src_dir = (habitat_setup_root / "src").resolve()
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from habitat_multi_agent_wrapper import HabitatMultiAgentWrapper  # type: ignore

    return HabitatMultiAgentWrapper


def _normalize_rgb_frame(rgb: np.ndarray) -> np.ndarray:
    frame = np.asarray(rgb)
    if frame.ndim != 3:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    if frame.dtype != np.uint8:
        max_value = float(np.max(frame)) if frame.size > 0 else 1.0
        if max_value <= 1.0:
            frame = (frame * 255.0).astype(np.uint8)
        else:
            frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    return frame


def _choose_action(
    pointgoal: Optional[np.ndarray],
    drift_x: float,
    drift_z: float,
    turn_threshold_rad: float,
    drift_comp_gain: float,
) -> str:
    if pointgoal is None or len(pointgoal) < 2:
        if abs(drift_x) + abs(drift_z) > 0.35:
            return "turn_left" if drift_x > 0.0 else "turn_right"
        return "move_forward"

    distance = float(pointgoal[0])
    angle = float(pointgoal[1])
    if distance <= 0.45:
        return "stop"

    drift_yaw = float(np.arctan2(drift_x, max(1e-6, drift_z)))
    adjusted_angle = angle - drift_comp_gain * drift_yaw

    if adjusted_angle > turn_threshold_rad:
        return "turn_left"
    if adjusted_angle < -turn_threshold_rad:
        return "turn_right"
    return "move_forward"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def run_habitat_ocean_proxy(config: RunConfig) -> dict[str, str]:
    HabitatMultiAgentWrapper = _import_habitat_wrapper()
    output_dir = Path(config.output_dir) if config.output_dir else _default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    screenshots_dir = output_dir / "screenshots"
    videos_dir = output_dir / "videos"
    trajectories_dir = output_dir / "trajectories"
    topdown_dir = output_dir / "topdown"
    for directory in (screenshots_dir, videos_dir, trajectories_dir, topdown_dir):
        directory.mkdir(parents=True, exist_ok=True)

    env = HabitatMultiAgentWrapper(
        config_path=None,
        scene_id=None,
        max_episode_steps=max(config.max_steps + 5, 30),
        camera_view="third_person",
    )
    preset = str(config.preset).strip().lower()
    screenshot_interval = max(1, int(config.screenshot_interval))
    topdown_interval = max(1, int(config.topdown_interval))
    max_screenshots = max(1, int(config.max_screenshots_per_episode))
    max_topdown = max(1, int(config.max_topdown_per_episode))
    write_video = bool(config.write_video)
    video_fps = max(1.0, float(config.video_fps))

    if preset == "compact":
        screenshot_interval = max(screenshot_interval, 40)
        topdown_interval = max(topdown_interval, 10)
        max_screenshots = min(max_screenshots, 6)
        max_topdown = min(max_topdown, 16)

    drift_field: Optional[CachedDriftField] = None
    drift_origin_lat = config.drift_origin_lat
    drift_origin_lon = config.drift_origin_lon
    if config.drift_cache_path:
        drift_field = CachedDriftField(config.drift_cache_path)
        if drift_origin_lat is None or drift_origin_lon is None:
            drift_origin_lat, drift_origin_lon = drift_field.center_latlon()

    obstacle_proxy_mode = str(config.obstacle_proxy_mode).strip().lower()
    if obstacle_proxy_mode not in {"off", "terminate"}:
        raise ValueError(
            f"Unsupported obstacle_proxy_mode='{config.obstacle_proxy_mode}', choose from off|terminate"
        )
    obstacle_proxy_enabled = False
    obstacle_proxy_reason = "off"
    if obstacle_proxy_mode != "off":
        if drift_field is None or drift_origin_lat is None or drift_origin_lon is None:
            obstacle_proxy_reason = "disabled_missing_drift_cache_or_origin"
            obstacle_proxy_mode = "off"
        else:
            has_land_mask = drift_field.land_mask is not None
            has_elevation = drift_field.elevation is not None
            elevation_enabled = has_elevation and config.obstacle_elevation_threshold is not None
            if not has_land_mask and not elevation_enabled:
                obstacle_proxy_reason = "disabled_missing_obstacle_layers"
                obstacle_proxy_mode = "off"
            else:
                obstacle_proxy_enabled = True
                obstacle_proxy_reason = "enabled"

    episode_metrics: list[dict[str, Any]] = []
    first_video_path: Optional[Path] = None
    first_traj_path: Optional[Path] = None

    try:
        for episode in range(config.episodes):
            reset_data = env.reset()
            observations = reset_data["observations"]
            trajectory_rows: list[dict[str, Any]] = []
            steps = 0
            reward_sum = 0.0
            done = False
            obstacle_hits = 0
            obstacle_terminated = 0.0

            video_writer = None
            video_path = videos_dir / f"episode_{episode:03d}.mp4"
            screenshots_saved = 0
            topdown_saved = 0

            while steps < config.max_steps and not done:
                agent_state = env.get_agent_state()
                position = np.asarray(agent_state.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
                pointgoal = agent_state.get("pointgoal")
                if pointgoal is not None:
                    pointgoal = np.asarray(pointgoal).reshape(-1)

                if drift_field is not None and drift_origin_lat is not None and drift_origin_lon is not None:
                    drift_x, drift_z = drift_field.sample_xz(
                        x_m=float(position[0]),
                        z_m=float(position[2]),
                        origin_lat=float(drift_origin_lat),
                        origin_lon=float(drift_origin_lon),
                    )
                else:
                    drift_x, drift_z = sample_drift_xz(position, steps, config.drift)
                action_name = _choose_action(
                    pointgoal=pointgoal,
                    drift_x=drift_x,
                    drift_z=drift_z,
                    turn_threshold_rad=config.turn_threshold_rad,
                    drift_comp_gain=config.drift_compensation_gain,
                )

                observations, reward, done, info = env.step({"action": action_name})
                reward_sum += float(reward)
                metrics = info.get("metrics", {})
                agent_state = info.get("agent_state", {})
                position = np.asarray(agent_state.get("position", [0.0, 0.0, 0.0]), dtype=np.float64)
                distance_to_goal = _safe_float(agent_state.get("distance_to_goal"), default=np.inf)
                land_mask_value = None
                elevation_value = None
                obstacle_blocked = False
                if obstacle_proxy_enabled and drift_field is not None:
                    land_mask_value = drift_field.sample_land_mask_xz(
                        x_m=float(position[0]),
                        z_m=float(position[2]),
                        origin_lat=float(drift_origin_lat),
                        origin_lon=float(drift_origin_lon),
                    )
                    elevation_value = drift_field.sample_elevation_xz(
                        x_m=float(position[0]),
                        z_m=float(position[2]),
                        origin_lat=float(drift_origin_lat),
                        origin_lon=float(drift_origin_lon),
                    )
                    if (
                        land_mask_value is not None
                        and land_mask_value >= float(config.obstacle_land_mask_threshold)
                    ):
                        obstacle_blocked = True
                    if (
                        config.obstacle_elevation_threshold is not None
                        and elevation_value is not None
                        and elevation_value >= float(config.obstacle_elevation_threshold)
                    ):
                        obstacle_blocked = True
                    if obstacle_blocked:
                        obstacle_hits += 1
                        if obstacle_proxy_mode == "terminate":
                            done = True
                            obstacle_terminated = 1.0

                rgb = env.get_rgb_observation(observations)
                if rgb is not None:
                    frame = _normalize_rgb_frame(rgb)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    if write_video and video_writer is None:
                        height, width = frame_bgr.shape[:2]
                        video_writer = cv2.VideoWriter(
                            str(video_path),
                            cv2.VideoWriter_fourcc(*"mp4v"),
                            video_fps,
                            (width, height),
                        )
                    if write_video and video_writer is not None:
                        video_writer.write(frame_bgr)
                    if (
                        screenshots_saved < max_screenshots
                        and (steps == 0 or (steps % screenshot_interval == 0))
                    ):
                        screenshot_path = screenshots_dir / f"episode_{episode:03d}_step_{steps:04d}.png"
                        cv2.imwrite(str(screenshot_path), frame_bgr)
                        screenshots_saved += 1

                if topdown_saved < max_topdown and (steps == 0 or (steps % topdown_interval == 0)):
                    topdown = env.get_top_down_map(output_size=(256, 256))
                    topdown_path = topdown_dir / f"episode_{episode:03d}_step_{steps:04d}.png"
                    cv2.imwrite(str(topdown_path), cv2.cvtColor(topdown, cv2.COLOR_RGB2BGR))
                    topdown_saved += 1

                trajectory_rows.append(
                    {
                        "episode": episode,
                        "step": steps,
                        "action": action_name,
                        "x": float(position[0]),
                        "y": float(position[1]),
                        "z": float(position[2]),
                        "distance_to_goal": distance_to_goal,
                        "drift_x_mps": drift_x,
                        "drift_z_mps": drift_z,
                        "reward": float(reward),
                        "success": _safe_float(metrics.get("success"), default=0.0),
                        "spl": _safe_float(metrics.get("spl"), default=0.0),
                        "land_mask_value": float(land_mask_value) if land_mask_value is not None else -1.0,
                        "elevation_value": float(elevation_value) if elevation_value is not None else -1.0,
                        "obstacle_blocked": float(1.0 if obstacle_blocked else 0.0),
                    }
                )
                steps += 1

                if action_name == "stop":
                    break

            if video_writer is not None:
                video_writer.release()

            metrics_final = env.env.get_metrics() if hasattr(env, "env") else {}
            success = _safe_float(metrics_final.get("success"), default=0.0) > 0.5
            final_distance = (
                float(trajectory_rows[-1]["distance_to_goal"]) if trajectory_rows else np.inf
            )
            traj_path = trajectories_dir / f"episode_{episode:03d}.csv"
            _write_csv(traj_path, trajectory_rows)

            if episode == 0:
                if write_video:
                    first_video_path = video_path
                first_traj_path = traj_path

            episode_metrics.append(
                {
                    "episode": episode,
                    "steps": steps,
                    "success": float(success),
                    "reward_sum": reward_sum,
                    "final_distance_to_goal": final_distance,
                    "spl": _safe_float(metrics_final.get("spl"), default=0.0),
                    "screenshots_saved": screenshots_saved,
                    "topdown_saved": topdown_saved,
                    "video_written": float(1.0 if write_video else 0.0),
                    "obstacle_hits": obstacle_hits,
                    "obstacle_terminated": obstacle_terminated,
                }
            )

    finally:
        env.close()

    success_rate = float(np.mean([row["success"] for row in episode_metrics])) if episode_metrics else 0.0
    avg_steps = float(np.mean([row["steps"] for row in episode_metrics])) if episode_metrics else 0.0
    obstacle_terminated_rate = (
        float(np.mean([row["obstacle_terminated"] for row in episode_metrics])) if episode_metrics else 0.0
    )

    run_config_path = output_dir / "run_config.json"
    with run_config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "config": {
                    **asdict(config),
                    "drift": config.drift.to_dict(),
                },
                "effective_capture": {
                    "preset": preset,
                    "screenshot_interval": screenshot_interval,
                    "topdown_interval": topdown_interval,
                    "max_screenshots_per_episode": max_screenshots,
                    "max_topdown_per_episode": max_topdown,
                    "write_video": write_video,
                    "video_fps": video_fps,
                },
                "runtime": {
                    "habitat_setup_root": str(_resolve_habitat_setup_root()),
                    "drift_source": (
                        f"cache:{drift_field.path}" if drift_field is not None else "synthetic"
                    ),
                    "drift_origin_lat": drift_origin_lat,
                    "drift_origin_lon": drift_origin_lon,
                    "obstacle_proxy": {
                        "mode": obstacle_proxy_mode,
                        "enabled": obstacle_proxy_enabled,
                        "reason": obstacle_proxy_reason,
                        "land_mask_available": bool(
                            drift_field is not None and drift_field.land_mask is not None
                        ),
                        "elevation_available": bool(
                            drift_field is not None and drift_field.elevation is not None
                        ),
                        "land_mask_threshold": float(config.obstacle_land_mask_threshold),
                        "elevation_threshold": config.obstacle_elevation_threshold,
                    },
                },
            },
            file,
            indent=2,
        )

    metrics_path = output_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "summary": {
                    "episodes": len(episode_metrics),
                    "success_rate": success_rate,
                    "avg_steps": avg_steps,
                    "obstacle_terminated_rate": obstacle_terminated_rate,
                },
                "episodes": episode_metrics,
            },
            file,
            indent=2,
        )

    return {
        "output_dir": str(output_dir),
        "run_config": str(run_config_path),
        "metrics_json": str(metrics_path),
        "first_video": str(first_video_path) if first_video_path else "",
        "first_trajectory": str(first_traj_path) if first_traj_path else "",
    }
