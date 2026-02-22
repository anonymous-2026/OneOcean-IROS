from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Optional

import numpy as np

from .controllers import GoalSeekingController, StationKeepingController
from .data import (
    CurrentSampler,
    DatasetContext,
    open_dataset,
    resolve_dataset_path,
    validate_time_depth_indices,
)
from .environment import NavigationConfig, OceanNavigationEnv
from .metrics import aggregate_metrics, compute_episode_metrics
from .visualization import render_trajectory_with_currents


SUPPORTED_TASKS = ("navigation", "station_keeping")
SUPPORTED_CONTROLLERS = (
    "auto",
    "goal_seek",
    "goal_seek_naive",
    "station_keep",
    "station_keep_naive",
)


@dataclass
class RunConfig:
    task: str = "navigation"
    controller: str = "auto"
    variant: str = "scene"
    dataset_path: Optional[str] = None
    episodes: int = 5
    seed: int = 42
    time_index: int = 0
    depth_index: int = 0
    dt_sec: float = 0.5
    max_steps: int = 600
    max_speed_mps: float = 1.8
    goal_distance_m: float = 250.0
    goal_tolerance_m: float = 25.0
    station_success_radius_m: float = 30.0
    station_mean_radius_m: float = 40.0
    include_tides: bool = True
    terminate_on_invalid_region: bool = True


def _default_output_dir(task: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"oneocean_{task}_{stamp}"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _resolve_controller(task: str, controller_name: str, max_speed_mps: float):
    if controller_name == "auto":
        controller_name = "goal_seek" if task == "navigation" else "station_keep"

    if task == "navigation" and controller_name in ("goal_seek", "goal_seek_naive"):
        return controller_name, GoalSeekingController(
            max_speed_mps=max_speed_mps,
            compensate_current=(controller_name == "goal_seek"),
        )
    if task == "station_keeping" and controller_name in ("station_keep", "station_keep_naive"):
        return controller_name, StationKeepingController(
            max_speed_mps=max_speed_mps,
            compensate_current=(controller_name == "station_keep"),
        )

    raise ValueError(
        f"Incompatible task/controller combination: task={task}, controller={controller_name}"
    )


def _run_navigation_episode(
    env: OceanNavigationEnv,
    controller: GoalSeekingController,
    config: RunConfig,
) -> tuple[list[dict[str, float]], bool, bool, bool]:
    env.reset(goal_distance_m=config.goal_distance_m)
    done = False
    success = False
    timeout = False
    invalid_terminated = False
    while not done:
        obs = env.observe()
        action = controller.act(obs)
        _obs, _reward, done, info = env.step(action, terminate_on_goal=True)
        success = info["success"] > 0.5
        timeout = info["timeout"] > 0.5
        invalid_terminated = info["invalid_region"] > 0.5 and done and not success and not timeout
    return env.trajectory, success, timeout, invalid_terminated


def _run_station_episode(
    env: OceanNavigationEnv,
    controller: StationKeepingController,
    config: RunConfig,
) -> tuple[list[dict[str, float]], bool, bool, bool, float]:
    env.reset(goal_distance_m=0.0)
    timeout = False
    invalid_terminated = False
    for _ in range(config.max_steps):
        obs = env.observe()
        action = controller.act(obs)
        _obs, _reward, _done, info = env.step(action, terminate_on_goal=False)
        if info["invalid_region"] > 0.5:
            invalid_terminated = True
            break

    trajectory = env.trajectory
    distances = np.asarray([row["distance_to_goal_m"] for row in trajectory], dtype=float)
    mean_radius = float(np.mean(distances))
    final_radius = float(distances[-1])
    success = (
        final_radius <= config.station_success_radius_m
        and mean_radius <= config.station_mean_radius_m
        and not invalid_terminated
    )
    return trajectory, success, timeout, invalid_terminated, mean_radius


def run_task(config: RunConfig, output_dir: Optional[str]) -> dict[str, str]:
    if config.task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported task: {config.task}")
    if config.controller not in SUPPORTED_CONTROLLERS:
        raise ValueError(f"Unsupported controller: {config.controller}")

    out_dir = Path(output_dir) if output_dir else _default_output_dir(config.task)
    out_dir.mkdir(parents=True, exist_ok=True)
    trajectories_dir = out_dir / "trajectories"
    trajectories_dir.mkdir(exist_ok=True)

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

    nav_cfg = NavigationConfig(
        dt_sec=config.dt_sec,
        max_steps=config.max_steps,
        max_speed_mps=config.max_speed_mps,
        goal_tolerance_m=config.goal_tolerance_m,
        terminate_on_invalid_region=config.terminate_on_invalid_region,
    )

    controller_name, controller = _resolve_controller(
        task=config.task, controller_name=config.controller, max_speed_mps=config.max_speed_mps
    )

    per_episode_metrics: list[dict[str, object]] = []
    first_episode_trajectory: list[dict[str, float]] = []

    for episode in range(config.episodes):
        env = OceanNavigationEnv(
            sampler=sampler,
            time_index=time_index,
            depth_index=depth_index,
            seed=config.seed + episode,
            config=nav_cfg,
        )
        episode_start = perf_counter()
        station_mean_radius = np.nan
        if config.task == "navigation":
            trajectory, success, timeout, invalid_terminated = _run_navigation_episode(
                env=env,
                controller=controller,
                config=config,
            )
        else:
            trajectory, success, timeout, invalid_terminated, station_mean_radius = _run_station_episode(
                env=env,
                controller=controller,
                config=config,
            )
        episode_wall_clock_sec = float(perf_counter() - episode_start)

        trajectory_path = trajectories_dir / f"episode_{episode:03d}.csv"
        _write_csv(trajectory_path, trajectory)
        if episode == 0:
            first_episode_trajectory = trajectory

        metrics = compute_episode_metrics(
            trajectory=trajectory,
            success=success,
            timeout=timeout,
            invalid_terminated=invalid_terminated,
            dt_sec=config.dt_sec,
        )
        metrics["station_mean_radius_m"] = (
            station_mean_radius if config.task == "station_keeping" else np.nan
        )
        metrics["episode_wall_clock_sec"] = episode_wall_clock_sec
        metrics["sim_steps_per_sec"] = float(
            max(0.0, (len(trajectory) - 1) / max(episode_wall_clock_sec, 1e-9))
        )
        metrics["controller_compensates_current"] = float(
            controller_name in ("goal_seek", "station_keep")
        )
        metrics["task"] = config.task
        metrics["controller"] = controller_name
        metrics["episode"] = float(episode)
        per_episode_metrics.append(metrics)

    summary = aggregate_metrics(per_episode_metrics)
    summary["task"] = config.task
    summary["controller"] = controller_name

    metrics_csv_path = out_dir / "metrics.csv"
    _write_csv(metrics_csv_path, per_episode_metrics)

    run_config_path = out_dir / "run_config.json"
    with run_config_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "run_config": asdict(config),
                "resolved_controller": controller_name,
                "dataset_context": {
                    "dataset_path": str(context.dataset_path),
                    "variant": context.variant,
                    "time_index": context.time_index,
                    "depth_index": context.depth_index,
                },
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

    plot_path = out_dir / "trajectory_overview.png"
    if first_episode_trajectory:
        render_trajectory_with_currents(
            ds=ds,
            trajectory=first_episode_trajectory,
            time_index=time_index,
            depth_index=depth_index,
            include_tides=config.include_tides,
            output_path=plot_path,
        )
    ds.close()

    return {
        "output_dir": str(out_dir),
        "metrics_csv": str(metrics_csv_path),
        "metrics_json": str(metrics_json_path),
        "run_config": str(run_config_path),
        "plot": str(plot_path),
    }
