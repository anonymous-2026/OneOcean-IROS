from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CompatConfig:
    run_dir: Path
    episode: int = 0
    dt_sec: float = 1.0
    output_dir: Path | None = None


def _read_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        return list(reader)


def _point_line_distance(px: float, pz: float, x1: float, z1: float, x2: float, z2: float) -> float:
    dx = x2 - x1
    dz = z2 - z1
    denom = dx * dx + dz * dz
    if denom <= 1e-12:
        return float(np.hypot(px - x1, pz - z1))
    t = ((px - x1) * dx + (pz - z1) * dz) / denom
    nx = x1 + t * dx
    nz = z1 + t * dz
    return float(np.hypot(px - nx, pz - nz))


def export_s1_compatible_metrics(config: CompatConfig) -> dict[str, str]:
    run_dir = config.run_dir.resolve()
    metrics_src = run_dir / "metrics.json"
    traj_src = run_dir / "trajectories" / f"episode_{config.episode:03d}.csv"
    if not metrics_src.exists():
        raise FileNotFoundError(f"metrics.json not found: {metrics_src}")
    if not traj_src.exists():
        raise FileNotFoundError(f"trajectory not found: {traj_src}")

    with metrics_src.open("r", encoding="utf-8") as file:
        metrics_data = json.load(file)
    episodes = metrics_data.get("episodes", [])
    if not episodes:
        raise ValueError(f"No episode entries in {metrics_src}")
    episode_row = next((row for row in episodes if int(row.get("episode", -1)) == config.episode), episodes[0])

    rows = _read_rows(traj_src)
    if len(rows) < 2:
        raise ValueError(f"Need at least two trajectory rows: {traj_src}")
    xs = np.asarray([float(row["x"]) for row in rows], dtype=np.float64)
    zs = np.asarray([float(row["z"]) for row in rows], dtype=np.float64)
    distances = np.asarray([float(row.get("distance_to_goal", "0")) for row in rows], dtype=np.float64)
    drift_x = np.asarray([float(row.get("drift_x_mps", "0")) for row in rows], dtype=np.float64)
    drift_z = np.asarray([float(row.get("drift_z_mps", "0")) for row in rows], dtype=np.float64)
    deltas = np.sqrt(np.diff(xs) ** 2 + np.diff(zs) ** 2)

    start_x, start_z = float(xs[0]), float(zs[0])
    end_x, end_z = float(xs[-1]), float(zs[-1])
    path_length = float(np.sum(deltas))
    displacement = float(np.hypot(end_x - start_x, end_z - start_z))
    goal_distance = float(distances[0]) if np.isfinite(distances[0]) else displacement
    cross_track = [
        _point_line_distance(float(x), float(z), start_x, start_z, end_x, end_z)
        for x, z in zip(xs, zs)
    ]

    success = float(episode_row.get("success", 0.0))
    steps = float(episode_row.get("steps", len(rows) - 1))
    timeout = float(1.0 if success < 0.5 and steps >= (len(rows) - 1) else 0.0)

    s1_like = {
        "success": success,
        "timeout": timeout,
        "invalid_terminated": 0.0,
        "steps": steps,
        "time_sec": float(steps * config.dt_sec),
        "final_distance_to_goal_m": float(distances[-1]),
        "path_length_m": path_length,
        "goal_distance_m": goal_distance,
        "displacement_m": displacement,
        "path_efficiency": float(displacement / max(path_length, 1e-9)),
        "mean_cross_track_error_m": float(np.mean(cross_track)),
        "mean_commanded_speed_mps": float(np.mean(np.hypot(drift_x, drift_z))),
        "energy_proxy": float(np.sum((np.hypot(drift_x, drift_z) ** 2) * config.dt_sec)),
        "invalid_steps": 0.0,
        "episode": float(config.episode),
    }

    output_dir = (config.output_dir or (run_dir / "compat")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    out_json = output_dir / "metrics_s1_compat.json"
    out_csv = output_dir / "metrics_s1_compat.csv"
    with out_json.open("w", encoding="utf-8") as file:
        json.dump({"episodes": [s1_like]}, file, indent=2)

    with out_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(s1_like.keys()))
        writer.writeheader()
        writer.writerow(s1_like)

    return {
        "output_dir": str(output_dir),
        "metrics_json": str(out_json),
        "metrics_csv": str(out_csv),
    }
