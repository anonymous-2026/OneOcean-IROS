from __future__ import annotations

from math import sqrt
from statistics import mean
from typing import Iterable

import numpy as np


def _point_line_distance(px: float, py: float, x1: float, y1: float, x2: float, y2: float) -> float:
    dx = x2 - x1
    dy = y2 - y1
    denom = dx * dx + dy * dy
    if denom <= 1e-12:
        return sqrt((px - x1) ** 2 + (py - y1) ** 2)
    t = ((px - x1) * dx + (py - y1) * dy) / denom
    nearest_x = x1 + t * dx
    nearest_y = y1 + t * dy
    return sqrt((px - nearest_x) ** 2 + (py - nearest_y) ** 2)


def compute_episode_metrics(
    trajectory: list[dict[str, float]],
    success: bool,
    timeout: bool,
    invalid_terminated: bool,
    dt_sec: float,
) -> dict[str, float]:
    if len(trajectory) < 2:
        raise ValueError("Trajectory must contain at least two points")

    xs = np.asarray([row["x_m"] for row in trajectory], dtype=float)
    ys = np.asarray([row["y_m"] for row in trajectory], dtype=float)
    distances = np.asarray([row["distance_to_goal_m"] for row in trajectory], dtype=float)
    cmd_speeds = np.asarray([row["cmd_speed_mps"] for row in trajectory], dtype=float)

    deltas = np.sqrt(np.diff(xs) ** 2 + np.diff(ys) ** 2)
    path_length = float(np.sum(deltas))
    start_x, start_y = float(xs[0]), float(ys[0])
    end_x, end_y = float(xs[-1]), float(ys[-1])
    goal_x = float(trajectory[0]["goal_x_m"])
    goal_y = float(trajectory[0]["goal_y_m"])
    goal_distance = float(sqrt((goal_x - start_x) ** 2 + (goal_y - start_y) ** 2))
    displacement = float(sqrt((end_x - start_x) ** 2 + (end_y - start_y) ** 2))

    cross_track = [
        _point_line_distance(float(x), float(y), start_x, start_y, goal_x, goal_y)
        for x, y in zip(xs, ys, strict=True)
    ]
    invalid_steps = int(sum(int(row["invalid_region"] > 0.5) for row in trajectory))

    return {
        "success": float(success),
        "timeout": float(timeout),
        "invalid_terminated": float(invalid_terminated),
        "steps": float(len(trajectory) - 1),
        "time_sec": float((len(trajectory) - 1) * dt_sec),
        "final_distance_to_goal_m": float(distances[-1]),
        "path_length_m": path_length,
        "goal_distance_m": goal_distance,
        "displacement_m": displacement,
        "path_efficiency": float(displacement / max(path_length, 1e-9)),
        "mean_cross_track_error_m": float(mean(cross_track)),
        "mean_commanded_speed_mps": float(np.mean(cmd_speeds)),
        "energy_proxy": float(np.sum((cmd_speeds**2) * dt_sec)),
        "invalid_steps": float(invalid_steps),
    }


def aggregate_metrics(metrics_rows: Iterable[dict[str, float]]) -> dict[str, float]:
    rows = list(metrics_rows)
    if not rows:
        return {}
    keys = [key for key in rows[0].keys()]
    summary: dict[str, float] = {}
    for key in keys:
        values: list[float] = []
        convertible = True
        for row in rows:
            try:
                values.append(float(row[key]))
            except (TypeError, ValueError):
                convertible = False
                break
        if not convertible:
            continue
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    summary["episodes"] = float(len(rows))
    summary["success_rate"] = float(np.mean([row["success"] for row in rows]))
    return summary
