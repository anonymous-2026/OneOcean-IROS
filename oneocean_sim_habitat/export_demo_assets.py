from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ExportConfig:
    run_dir: Path
    episode: int = 0
    output_dir: Path | None = None
    scale: float = 12.0
    terrain_grid_size: int = 56
    terrain_margin: float = 30.0
    seed: int = 20260223


def _load_trajectory(traj_path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with traj_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            rows.append({key: float(value) if key != "action" else value for key, value in row.items()})
    if not rows:
        raise ValueError(f"Trajectory is empty: {traj_path}")
    return rows


def _terrain_grid(rows: list[dict[str, float]], grid_size: int, margin: float) -> list[dict[str, float]]:
    xs = np.asarray([float(row["x"]) for row in rows], dtype=np.float64)
    zs = np.asarray([float(row["z"]) for row in rows], dtype=np.float64)
    x_min = float(np.min(xs) - margin)
    x_max = float(np.max(xs) + margin)
    z_min = float(np.min(zs) - margin)
    z_max = float(np.max(zs) + margin)
    x_values = np.linspace(x_min, x_max, grid_size)
    z_values = np.linspace(z_min, z_max, grid_size)

    terrain_points: list[dict[str, float]] = []
    for x in x_values:
        for z in z_values:
            y = (
                25.0
                + 6.0 * np.sin(x / 18.0)
                + 5.0 * np.cos(z / 21.0)
                + 2.0 * np.sin((x + z) / 27.0)
            )
            terrain_points.append({"x": float(x), "z": float(z), "y": float(y)})
    return terrain_points


def _path_points(rows: list[dict[str, float]], scale: float) -> list[dict[str, float]]:
    points: list[dict[str, float]] = []
    for row in rows:
        points.append(
            {
                "x": float(row["x"]) * scale,
                "y": max(0.1, float(row["y"]) * scale + 6.0),
                "z": float(row["z"]) * scale,
            }
        )
    return points


def export_demo_assets(config: ExportConfig) -> dict[str, str]:
    run_dir = config.run_dir.resolve()
    traj_path = run_dir / "trajectories" / f"episode_{config.episode:03d}.csv"
    if not traj_path.exists():
        raise FileNotFoundError(f"Trajectory file not found: {traj_path}")

    rows = _load_trajectory(traj_path)
    output_dir = (config.output_dir or (run_dir / "demo_export")).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    path_points = _path_points(rows, scale=config.scale)
    terrain_points = _terrain_grid(
        rows,
        grid_size=config.terrain_grid_size,
        margin=config.terrain_margin,
    )
    final_user = path_points[-1]
    waypoint_stride = max(1, len(path_points) // 12)
    waypoints = path_points[::waypoint_stride]
    if waypoints[-1] != final_user:
        waypoints.append(final_user)

    map_payload: dict[str, Any] = {
        "seed": int(config.seed),
        "cityBuildings": [],
        "mountainBuildings": [],
        "cabinPositions": [],
        "buildingColliders": [],
        "terrainMap": terrain_points,
        "finalUsers": [final_user],
    }

    path_payload: dict[str, Any] = {
        "mapSeed": int(config.seed),
        "waypoints": waypoints,
        "userHoverMarkers": [final_user],
        "experiments": [
            {
                "type": "s2_habitat",
                "name": "S2 Habitat Ocean Proxy",
                "paths": [path_points],
            }
        ],
    }

    map_path = output_dir / "drone_map_data.json"
    path_path = output_dir / "drone_path_data.json"
    with map_path.open("w", encoding="utf-8") as file:
        json.dump(map_payload, file, indent=2)
    with path_path.open("w", encoding="utf-8") as file:
        json.dump(path_payload, file, indent=2)

    manifest = {
        "source_run_dir": str(run_dir),
        "episode": int(config.episode),
        "trajectory_csv": str(traj_path),
        "demo_map_json": str(map_path),
        "demo_path_json": str(path_path),
        "compatibility": "demo_ref_v0_keys",
        "notes": [
            "Schema intentionally mirrors demo_ref keys for E2 compatibility.",
            "Semantics are ocean scenario (not UAV city).",
        ],
    }
    manifest_path = output_dir / "assets_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return {
        "output_dir": str(output_dir),
        "map_json": str(map_path),
        "path_json": str(path_path),
        "manifest_json": str(manifest_path),
    }
