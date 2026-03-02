#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..drift_cache import load_drift_cache


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export a headless run into demo-compatible JSON (env + paths).")
    ap.add_argument("--run-dir", type=str, required=True, help="runs/headless/... episode dir (contains agents/* and run_meta.json).")
    ap.add_argument("--out-dir", type=str, required=True, help="Output folder to write drone_map_data.json + drone_path_data.json.")
    ap.add_argument("--stride", type=int, default=4, help="Downsample stride over recorded steps.")
    ap.add_argument("--terrain-step-m", type=float, default=20.0, help="XZ sampling step for terrainMap export.")
    return ap.parse_args()


def _read_pose_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        r = csv.reader(fp)
        header = next(r)
        if header[:4] != ["t", "x", "y", "z"]:
            raise ValueError(f"Unexpected pose header in {path}: {header[:8]}")
        for row in r:
            if not row:
                continue
            rows.append((float(row[0]), float(row[1]), float(row[2]), float(row[3])))
    return np.asarray(rows, dtype=np.float64)


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    root_manifest: dict[str, Any] = {}
    root_manifest_path = run_dir / "results_manifest.json"
    if root_manifest_path.exists():
        try:
            root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
        except Exception:
            root_manifest = {}
    env_cfg = meta.get("env_config", {})
    drift_npz = str(env_cfg.get("drift_cache_npz", ""))
    if not drift_npz:
        raise ValueError("run_meta.json is missing env_config.drift_cache_npz")

    lo = np.asarray(meta["bounds_xyz"]["lo"], dtype=np.float64)
    hi = np.asarray(meta["bounds_xyz"]["hi"], dtype=np.float64)
    origin_lat = float(meta.get("tile", {}).get("origin_lat", float("nan")))
    origin_lon = float(meta.get("tile", {}).get("origin_lon", float("nan")))
    n_agents = int(meta.get("n_agents", 1))

    # --- Export PATH data ---
    stride = int(max(1, args.stride))
    paths = []
    frames_exported: int | None = None
    for i in range(n_agents):
        p = run_dir / "agents" / f"agent_{i:03d}" / "pose_groundtruth" / "data.csv"
        pose = _read_pose_csv(p)[::stride]
        if frames_exported is None:
            frames_exported = int(pose.shape[0])
        # Demo schema uses {x,y,z}. We export y as (-depth) so deeper is more negative.
        one = [{"x": float(x), "y": float(-y), "z": float(z)} for (_t, x, y, z) in pose.tolist()]
        paths.append(one)

    task_id = str(meta.get("task", {}).get("kind", "unknown"))
    path_data = {
        "mapSeed": int(meta.get("seed", 0)),
        "waypoints": [],
        "userHoverMarkers": [],
        "experiments": [
            {
                "name": f"H1-{task_id}",
                "type": "PATH",
                "paths": paths,
            }
        ],
        "notes": {
            "source_run_dir": str(run_dir),
            "coord_convention": "x,z are meters in headless tile; exported y is -depth_m (so deeper => more negative).",
        },
    }
    (out_dir / "drone_path_data.json").write_text(json.dumps(path_data), encoding="utf-8")

    # --- Export ENV data (best-effort terrainMap) ---
    drift_field, drift_info = load_drift_cache(drift_npz)
    terrain = []
    step_m = float(max(5.0, args.terrain_step_m))
    xs = np.arange(float(lo[0]), float(hi[0]) + 1e-6, step_m, dtype=np.float64)
    zs = np.arange(float(lo[2]), float(hi[2]) + 1e-6, step_m, dtype=np.float64)
    for x in xs.tolist():
        for z in zs.tolist():
            elev = drift_field.sample_elevation_xz(x_m=float(x), z_m=float(z), origin_lat=origin_lat, origin_lon=origin_lon)
            if elev is None or not np.isfinite(float(elev)):
                continue
            # Demo uses y as height; we export seabed "height" as elevation (meters, typically negative).
            terrain.append({"x": float(round(x, 1)), "z": float(round(z, 1)), "y": float(round(float(elev), 2))})

    env_data: dict[str, Any] = {
        "seed": int(meta.get("seed", 0)),
        "cityBuildings": [],
        "mountainBuildings": [],
        "buildingColliders": [],
        "cabinPositions": [],
        "finalUsers": [],
        "terrainMap": terrain,
        "notes": {
            "source_run_dir": str(run_dir),
            "drift_npz": str(Path(drift_npz).expanduser().resolve()),
            "origin_lat": origin_lat,
            "origin_lon": origin_lon,
            "tile_bounds_xyz": {"lo": lo.tolist(), "hi": hi.tolist()},
            "drift_info": drift_info.to_dict(),
        },
    }
    (out_dir / "drone_map_data.json").write_text(json.dumps(env_data), encoding="utf-8")

    # Optional convenience bundle: single JSON that includes both env + paths + manifest.
    bundle = {
        "type": "oneocean_h1_replay_bundle",
        "export_manifest": {
            "run_dir": str(run_dir),
            "out_dir": str(out_dir),
            "written": ["drone_map_data.json", "drone_path_data.json", "replay_bundle.json"],
            "task": task_id,
            "n_agents": n_agents,
            "stride": stride,
            "frames_exported": frames_exported,
            "terrain_step_m": float(step_m),
            "terrain_points": int(len(terrain)),
            "dt_s": float(meta.get("env_config", {}).get("dt_s", float("nan"))),
            "git": root_manifest.get("git"),
            "source_results_manifest": str(root_manifest_path) if root_manifest_path.exists() else None,
        },
        "drone_map_data": env_data,
        "drone_path_data": path_data,
    }
    (out_dir / "replay_bundle.json").write_text(json.dumps(bundle), encoding="utf-8")

    (out_dir / "export_manifest.json").write_text(json.dumps(bundle["export_manifest"], indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(out_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
