#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import trimesh


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--radius-m", type=float, default=0.12)
    ap.add_argument("--length-m", type=float, default=0.65)
    ap.add_argument("--scale", type=float, default=1.0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # A simple capsule proxy (readable as a small UUV at a glance).
    mesh = trimesh.creation.capsule(radius=float(args.radius_m), height=float(args.length_m))
    mesh.apply_translation([0.0, 0.0, 0.0])

    obj_path = out_dir / "uuv_proxy.obj"
    mesh.export(obj_path)

    cfg = {
        "render_asset": "uuv_proxy.obj",
        "collision_asset": "uuv_proxy.obj",
        "up": [0.0, 1.0, 0.0],
        "front": [0.0, 0.0, -1.0],
        "scale": [float(args.scale), float(args.scale), float(args.scale)],
        "margin": 0.01,
        "friction_coefficient": 0.5,
        "restitution_coefficient": 0.1,
        "units_to_meters": 1.0,
        "force_flat_shading": False,
        "mass": 1.0,
        "COM": [0.0, 0.0, 0.0],
        "use_bounding_box_for_collision": True,
        "join_collision_meshes": True,
    }
    cfg_path = out_dir / "uuv_proxy.object_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")

    print(str(cfg_path))


if __name__ == "__main__":
    main()

