from __future__ import annotations

import argparse

from ..settings import SettingsSpec, build_settings, write_settings_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write an AirSim settings.json for UNav-Sim ROV multi-vehicle runs.")
    parser.add_argument("--vehicle-count", type=int, default=8)
    parser.add_argument("--base-name", default="Rov")
    parser.add_argument("--spacing-m", type=float, default=2.0)
    parser.add_argument("--start-x", type=float, default=0.0)
    parser.add_argument("--start-y", type=float, default=0.0)
    parser.add_argument("--start-z", type=float, default=-2.0)
    parser.add_argument("--camera-name", default="front_right_custom")
    parser.add_argument("--camera-width", type=int, default=1280)
    parser.add_argument("--camera-height", type=int, default=720)
    parser.add_argument("--camera-fov-deg", type=float, default=90.0)
    parser.add_argument(
        "--physics-engine-name",
        default="ExternalPhysicsEngine",
        help="Recommended for dataset-driven drift injection. Use empty string to omit.",
    )
    parser.add_argument(
        "--out",
        default="~/Documents/AirSim/settings.json",
        help="AirSim reads from ~/Documents/AirSim/settings.json by default.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    spec = SettingsSpec(
        vehicle_count=args.vehicle_count,
        base_name=args.base_name,
        spacing_m=args.spacing_m,
        start_x=args.start_x,
        start_y=args.start_y,
        start_z=args.start_z,
        camera_name=args.camera_name,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        camera_fov_deg=args.camera_fov_deg,
        physics_engine_name=(args.physics_engine_name or None),
    )
    settings = build_settings(spec)
    out_path = write_settings_json(args.out, settings)
    print(str(out_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

