from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..underwater.runner import CameraConfig, UnderwaterRunConfig, run_underwater_tasks


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Habitat-Sim underwater qualitative tasks (Agent H / S2).")
    parser.add_argument("--stage-obj", required=True, help="Path to underwater stage OBJ (from build_underwater_stage).")
    parser.add_argument("--stage-meta", required=True, help="Path to underwater stage meta JSON.")
    parser.add_argument("--drift-cache-path", default="", help="Optional npz drift cache (from prepare_drift_cache).")
    parser.add_argument("--output-dir", default="", help="Optional output directory under runs/.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dt-s", type=float, default=1.0)
    parser.add_argument("--max-steps-task1", type=int, default=240)
    parser.add_argument("--max-steps-task2", type=int, default=320)
    parser.add_argument("--video-fps", type=float, default=24.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--gif-stride", type=int, default=2)
    parser.add_argument("--current-gain", type=float, default=1.0, help="Multiply sampled currents for stronger drift.")

    parser.add_argument("--cam-width", type=int, default=960)
    parser.add_argument("--cam-height", type=int, default=540)
    parser.add_argument("--cam-orbit-radius-m", type=float, default=18.0)
    parser.add_argument("--cam-orbit-height-m", type=float, default=10.0)
    parser.add_argument("--cam-orbit-period-steps", type=int, default=240)

    args = parser.parse_args()

    cam = CameraConfig(
        width=int(args.cam_width),
        height=int(args.cam_height),
        orbit_radius_m=float(args.cam_orbit_radius_m),
        orbit_height_m=float(args.cam_orbit_height_m),
        orbit_period_steps=int(args.cam_orbit_period_steps),
    )
    cfg = UnderwaterRunConfig(
        stage_obj=str(Path(args.stage_obj).expanduser()),
        stage_meta=str(Path(args.stage_meta).expanduser()),
        drift_cache_path=str(Path(args.drift_cache_path).expanduser()) if args.drift_cache_path else None,
        output_dir=str(Path(args.output_dir).expanduser()) if args.output_dir else None,
        invocation=" ".join([sys.executable] + sys.argv),
        seed=int(args.seed),
        dt_s=float(args.dt_s),
        max_steps_task1=int(args.max_steps_task1),
        max_steps_task2=int(args.max_steps_task2),
        write_video=not bool(args.no_video),
        video_fps=float(args.video_fps),
        gif_stride=int(args.gif_stride),
        current_gain=float(args.current_gain),
        camera=cam,
    )
    outputs = run_underwater_tasks(cfg)
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
