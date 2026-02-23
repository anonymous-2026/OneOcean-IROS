from __future__ import annotations

import argparse
import json
import sys

from ..runner3d import Run3DConfig, run_task_3d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OneOcean S1 3D task runner (MuJoCo primary track; quality-gate compliant)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="nav_obstacles_3d",
        choices=["nav_obstacles_3d", "plume_source_localization_3d"],
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="auto",
        choices=["auto", "compensated", "naive"],
    )
    parser.add_argument("--variant", type=str, default="scene", choices=["tiny", "scene", "public"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--depth-index", type=int, default=0)
    parser.add_argument("--disable-tides", action="store_true")
    parser.add_argument("--dt-sec", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--target-domain-size-m", type=float, default=1000.0)
    parser.add_argument("--meters-per-sim-meter", type=float, default=None)
    parser.add_argument("--current-scale", type=float, default=80.0)
    parser.add_argument("--output-dir", type=str, default=None)

    parser.add_argument("--no-media", action="store_true")
    parser.add_argument("--record-all-episodes", action="store_true")
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=544)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera", type=str, default="orbit", help="cam_main|cam_low|orbit")

    parser.add_argument("--nav-goal-distance-m", type=float, default=60.0)
    parser.add_argument("--nav-goal-tolerance-m", type=float, default=12.0)
    parser.add_argument("--nav-obstacle-count", type=int, default=14)

    parser.add_argument("--plume-detection-threshold", type=float, default=0.16)
    parser.add_argument("--plume-source-tolerance-m", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Run3DConfig(
        task=args.task,
        controller=args.controller,
        variant=args.variant,
        dataset_path=args.dataset_path,
        episodes=args.episodes,
        seed=args.seed,
        time_index=args.time_index,
        depth_index=args.depth_index,
        include_tides=not args.disable_tides,
        dt_sec=args.dt_sec,
        max_steps=args.max_steps,
        target_domain_size_m=args.target_domain_size_m,
        meters_per_sim_meter=args.meters_per_sim_meter,
        current_scale=args.current_scale,
        record_media=not args.no_media,
        record_all_episodes=args.record_all_episodes,
        render_width=args.render_width,
        render_height=args.render_height,
        fps=args.fps,
        camera=args.camera,
        nav_goal_distance_m=args.nav_goal_distance_m,
        nav_goal_tolerance_m=args.nav_goal_tolerance_m,
        nav_obstacle_count=args.nav_obstacle_count,
        plume_detection_threshold=args.plume_detection_threshold,
        plume_source_tolerance_m=args.plume_source_tolerance_m,
    )

    cmd = f"{sys.executable} -m oneocean_sim.cli.run_3d_task " + " ".join(sys.argv[1:])
    outputs = run_task_3d(config=cfg, output_dir=args.output_dir, command=cmd)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
