from __future__ import annotations

import argparse
import json

from ..runner import RunConfigS3, run_task_s3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OneOcean S3 tasks (SAPIEN backup track, 3D underwater quality gate)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="reef_navigation",
        choices=["reef_navigation", "formation_navigation"],
    )
    parser.add_argument("--variant", type=str, default="scene", choices=["tiny", "scene", "public"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--depth-index", type=int, default=0)
    parser.add_argument("--dt-sec", type=float, default=0.12)
    parser.add_argument("--max-steps", type=int, default=240)
    parser.add_argument("--max-rel-speed-mps", type=float, default=1.6)
    parser.add_argument("--velocity-tau-sec", type=float, default=0.6)
    parser.add_argument("--terrain-grid-size", type=int, default=33)
    parser.add_argument("--terrain-z-min-m", type=float, default=-30.0)
    parser.add_argument("--terrain-z-max-m", type=float, default=-5.0)
    parser.add_argument("--obstacle-count", type=int, default=14)
    parser.add_argument("--depth-clearance-m", type=float, default=4.0)

    parser.add_argument("--goal-distance-m", type=float, default=85.0)
    parser.add_argument("--goal-bearing-deg", type=float, default=None)
    parser.add_argument("--goal-tolerance-m", type=float, default=7.0)
    parser.add_argument("--formation-offset-y-m", type=float, default=10.0)
    parser.add_argument("--formation-tolerance-m", type=float, default=4.0)

    parser.add_argument("--disable-tides", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--render-fps", type=int, default=12)
    parser.add_argument("--render-frame-stride", type=int, default=2)
    parser.add_argument("--camera-mode", type=str, default="follow", choices=["follow", "orbit"])
    parser.add_argument("--render-episode-index", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfigS3(
        task=args.task,
        variant=args.variant,
        dataset_path=args.dataset_path,
        episodes=args.episodes,
        seed=args.seed,
        time_index=args.time_index,
        depth_index=args.depth_index,
        dt_sec=args.dt_sec,
        max_steps=args.max_steps,
        max_rel_speed_mps=args.max_rel_speed_mps,
        velocity_tau_sec=args.velocity_tau_sec,
        terrain_grid_size=args.terrain_grid_size,
        terrain_z_min_m=args.terrain_z_min_m,
        terrain_z_max_m=args.terrain_z_max_m,
        obstacle_count=args.obstacle_count,
        depth_clearance_m=args.depth_clearance_m,
        goal_distance_m=args.goal_distance_m,
        goal_bearing_deg=args.goal_bearing_deg,
        goal_tolerance_m=args.goal_tolerance_m,
        formation_offset_y_m=args.formation_offset_y_m,
        formation_tolerance_m=args.formation_tolerance_m,
        include_tides=not args.disable_tides,
        render=not args.no_render,
        render_width=args.render_width,
        render_height=args.render_height,
        render_fps=args.render_fps,
        render_frame_stride=args.render_frame_stride,
        camera_mode=args.camera_mode,
        render_episode_index=args.render_episode_index,
    )
    outputs = run_task_s3(config=config, output_dir=args.output_dir)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
