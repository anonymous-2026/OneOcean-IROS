from __future__ import annotations

import argparse
import json

from ..runner import RunConfig, run_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OneOcean S1 task runner (MuJoCo primary track)"
    )
    parser.add_argument(
        "--task",
        type=str,
        default="navigation",
        choices=["navigation", "station_keeping"],
    )
    parser.add_argument(
        "--controller",
        type=str,
        default="auto",
        choices=["auto", "goal_seek", "goal_seek_naive", "station_keep", "station_keep_naive"],
    )
    parser.add_argument("--variant", type=str, default="scene", choices=["tiny", "scene", "public"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--depth-index", type=int, default=0)
    parser.add_argument("--dt-sec", type=float, default=0.5)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-speed-mps", type=float, default=1.8)
    parser.add_argument("--goal-distance-m", type=float, default=250.0)
    parser.add_argument("--goal-tolerance-m", type=float, default=25.0)
    parser.add_argument("--station-success-radius-m", type=float, default=30.0)
    parser.add_argument("--station-mean-radius-m", type=float, default=40.0)
    parser.add_argument("--disable-tides", action="store_true")
    parser.add_argument("--allow-invalid-region", action="store_true")
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        task=args.task,
        controller=args.controller,
        variant=args.variant,
        dataset_path=args.dataset_path,
        episodes=args.episodes,
        seed=args.seed,
        time_index=args.time_index,
        depth_index=args.depth_index,
        dt_sec=args.dt_sec,
        max_steps=args.max_steps,
        max_speed_mps=args.max_speed_mps,
        goal_distance_m=args.goal_distance_m,
        goal_tolerance_m=args.goal_tolerance_m,
        station_success_radius_m=args.station_success_radius_m,
        station_mean_radius_m=args.station_mean_radius_m,
        include_tides=not args.disable_tides,
        terminate_on_invalid_region=not args.allow_invalid_region,
    )
    outputs = run_task(config=config, output_dir=args.output_dir)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
