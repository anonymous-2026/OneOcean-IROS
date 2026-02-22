from __future__ import annotations

import argparse
import json

from ..drift import DriftConfig
from ..runner import RunConfig, run_habitat_ocean_proxy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run OneOcean S2 Habitat-Lab ocean proxy track"
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--preset", type=str, default="default", choices=["default", "compact"])
    parser.add_argument("--screenshot-interval", type=int, default=25)
    parser.add_argument("--topdown-interval", type=int, default=1)
    parser.add_argument("--max-screenshots-per-episode", type=int, default=200)
    parser.add_argument("--max-topdown-per-episode", type=int, default=1000)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--stop-distance-m", type=float, default=0.45)
    parser.add_argument("--turn-threshold-rad", type=float, default=0.22)
    parser.add_argument("--drift-compensation-gain", type=float, default=0.75)
    parser.add_argument("--drift-mode", type=str, default="synthetic_wave")
    parser.add_argument("--drift-amplitude-mps", type=float, default=0.35)
    parser.add_argument("--drift-spatial-scale-m", type=float, default=8.0)
    parser.add_argument("--drift-temporal-scale-steps", type=float, default=20.0)
    parser.add_argument("--drift-bias-x-mps", type=float, default=0.0)
    parser.add_argument("--drift-bias-z-mps", type=float, default=0.0)
    parser.add_argument("--drift-cache-path", type=str, default=None)
    parser.add_argument("--drift-origin-lat", type=float, default=None)
    parser.add_argument("--drift-origin-lon", type=float, default=None)
    parser.add_argument("--obstacle-proxy-mode", type=str, default="off", choices=["off", "terminate"])
    parser.add_argument("--obstacle-land-mask-threshold", type=float, default=0.5)
    parser.add_argument("--obstacle-elevation-threshold", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drift_cfg = DriftConfig(
        mode=args.drift_mode,
        amplitude_mps=args.drift_amplitude_mps,
        spatial_scale_m=args.drift_spatial_scale_m,
        temporal_scale_steps=args.drift_temporal_scale_steps,
        bias_x_mps=args.drift_bias_x_mps,
        bias_z_mps=args.drift_bias_z_mps,
    )
    run_cfg = RunConfig(
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        output_dir=args.output_dir,
        preset=args.preset,
        screenshot_interval=args.screenshot_interval,
        topdown_interval=args.topdown_interval,
        max_screenshots_per_episode=args.max_screenshots_per_episode,
        max_topdown_per_episode=args.max_topdown_per_episode,
        write_video=not args.no_video,
        video_fps=args.video_fps,
        stop_distance_m=args.stop_distance_m,
        turn_threshold_rad=args.turn_threshold_rad,
        drift_compensation_gain=args.drift_compensation_gain,
        drift=drift_cfg,
        drift_cache_path=args.drift_cache_path,
        drift_origin_lat=args.drift_origin_lat,
        drift_origin_lon=args.drift_origin_lon,
        obstacle_proxy_mode=args.obstacle_proxy_mode,
        obstacle_land_mask_threshold=args.obstacle_land_mask_threshold,
        obstacle_elevation_threshold=args.obstacle_elevation_threshold,
    )
    outputs = run_habitat_ocean_proxy(run_cfg)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
