from __future__ import annotations

import argparse
import json

from ..batch_regression import BatchConfig, default_case_library, run_batch_regression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run S2 multi-case regression and package per-case artifacts"
    )
    parser.add_argument(
        "--cases",
        type=str,
        default="synthetic_compact,cache_compact,cache_obstacle",
        help="Comma-separated case names from default case library",
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--drift-cache-path", type=str, default=None)
    parser.add_argument("--drift-origin-lat", type=float, default=None)
    parser.add_argument("--drift-origin-lon", type=float, default=None)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--bundle-screenshot-count", type=int, default=2)
    parser.add_argument("--bundle-topdown-count", type=int, default=2)
    parser.add_argument("--bundle-no-video", action="store_true")
    parser.add_argument("--build-best-media-package", action="store_true")
    parser.add_argument("--best-media-output-dir", type=str, default=None)
    parser.add_argument("--publish-best-e2", action="store_true")
    parser.add_argument("--e2-target-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    library = default_case_library()
    case_names = [item.strip() for item in args.cases.split(",") if item.strip()]
    if not case_names:
        raise ValueError("No valid case names provided in --cases")

    unknown = [name for name in case_names if name not in library]
    if unknown:
        known_names = ", ".join(sorted(library.keys()))
        raise ValueError(f"Unknown cases {unknown}. Available: {known_names}")

    config = BatchConfig(
        cases=[library[name] for name in case_names],
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        output_root=args.output_root,
        drift_cache_path=args.drift_cache_path,
        drift_origin_lat=args.drift_origin_lat,
        drift_origin_lon=args.drift_origin_lon,
        write_video=not args.no_video,
        video_fps=args.video_fps,
        bundle_screenshot_count=args.bundle_screenshot_count,
        bundle_topdown_count=args.bundle_topdown_count,
        bundle_include_video=not args.bundle_no_video,
        build_best_media_package=args.build_best_media_package,
        best_media_output_dir=args.best_media_output_dir,
        publish_best_e2=args.publish_best_e2,
        e2_target_dir=args.e2_target_dir,
    )
    outputs = run_batch_regression(config)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
