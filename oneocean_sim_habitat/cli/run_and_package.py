from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..build_media_package import build_media_package
from ..build_compact_bundle import build_compact_bundle
from ..drift import DriftConfig
from ..export_demo_assets import ExportConfig, export_demo_assets
from ..export_s1_compatible_metrics import CompatConfig, export_s1_compatible_metrics
from ..publish_e2_demo_assets import publish_e2_demo_assets
from ..runner import RunConfig, run_habitat_ocean_proxy


def _normalize_path(value: str) -> str:
    if not value:
        return ""
    return str(Path(value).expanduser().resolve())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run S2 Habitat track and package demo/metrics artifacts in one command"
    )
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--preset", type=str, default="compact", choices=["default", "compact"])
    parser.add_argument("--drift-cache-path", type=str, default=None)
    parser.add_argument("--drift-origin-lat", type=float, default=None)
    parser.add_argument("--drift-origin-lon", type=float, default=None)
    parser.add_argument("--drift-amplitude-mps", type=float, default=0.35)
    parser.add_argument("--drift-spatial-scale-m", type=float, default=8.0)
    parser.add_argument("--drift-temporal-scale-steps", type=float, default=20.0)
    parser.add_argument("--drift-bias-x-mps", type=float, default=0.0)
    parser.add_argument("--drift-bias-z-mps", type=float, default=0.0)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--video-fps", type=float, default=12.0)
    parser.add_argument("--obstacle-proxy-mode", type=str, default="off", choices=["off", "terminate"])
    parser.add_argument("--obstacle-land-mask-threshold", type=float, default=0.5)
    parser.add_argument("--obstacle-elevation-threshold", type=float, default=None)
    parser.add_argument("--episode-for-export", type=int, default=0)
    parser.add_argument("--no-demo-export", action="store_true")
    parser.add_argument("--no-compat-export", action="store_true")
    parser.add_argument("--no-bundle", action="store_true")
    parser.add_argument("--bundle-screenshot-count", type=int, default=4)
    parser.add_argument("--bundle-topdown-count", type=int, default=6)
    parser.add_argument("--bundle-no-video", action="store_true")
    parser.add_argument("--build-media-package", action="store_true")
    parser.add_argument("--media-output-dir", type=str, default=None)
    parser.add_argument("--media-scene-count", type=int, default=3)
    parser.add_argument("--media-progress-count", type=int, default=4)
    parser.add_argument("--media-topdown-count", type=int, default=3)
    parser.add_argument("--media-gif-fps", type=float, default=8.0)
    parser.add_argument("--media-gif-max-frames", type=int, default=80)
    parser.add_argument("--publish-e2", action="store_true")
    parser.add_argument("--e2-target-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    drift_cfg = DriftConfig(
        mode="synthetic_wave",
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
        drift=drift_cfg,
        drift_cache_path=args.drift_cache_path,
        drift_origin_lat=args.drift_origin_lat,
        drift_origin_lon=args.drift_origin_lon,
        write_video=not args.no_video,
        video_fps=args.video_fps,
        obstacle_proxy_mode=args.obstacle_proxy_mode,
        obstacle_land_mask_threshold=args.obstacle_land_mask_threshold,
        obstacle_elevation_threshold=args.obstacle_elevation_threshold,
    )
    run_outputs = run_habitat_ocean_proxy(run_cfg)
    run_outputs = {key: _normalize_path(value) for key, value in run_outputs.items()}
    run_dir = Path(run_outputs["output_dir"])

    package_outputs: dict[str, dict[str, str]] = {"run": run_outputs}
    if not args.no_demo_export:
        package_outputs["demo_export"] = export_demo_assets(
            ExportConfig(run_dir=run_dir, episode=args.episode_for_export)
        )
    if not args.no_compat_export:
        package_outputs["compat_export"] = export_s1_compatible_metrics(
            CompatConfig(run_dir=run_dir, episode=args.episode_for_export)
        )
    if not args.no_bundle:
        package_outputs["compact_bundle"] = build_compact_bundle(
            run_dir=run_dir,
            screenshot_count=args.bundle_screenshot_count,
            topdown_count=args.bundle_topdown_count,
            include_video=not args.bundle_no_video,
        )
    if args.build_media_package:
        package_outputs["media_package"] = build_media_package(
            run_dir=run_dir,
            output_dir=Path(args.media_output_dir) if args.media_output_dir else None,
            scene_count=args.media_scene_count,
            progress_count=args.media_progress_count,
            topdown_count=args.media_topdown_count,
            gif_fps=args.media_gif_fps,
            gif_max_frames=args.media_gif_max_frames,
        )
    if args.publish_e2:
        package_outputs["e2_publish"] = publish_e2_demo_assets(
            run_dir=run_dir,
            target_dir=args.e2_target_dir,
        )

    manifest_path = run_dir / "run_and_package_manifest.json"
    package_outputs["manifest"] = {"run_and_package_manifest": str(manifest_path)}
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(package_outputs, file, indent=2)
    print(json.dumps(package_outputs, indent=2))


if __name__ == "__main__":
    main()
