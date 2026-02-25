from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from ..external_assets.polyhaven import download_gltf_model, write_polyhaven_manifest
from ..underwater.runner import CameraConfig
from ..underwater.showcase import ShowcaseConfig, render_showcase


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Render a small 'external scene' showcase using a Poly Haven CC0 model placed on our underwater stage."
    )
    parser.add_argument("--stage-obj", required=True)
    parser.add_argument("--stage-meta", default="", help="Optional stage meta JSON (not required for showcase render).")

    parser.add_argument(
        "--polyhaven-model-id",
        default="",
        help="If provided, downloads this Poly Haven model into runs/_cache/polyhaven and renders it.",
    )
    parser.add_argument(
        "--model-gltf",
        default="",
        help="Path to an existing GLTF/GLB model (skips download).",
    )
    parser.add_argument("--cache-dir", default="runs/_cache/polyhaven")
    parser.add_argument("--resolution", default="1k", choices=("1k", "2k", "4k", "8k"))
    parser.add_argument("--overwrite", action="store_true")

    parser.add_argument("--model-count", type=int, default=1)
    parser.add_argument("--model-scale", type=float, default=1.0)

    parser.add_argument("--output-dir", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--fps", type=float, default=24.0)
    parser.add_argument("--gif-stride", type=int, default=2)

    parser.add_argument("--cam-width", type=int, default=960)
    parser.add_argument("--cam-height", type=int, default=540)
    parser.add_argument("--cam-orbit-radius-m", type=float, default=22.0)
    parser.add_argument("--cam-orbit-height-m", type=float, default=2.4)
    parser.add_argument("--cam-orbit-period-steps", type=int, default=240)

    parser.add_argument("--fog-density", type=float, default=0.070)
    parser.add_argument("--no-silt", action="store_true")
    parser.add_argument("--silt-count", type=int, default=120)
    parser.add_argument("--silt-scale", type=float, default=0.028)

    args = parser.parse_args()

    model_gltf: str
    if args.model_gltf:
        model_gltf = str(Path(args.model_gltf).expanduser())
    elif args.polyhaven_model_id:
        out_dir = Path(args.cache_dir)
        gltf = download_gltf_model(
            asset_id=str(args.polyhaven_model_id),
            out_dir=out_dir,
            resolution=str(args.resolution),
            overwrite=bool(args.overwrite),
        )
        manifest = out_dir / str(args.polyhaven_model_id) / "polyhaven_manifest.json"
        write_polyhaven_manifest(
            asset_id=str(args.polyhaven_model_id),
            local_root=out_dir / str(args.polyhaven_model_id),
            primary_path=gltf,
            manifest_path=manifest,
        )
        model_gltf = str(gltf)
    else:
        raise SystemExit("Provide either --polyhaven-model-id or --model-gltf")

    cam = CameraConfig(
        width=int(args.cam_width),
        height=int(args.cam_height),
        orbit_radius_m=float(args.cam_orbit_radius_m),
        orbit_height_m=float(args.cam_orbit_height_m),
        orbit_period_steps=int(args.cam_orbit_period_steps),
    )
    cfg = ShowcaseConfig(
        stage_obj=str(Path(args.stage_obj).expanduser()),
        stage_meta=str(Path(args.stage_meta).expanduser()) if args.stage_meta else None,
        model_gltf=model_gltf,
        model_count=int(args.model_count),
        model_scale=float(args.model_scale),
        output_dir=str(Path(args.output_dir).expanduser()) if args.output_dir else None,
        invocation=" ".join([sys.executable] + sys.argv),
        seed=int(args.seed),
        steps=int(args.steps),
        fps=float(args.fps),
        gif_stride=int(args.gif_stride),
        camera=cam,
        fog_density=float(args.fog_density),
        enable_silt=not bool(args.no_silt),
        silt_count=int(args.silt_count),
        silt_scale=float(args.silt_scale),
    )
    outputs = render_showcase(cfg)
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

