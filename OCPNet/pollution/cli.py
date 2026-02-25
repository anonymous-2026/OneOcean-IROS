import argparse
import json
from pathlib import Path

import numpy as np

from .runner import run_synthetic_diffusion_case
from .viz import (
    analyze_nc_file,
    generate_synthetic_diffusion_series,
    plot_3d_currents,
    plot_pollutant_diffusion,
    simulate_diffusion_from_dataset,
)


def _default_combined_nc() -> str:
    return "/data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/combined_environment.nc"

def _default_public_nc() -> str:
    return "/data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/public/combined/combined_environment.nc"


def _cmd_run_synthetic(args: argparse.Namespace) -> int:
    output_dir = Path(args.output_dir).resolve()
    media_dir = output_dir / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    summary = run_synthetic_diffusion_case(
        output_dir=output_dir,
        nx=args.nx,
        ny=args.ny,
        nz=args.nz,
        steps=args.steps,
        time_step=args.time_step,
    )

    lons = np.arange(args.lon_min, args.lon_max + args.resolution, args.resolution)
    lats = np.arange(args.lat_min, args.lat_max + args.resolution, args.resolution)
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    all_days = generate_synthetic_diffusion_series(lon_grid, lat_grid, days=args.diffusion_days, seed=args.seed)
    sampled = [all_days[idx - 1] for idx in args.sample_days]
    media_outputs = plot_pollutant_diffusion(
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        pollutant_data=sampled,
        days=args.sample_days,
        pollutant_name="Synthetic Microplastic Diffusion",
        pollutant_data_all_days=all_days,
        output_dir=media_dir,
        prefix="microplastic_diffusion",
    )

    report = {
        "summary": summary,
        "media_outputs": media_outputs,
        "output_dir": str(output_dir),
    }
    report_file = output_dir / "run_report.json"
    report_file.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


def _cmd_render_currents(args: argparse.Namespace) -> int:
    outputs = plot_3d_currents(
        dataset_or_path=args.nc_path,
        output_dir=args.output_dir,
        skip=args.skip,
        time_index=args.time_index,
        depth_index=args.depth_index,
    )
    print(json.dumps(outputs, indent=2))
    return 0


def _cmd_analyze_nc(args: argparse.Namespace) -> int:
    stats, time_info = analyze_nc_file(args.nc_path)
    payload = {"time_info": time_info, "variables": stats}
    print(json.dumps(payload, indent=2, default=str))
    return 0


def _cmd_run_dataset_driven(args: argparse.Namespace) -> int:
    outputs = simulate_diffusion_from_dataset(
        nc_path=args.nc_path,
        output_dir=args.output_dir,
        depth_index=args.depth_index,
        time_start=args.time_start,
        time_count=args.time_count,
        spatial_stride=args.spatial_stride,
        diffusion_coeff=args.diffusion_coeff,
        frame_seconds=args.frame_seconds,
        substeps=args.substeps,
        prefix=args.prefix,
    )
    print(json.dumps(outputs, indent=2, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pollution simulation and visualization utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_synth = subparsers.add_parser("run-synthetic", help="Run a quick synthetic 3D diffusion simulation and media export.")
    run_synth.add_argument("--output-dir", default="OCPNet/output/pollution_refactor/synthetic_run")
    run_synth.add_argument("--nx", type=int, default=24)
    run_synth.add_argument("--ny", type=int, default=24)
    run_synth.add_argument("--nz", type=int, default=12)
    run_synth.add_argument("--steps", type=int, default=40)
    run_synth.add_argument("--time-step", type=float, default=20.0)
    run_synth.add_argument("--lon-min", type=float, default=130.0)
    run_synth.add_argument("--lon-max", type=float, default=145.0)
    run_synth.add_argument("--lat-min", type=float, default=30.0)
    run_synth.add_argument("--lat-max", type=float, default=45.0)
    run_synth.add_argument("--resolution", type=float, default=0.2)
    run_synth.add_argument("--diffusion-days", type=int, default=80)
    run_synth.add_argument("--sample-days", type=int, nargs="+", default=[5, 20, 35, 55, 75])
    run_synth.add_argument("--seed", type=int, default=7)
    run_synth.set_defaults(func=_cmd_run_synthetic)

    currents = subparsers.add_parser("render-currents", help="Render 3D current-over-bathymetry figures from combined_environment.nc.")
    currents.add_argument("--nc-path", default=_default_combined_nc())
    currents.add_argument("--output-dir", default="OCPNet/output/pollution_refactor/current_viz")
    currents.add_argument("--skip", type=int, default=15)
    currents.add_argument("--time-index", type=int, default=0)
    currents.add_argument("--depth-index", type=int, default=0)
    currents.set_defaults(func=_cmd_render_currents)

    analyze = subparsers.add_parser("analyze-nc", help="Print variable statistics for a NetCDF dataset.")
    analyze.add_argument("--nc-path", default=_default_combined_nc())
    analyze.set_defaults(func=_cmd_analyze_nc)

    dataset_run = subparsers.add_parser(
        "run-dataset-driven",
        help="Simulate and render dataset-driven diffusion proxy from combined current fields.",
    )
    dataset_run.add_argument("--nc-path", default=_default_public_nc())
    dataset_run.add_argument("--output-dir", default="OCPNet/output/pollution_refactor/dataset_diffusion")
    dataset_run.add_argument("--depth-index", type=int, default=0)
    dataset_run.add_argument("--time-start", type=int, default=0)
    dataset_run.add_argument("--time-count", type=int, default=24)
    dataset_run.add_argument("--spatial-stride", type=int, default=2)
    dataset_run.add_argument("--diffusion-coeff", type=float, default=18.0)
    dataset_run.add_argument("--frame-seconds", type=float, default=1800.0)
    dataset_run.add_argument("--substeps", type=int, default=3)
    dataset_run.add_argument("--prefix", default="dataset_diffusion")
    dataset_run.set_defaults(func=_cmd_run_dataset_driven)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
