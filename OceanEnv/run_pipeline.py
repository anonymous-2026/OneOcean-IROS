#!/usr/bin/env python3
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from Combine import interpolate_and_merge_fast
from GOPAF_Data import fetch_and_merge_copernicus_data
from GeoTIFF_Data import Get_GeoTIFF_Data


def _parse_depth_list(s: str) -> list[float]:
    parts = [p.strip() for p in s.split(",") if p.strip()]
    return [float(p) for p in parts]


def main() -> int:
    p = argparse.ArgumentParser(description="End-to-end data pipeline (terrain -> water -> combined netCDF).")

    p.add_argument("--lat-min", type=float, default=32.0)
    p.add_argument("--lat-max", type=float, default=33.0)
    p.add_argument("--lon-min", type=float, default=-66.5)
    p.add_argument("--lon-max", type=float, default=-65.5)
    p.add_argument("--elev-min", type=float, default=-10000.0)
    p.add_argument("--elev-max", type=float, default=0.0)

    p.add_argument("--start", default="2024-06-01T00:00:00", help="ISO datetime, e.g. 2024-06-01T00:00:00")
    p.add_argument("--end", default="2024-06-30T00:00:00", help="ISO datetime, e.g. 2024-06-30T00:00:00")

    p.add_argument("--min-depth", type=float, default=0.0)
    p.add_argument("--max-depth", type=float, default=200.0)
    p.add_argument(
        "--depths",
        default="",
        help="Optional comma-separated depths (meters). If set, fetches per-depth and stacks.",
    )
    p.add_argument(
        "--basic-dataset-id",
        default="cmems_mod_glo_phy_my_0.083deg_P1D-m",
        help="CMEMS dataset_id used for basic physics + uo/vo (must contain so/thetao/uo/vo/zos).",
    )
    p.add_argument(
        "--include-tides",
        action="store_true",
        help="Also fetch utide/vtide/utotal/vtotal (typically surface-only; will be broadcast to all depths).",
    )
    p.add_argument(
        "--tide-time-align",
        choices=["nearest", "linear"],
        default="nearest",
        help="Align hourly tide variables to the basic dataset time grid (engineering compromise).",
    )
    p.add_argument(
        "--tide-depth-profile",
        choices=["broadcast", "exp_decay", "linear"],
        default="broadcast",
        help="Depth profile applied when tide data is surface-only and must be broadcast to multiple depths.",
    )
    p.add_argument(
        "--tide-z0-m",
        type=float,
        default=50.0,
        help="Scale height (m) for exp_decay tide depth profile.",
    )
    p.add_argument(
        "--tide-zmax-m",
        type=float,
        default=200.0,
        help="Cutoff depth (m) for linear tide depth profile (weights go to 0 at zmax).",
    )

    p.add_argument("--skip-water-fetch", action="store_true", help="Use existing water file; do not call CMEMS.")
    p.add_argument(
        "--water-file",
        default=str(Path(__file__).resolve().parent / "Data" / "GOPAF" / "combined_gopaf_data.nc"),
        help="Existing water netCDF path (used when --skip-water-fetch is set).",
    )
    p.add_argument(
        "--water-out",
        default="combined_gopaf_data.nc",
        help="Output filename inside Data/GOPAF/ when fetching from CMEMS.",
    )

    p.add_argument(
        "--terrain-file",
        default=str(Path(__file__).resolve().parent / "output" / "filtered_data.tif"),
        help="GeoTIFF path to write/use for terrain crop.",
    )
    p.add_argument(
        "--terrain-out-dir",
        default=str(Path(__file__).resolve().parent / "output"),
        help="Directory for terrain outputs (filtered_data.tif, plots).",
    )

    p.add_argument(
        "--combined-out-dir",
        default=str(Path(__file__).resolve().parent / "Data" / "Combined"),
        help="Directory to write combined_environment.nc",
    )
    p.add_argument("--overwrite", action="store_true", help="Backup existing combined_environment.nc then overwrite.")
    p.add_argument(
        "--allow-water-coord-fixups",
        action="store_true",
        help="Allow risky auto-fixups if water lat/lon look swapped (default: fail).",
    )
    p.add_argument(
        "--target-res-deg",
        type=float,
        default=None,
        help="Optional output grid resolution in degrees (resamples terrain before merging).",
    )
    p.add_argument(
        "--allow-extrapolation",
        action="store_true",
        help="Allow lat/lon extrapolation during interpolation to avoid edge NaNs for tiny bboxes.",
    )

    args = p.parse_args()

    terrain_out_dir = Path(args.terrain_out_dir)
    terrain_out_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: terrain crop (writes filtered_data.tif under terrain_out_dir)
    Get_GeoTIFF_Data(
        args.lat_min,
        args.lat_max,
        args.lon_min,
        args.lon_max,
        args.elev_min,
        args.elev_max,
        str(terrain_out_dir),
    )

    terrain_file = Path(args.terrain_file)
    if not terrain_file.exists():
        raise SystemExit(f"Terrain file not found: {terrain_file}")

    # Step 2: water fetch (optional)
    water_file: Path
    if args.skip_water_fetch:
        water_file = Path(args.water_file)
        if not water_file.exists():
            raise SystemExit(f"Water file not found: {water_file}")
    else:
        depths = _parse_depth_list(args.depths) if args.depths else None
        fetch_and_merge_copernicus_data(
            username=None,
            password=None,
            minimum_longitude=args.lon_min,
            maximum_longitude=args.lon_max,
            minimum_latitude=args.lat_min,
            maximum_latitude=args.lat_max,
            start_datetime=args.start,
            end_datetime=args.end,
            minimum_depth=args.min_depth,
            maximum_depth=args.max_depth,
            output_filename=args.water_out,
            depths=depths,
            basic_dataset_id=args.basic_dataset_id,
            include_tides=args.include_tides,
            tide_time_align=args.tide_time_align,
            tide_depth_profile=args.tide_depth_profile,
            tide_z0_m=args.tide_z0_m,
            tide_zmax_m=args.tide_zmax_m,
            overwrite=args.overwrite,
        )
        water_file = Path(__file__).resolve().parent / "Data" / "GOPAF" / args.water_out
        if not water_file.exists():
            raise SystemExit(f"Expected water output not found: {water_file}")

    # Step 3: merge
    combined_out_dir = Path(args.combined_out_dir)
    combined_out_dir.mkdir(parents=True, exist_ok=True)
    combined_path = combined_out_dir / "combined_environment.nc"

    if combined_path.exists() and args.overwrite:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        combined_path.replace(combined_out_dir / f"combined_environment.nc.bak.{ts}")

    interpolate_and_merge_fast(
        str(terrain_file),
        str(water_file),
        str(combined_out_dir),
        method="linear",
        chunks={"time": 24},
        dtype="float32",
        compression_level=4,
        strict_coordinate_check=not args.allow_water_coord_fixups,
        target_res_deg=args.target_res_deg,
        allow_extrapolation=args.allow_extrapolation,
    )

    print(f"Done. Combined dataset: {combined_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

