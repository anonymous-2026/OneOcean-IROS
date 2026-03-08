#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from Combine import interpolate_and_merge_fast
from GOPAF_Data import fetch_and_merge_copernicus_data
from GeoTIFF_Data import Get_GeoTIFF_Data


@dataclass(frozen=True)
class Variant:
    name: str
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    start: str
    end: str
    min_depth: float
    max_depth: float
    basic_dataset_id: str
    include_tides: bool
    tide_time_align: str
    tide_depth_profile: str
    tide_z0_m: float
    tide_zmax_m: float
    target_res_deg: float | None


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")


def _write_metadata(out_dir: Path, v: Variant, *, terrain_file: Path, water_file: Path, combined_file: Path) -> None:
    out = {
        "generated_at_utc": _utcnow(),
        "variant": asdict(v),
        "paths": {
            "terrain_file": str(terrain_file),
            "water_file": str(water_file),
            "combined_file": str(combined_file),
        },
    }
    (out_dir / "variant.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def _run_one(v: Variant, *, base_dir: Path, out_root: Path, overwrite: bool) -> Path:
    out_dir = out_root / v.name
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Terrain crop
    terrain_out_dir = out_dir / "terrain"
    terrain_out_dir.mkdir(parents=True, exist_ok=True)
    terrain_file = terrain_out_dir / "filtered_data.tif"
    if not terrain_file.exists():
        Get_GeoTIFF_Data(v.lat_min, v.lat_max, v.lon_min, v.lon_max, -10000.0, 0.0, str(terrain_out_dir))
    if not terrain_file.exists():
        raise SystemExit(f"Terrain file not found after crop: {terrain_file}")

    # 2) Water fetch
    water_dir = out_dir / "water"
    water_dir.mkdir(parents=True, exist_ok=True)
    water_out_name = "combined_gopaf_data.nc"
    water_file = water_dir / water_out_name
    if not water_file.exists():
        fetch_and_merge_copernicus_data(
            username=None,
            password=None,
            minimum_longitude=v.lon_min,
            maximum_longitude=v.lon_max,
            minimum_latitude=v.lat_min,
            maximum_latitude=v.lat_max,
            start_datetime=v.start,
            end_datetime=v.end,
            minimum_depth=v.min_depth,
            maximum_depth=v.max_depth,
            output_filename=water_out_name,
            basic_dataset_id=v.basic_dataset_id,
            include_tides=v.include_tides,
            tide_time_align=v.tide_time_align,
            tide_depth_profile=v.tide_depth_profile,
            tide_z0_m=v.tide_z0_m,
            tide_zmax_m=v.tide_zmax_m,
            output_directory=water_dir,
            overwrite=overwrite,
        )
    if not water_file.exists():
        raise SystemExit(f"Water file not found after fetch: {water_file}")

    # 3) Combine
    combined_out_dir = out_dir / "combined"
    combined_out_dir.mkdir(parents=True, exist_ok=True)
    combined_file = combined_out_dir / "combined_environment.nc"
    if combined_file.exists() and overwrite:
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        combined_file.replace(combined_out_dir / f"combined_environment.nc.bak.{ts}")

    interpolate_and_merge_fast(
        str(terrain_file),
        str(water_file),
        str(combined_out_dir),
        method="linear",
        chunks={"time": 24},
        dtype="float32",
        compression_level=4,
        strict_coordinate_check=True,
        target_res_deg=v.target_res_deg,
        allow_extrapolation=(v.name == "tiny"),
    )

    _write_metadata(out_dir, v, terrain_file=terrain_file, water_file=water_file, combined_file=combined_file)
    return combined_file


def main() -> int:
    p = argparse.ArgumentParser(description="Generate multiple combined_environment.nc variants (tiny/scene/public).")
    p.add_argument(
        "--out-root",
        default=str(Path(__file__).resolve().parent / "Data" / "Combined" / "variants"),
        help="Output root directory; each variant is written under a subfolder.",
    )
    p.add_argument("--overwrite", action="store_true", help="Backup existing outputs then overwrite.")
    p.add_argument(
        "--which",
        default="tiny,scene,public",
        help="Comma-separated variant names to generate: tiny,scene,public",
    )
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing per-variant terrain/water files if present (skip recrop/refetch).",
    )

    args = p.parse_args()
    which = {x.strip() for x in args.which.split(",") if x.strip()}

    base_dir = Path(__file__).resolve().parent
    out_root = Path(args.out_root)

    variants: list[Variant] = [
        Variant(
            name="tiny",
            lat_min=32.40,
            lat_max=32.60,
            lon_min=-66.20,
            lon_max=-66.00,
            start="2024-06-01T00:00:00",
            end="2024-06-03T00:00:00",
            min_depth=0.0,
            max_depth=50.0,
            basic_dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            include_tides=True,
            tide_time_align="nearest",
            tide_depth_profile="exp_decay",
            tide_z0_m=30.0,
            tide_zmax_m=200.0,
            target_res_deg=None,
        ),
        Variant(
            name="scene",
            lat_min=32.0,
            lat_max=33.0,
            lon_min=-66.5,
            lon_max=-65.5,
            start="2024-06-01T00:00:00",
            end="2024-06-30T00:00:00",
            min_depth=0.0,
            max_depth=200.0,
            basic_dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            include_tides=True,
            tide_time_align="nearest",
            tide_depth_profile="exp_decay",
            tide_z0_m=50.0,
            tide_zmax_m=200.0,
            target_res_deg=None,
        ),
        Variant(
            name="public",
            lat_min=30.0,
            lat_max=40.0,
            lon_min=-72.0,
            lon_max=-62.0,
            start="2021-01-01T00:00:00",
            end="2024-12-01T00:00:00",
            min_depth=0.0,
            max_depth=200.0,
            basic_dataset_id="cmems_mod_glo_phy_my_0.083deg_P1M-m",
            include_tides=False,
            tide_time_align="nearest",
            tide_depth_profile="broadcast",
            tide_z0_m=50.0,
            tide_zmax_m=200.0,
            target_res_deg=0.25,
        ),
    ]

    selected = [v for v in variants if v.name in which]
    if not selected:
        raise SystemExit(f"No variants selected by --which={args.which!r}")

    for v in selected:
        if not args.reuse_existing:
            # Force regeneration by removing the per-variant cached files (keep variant directory itself).
            # Avoid shell rm: delete with Python to keep behavior consistent.
            v_dir = out_root / v.name
            for sub in ["terrain/filtered_data.tif", "water/combined_gopaf_data.nc"]:
                pth = v_dir / sub
                if pth.exists():
                    pth.unlink()

        out = _run_one(v, base_dir=base_dir, out_root=out_root, overwrite=args.overwrite)
        print(f"[{v.name}] OK: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
