#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path

import xarray as xr

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


def _write_metadata(
    out_dir: Path,
    v: Variant,
    *,
    terrain_file: Path,
    water_file: Path,
    combined_file: Path,
    web_zarr: Path | None = None,
) -> None:
    # Record the *actual* combined dataset time range (CMEMS may clip requests to available times).
    actual_time = {}
    try:
        with xr.open_dataset(combined_file, engine="h5netcdf") as ds:
            if "time" in ds.coords and ds["time"].size:
                t0 = ds["time"].values[0]
                t1 = ds["time"].values[-1]
                actual_time = {
                    "start": str(t0),
                    "end": str(t1),
                    "n": int(ds["time"].size),
                }
    except Exception:
        # Metadata should never make generation fail; keep it best-effort.
        actual_time = {}

    out = {
        "generated_at_utc": _utcnow(),
        "variant": asdict(v),
        "actual": {"time": actual_time} if actual_time else {},
        "paths": {
            "terrain_file": str(terrain_file),
            "water_file": str(water_file),
            "combined_file": str(combined_file),
            **({"web_zarr": str(web_zarr)} if web_zarr is not None else {}),
        },
    }
    (out_dir / "variant.json").write_text(json.dumps(out, indent=2), encoding="utf-8")


def _export_web_zarr(
    combined_file: Path,
    out_zarr: Path,
    *,
    time_chunk: int = 60,
    keep_vars: list[str] | None = None,
) -> None:
    """
    Export a small combined_environment dataset into a browser-friendly Zarr store for GitHub Pages.
    Intended for small variants only (e.g. tiny).
    """
    out_zarr.parent.mkdir(parents=True, exist_ok=True)
    try:
        from numcodecs import Blosc
    except Exception as e:  # pragma: no cover
        raise RuntimeError("Missing dependency: numcodecs (needed for Zarr compression).") from e

    compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

    ds = xr.open_dataset(combined_file, engine="h5netcdf", chunks={"time": time_chunk})
    try:
        if keep_vars is not None:
            keep = [v for v in keep_vars if v in ds.data_vars]
            ds = ds[keep]

        # Rechunk to match our intended Zarr chunking to avoid dask/backend misalignment errors.
        chunk_map: dict[str, int] = {}
        if "time" in ds.dims:
            chunk_map["time"] = min(time_chunk, int(ds.sizes["time"]))
        for dim in ["depth", "latitude", "longitude"]:
            if dim in ds.dims:
                chunk_map[dim] = int(ds.sizes[dim])
        if chunk_map:
            ds = ds.chunk(chunk_map)

        encoding = {v: {"compressor": compressor} for v in ds.data_vars}
        # Chunk on time; keep spatial dims unchunked (small bbox) to minimize file count for web hosting.
        for v in ds.data_vars:
            arr = ds[v]
            chunks = []
            for dim in arr.dims:
                if dim == "time":
                    chunks.append(min(time_chunk, int(arr.sizes.get("time", 1))))
                else:
                    chunks.append(int(arr.sizes[dim]))
            encoding[v]["chunks"] = tuple(chunks)

        if out_zarr.exists():
            import shutil

            shutil.rmtree(out_zarr)

        ds.to_zarr(out_zarr, mode="w", consolidated=True, encoding=encoding, zarr_version=2)
    finally:
        ds.close()


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
        # For surface-only variants, explicitly request a single depth level and disable the
        # “depth list fallback” to avoid accidentally downloading extra depths.
        # Surface depth in CMEMS globals is typically ~0.494m; request that explicitly for stability.
        depths = [0.49402499198913574] if v.max_depth <= 1.0 else None
        fallback_to_depth_list = False if depths is not None else True
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
            depths=depths,
            fallback_to_depth_list=fallback_to_depth_list,
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

    return combined_file


def main() -> int:
    p = argparse.ArgumentParser(description="Generate combined_environment.nc variants (tiny/scene/public).")
    p.add_argument(
        "--out-root",
        default=str(Path(__file__).resolve().parent / "Data" / "Combined" / "variants"),
        help="Output root directory; each variant is written under a subfolder.",
    )
    p.add_argument("--overwrite", action="store_true", help="Backup existing outputs then overwrite.")
    p.add_argument(
        "--which",
        default="scene,public",
        help="Comma-separated variant names to generate: tiny,scene,public",
    )
    p.add_argument(
        "--variant-file",
        default=None,
        help="Path to a Variant JSON file (either a raw Variant dict or a saved variant.json with a top-level 'variant').",
    )
    p.add_argument("--name", default=None, help="Override output variant name (subfolder) for --variant-file/overrides.")

    # CLI overrides (avoid editing the script for quick experiments).
    p.add_argument("--bbox", nargs=4, type=float, default=None, metavar=("LAT_MIN", "LAT_MAX", "LON_MIN", "LON_MAX"))
    p.add_argument("--start", default=None, help="Override start datetime (e.g., 2025-01-01T00:00:00).")
    p.add_argument("--end", default=None, help="Override end datetime (e.g., 2025-12-31T00:00:00).")
    p.add_argument("--min-depth", type=float, default=None, help="Override minimum depth (m).")
    p.add_argument("--max-depth", type=float, default=None, help="Override maximum depth (m).")
    p.add_argument("--basic-dataset-id", default=None, help="Override CMEMS basic dataset id.")
    p.add_argument("--include-tides", action="store_true", help="Force include tides.")
    p.add_argument("--no-tides", action="store_true", help="Force disable tides.")
    p.add_argument("--tide-time-align", default=None, help="Override tide_time_align (nearest|linear).")
    p.add_argument("--tide-depth-profile", default=None, help="Override tide_depth_profile (broadcast|exp_decay|linear).")
    p.add_argument("--tide-z0-m", type=float, default=None, help="Override tide_z0_m for exp_decay profile.")
    p.add_argument("--tide-zmax-m", type=float, default=None, help="Override tide_zmax_m for linear profile.")
    p.add_argument(
        "--target-res-deg",
        default=None,
        help="Override target output resolution (deg). Use 'none' to disable resampling.",
    )

    # Web export (Zarr) for small variants.
    p.add_argument("--export-zarr", action="store_true", help="Export a Zarr store for web visualization.")
    p.add_argument(
        "--zarr-out-name",
        default="combined_environment.zarr",
        help="Zarr store name under <variant>/web/ (default: combined_environment.zarr).",
    )
    p.add_argument("--zarr-time-chunk", type=int, default=60, help="Time chunk size for Zarr export.")
    p.add_argument(
        "--zarr-keep-vars",
        default="thetao,uo,vo,zos,elevation,land_mask",
        help="Comma-separated data_vars to keep in Zarr export (others dropped).",
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

    builtin_variants: list[Variant] = [
        Variant(
            # Web-friendly Boston nearshore "tiny": surface-only + small bbox + higher (interpolated) output resolution.
            name="tiny",
            lat_min=42.10,
            lat_max=42.70,
            lon_min=-71.20,
            lon_max=-70.20,
            start="2025-01-01T00:00:00",
            end="2025-12-31T00:00:00",
            min_depth=0.0,
            max_depth=1.0,  # surface-only to keep artifacts small for GitHub Pages
            basic_dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            include_tides=False,
            tide_time_align="nearest",
            tide_depth_profile="broadcast",
            tide_z0_m=50.0,
            tide_zmax_m=200.0,
            # Upsample modestly for smoother web viz (this is interpolation, not new physical resolution).
            target_res_deg=0.01,
        ),
        Variant(
            name="scene",
            lat_min=32.0,
            lat_max=33.0,
            lon_min=-66.5,
            lon_max=-65.5,
            start="2025-12-01T00:00:00",
            end="2025-12-31T00:00:00",
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
            start="2025-01-01T00:00:00",
            end="2025-12-31T00:00:00",
            min_depth=0.0,
            max_depth=200.0,
            basic_dataset_id="cmems_mod_glo_phy_my_0.083deg_P1D-m",
            include_tides=False,
            tide_time_align="linear",
            tide_depth_profile="exp_decay",
            tide_z0_m=50.0,
            tide_zmax_m=200.0,
            target_res_deg=0.25,
        ),
    ]

    if args.variant_file is not None:
        src = Path(args.variant_file)
        raw = json.loads(src.read_text(encoding="utf-8"))
        vdict = raw.get("variant", raw)
        selected = [Variant(**vdict)]
    else:
        selected = [v for v in builtin_variants if v.name in which]
    if not selected:
        raise SystemExit(f"No variants selected by --which={args.which!r}")

    if args.include_tides and args.no_tides:
        raise SystemExit("Use at most one of --include-tides / --no-tides.")

    def _parse_target_res(s: str | None) -> float | None:
        if s is None:
            return None
        if isinstance(s, str) and s.strip().lower() in {"none", "null"}:
            return None
        return float(s)

    def apply_overrides(v: Variant) -> Variant:
        updates: dict = {}
        if args.name is not None:
            updates["name"] = args.name
        if args.bbox is not None:
            lat_min, lat_max, lon_min, lon_max = args.bbox
            updates.update({"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max})
        if args.start is not None:
            updates["start"] = args.start
        if args.end is not None:
            updates["end"] = args.end
        if args.min_depth is not None:
            updates["min_depth"] = float(args.min_depth)
        if args.max_depth is not None:
            updates["max_depth"] = float(args.max_depth)
        if args.basic_dataset_id is not None:
            updates["basic_dataset_id"] = args.basic_dataset_id
        if args.include_tides:
            updates["include_tides"] = True
        if args.no_tides:
            updates["include_tides"] = False
        if args.tide_time_align is not None:
            updates["tide_time_align"] = args.tide_time_align
        if args.tide_depth_profile is not None:
            updates["tide_depth_profile"] = args.tide_depth_profile
        if args.tide_z0_m is not None:
            updates["tide_z0_m"] = float(args.tide_z0_m)
        if args.tide_zmax_m is not None:
            updates["tide_zmax_m"] = float(args.tide_zmax_m)
        if args.target_res_deg is not None:
            updates["target_res_deg"] = _parse_target_res(args.target_res_deg)
        return replace(v, **updates) if updates else v

    for v0 in selected:
        v = apply_overrides(v0)
        if not args.reuse_existing:
            v_dir = out_root / v.name
            for sub in ["terrain/filtered_data.tif", "water/combined_gopaf_data.nc"]:
                pth = v_dir / sub
                if pth.exists():
                    pth.unlink()

        out = _run_one(v, base_dir=base_dir, out_root=out_root, overwrite=args.overwrite)

        web_zarr = None
        if args.export_zarr:
            keep_vars = [x.strip() for x in args.zarr_keep_vars.split(",") if x.strip()]
            web_zarr = (out_root / v.name / "web" / args.zarr_out_name).resolve()
            _export_web_zarr(Path(out), web_zarr, time_chunk=int(args.zarr_time_chunk), keep_vars=keep_vars)

        _write_metadata(
            out_root / v.name,
            v,
            terrain_file=(out_root / v.name / "terrain" / "filtered_data.tif"),
            water_file=(out_root / v.name / "water" / "combined_gopaf_data.nc"),
            combined_file=Path(out),
            web_zarr=web_zarr,
        )
        print(f"[{v.name}] OK: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
