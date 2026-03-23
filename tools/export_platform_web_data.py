#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_INPUT = REPO_ROOT / "Data_pipeline" / "Data" / "Combined" / "variants" / "public" / "combined" / "combined_environment.nc"
DEFAULT_OUTPUT = REPO_ROOT / "docs" / "static" / "data" / "oneocean_public_currents_subset.json"


def rounded_list(array, digits=4):
    return np.round(np.asarray(array, dtype=np.float32), digits).tolist()


def resolve_input(path: Path) -> Path:
    if path.exists():
        return path
    raise FileNotFoundError(
        f"Input dataset not found at {path}. Pass --input with a local combined_environment.nc path."
    )


def main():
    parser = argparse.ArgumentParser(description="Export a GitHub-Pages-safe ocean-data subset for the OneOcean platform page.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input combined_environment.nc path.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path under docs/static/data/.")
    parser.add_argument("--time-step", type=int, default=30, help="Stride for selecting time indices.")
    parser.add_argument("--space-step", type=int, default=2, help="Stride for latitude/longitude downsampling.")
    parser.add_argument("--depth-count", type=int, default=4, help="Number of sampled depth levels for the web subset.")
    args = parser.parse_args()

    input_path = resolve_input(args.input)
    ds = xr.open_dataset(input_path)
    try:
        time_indices = list(range(0, int(ds.sizes["time"]), args.time_step))
        if time_indices[-1] != int(ds.sizes["time"]) - 1:
            time_indices.append(int(ds.sizes["time"]) - 1)
        depth_indices = np.unique(np.round(np.linspace(0, int(ds.sizes["depth"]) - 1, args.depth_count)).astype(int))
        lat_indices = np.arange(0, int(ds.sizes["latitude"]), args.space_step, dtype=int)
        lon_indices = np.arange(0, int(ds.sizes["longitude"]), args.space_step, dtype=int)
        if lat_indices[-1] != int(ds.sizes["latitude"]) - 1:
            lat_indices = np.append(lat_indices, int(ds.sizes["latitude"]) - 1)
        if lon_indices[-1] != int(ds.sizes["longitude"]) - 1:
            lon_indices = np.append(lon_indices, int(ds.sizes["longitude"]) - 1)

        subset = ds.isel(time=time_indices, depth=depth_indices, latitude=lat_indices, longitude=lon_indices)
        zos_subset = ds["zos"].isel(time=time_indices, latitude=lat_indices, longitude=lon_indices)
        payload = {
            "metadata": {
                "title": "OneOcean public subset for GitHub Pages explorer",
                "source_dataset": input_path.name,
                "source_variant": "public",
                "time_stride": args.time_step,
                "space_stride": args.space_step,
                "depth_count": int(len(depth_indices)),
                "grid_shape": [int(len(lat_indices)), int(len(lon_indices))],
                "variables": ["uo", "vo", "thetao", "so", "zos", "elevation", "land_mask"],
                "units": {
                    "currents": "m/s",
                    "temperature": "degC",
                    "salinity": "psu",
                    "ssh": "m",
                    "elevation": "m",
                    "latitude": "degree_north",
                    "longitude": "degree_east",
                },
            },
            "time": [str(value) for value in subset["time"].values],
            "depth": rounded_list(subset["depth"].values, digits=3),
            "latitude": rounded_list(subset["latitude"].values, digits=3),
            "longitude": rounded_list(subset["longitude"].values, digits=3),
            "elevation": rounded_list(subset["elevation"].values, digits=1),
            "land_mask": rounded_list(subset["land_mask"].values, digits=0),
            "u": rounded_list(subset["uo"].values, digits=4),
            "v": rounded_list(subset["vo"].values, digits=4),
            "temperature": rounded_list(subset["thetao"].values, digits=4),
            "salinity": rounded_list(subset["so"].values, digits=4),
            "ssh": rounded_list(zos_subset.values, digits=5),
        }
    finally:
        ds.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
