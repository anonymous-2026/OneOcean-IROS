#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path

import xarray as xr

import copernicusmarine


def _require_env(key: str) -> str:
    v = os.environ.get(key)
    if not v:
        raise SystemExit(f"Missing env var: {key}")
    return v


def _depth_values(path: Path) -> list[float]:
    ds = xr.open_dataset(path)
    if "depth" not in ds:
        return []
    return [float(x) for x in ds["depth"].values]


def main() -> int:
    username = _require_env("COPERNICUSMARINE_USERNAME")
    password = _require_env("COPERNICUSMARINE_PASSWORD")

    out_dir = Path("OceanEnv/Data_pipeline/Data/GOPAF/_depth_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Small bbox/time for a quick test.
    kwargs = dict(
        minimum_longitude=-66.5,
        maximum_longitude=-65.5,
        minimum_latitude=32.0,
        maximum_latitude=33.0,
        start_datetime="2024-06-01T00:00:00",
        end_datetime="2024-06-01T03:00:00",
    )

    dsid = "cmems_mod_glo_phy_anfc_0.083deg_PT1H-m"
    out_range = out_dir / "range.nc"
    out_single = out_dir / "single_0p494.nc"

    # Depth range request
    copernicusmarine.subset(
        dataset_id=dsid,
        username=username,
        password=password,
        variables=["uo", "vo"],
        minimum_depth=0.0,
        maximum_depth=200.0,
        output_filename=str(out_range),
        **kwargs,
    )
    print("range depth values:", _depth_values(out_range))

    # Single depth request
    try:
        copernicusmarine.subset(
            dataset_id=dsid,
            username=username,
            password=password,
            variables=["uo", "vo"],
            minimum_depth=0.49402499198913574,
            maximum_depth=0.49402499198913574,
            output_filename=str(out_single),
            **kwargs,
        )
        print("single depth values:", _depth_values(out_single))
    except Exception as e:
        print("single depth request failed:", type(e).__name__, str(e))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
