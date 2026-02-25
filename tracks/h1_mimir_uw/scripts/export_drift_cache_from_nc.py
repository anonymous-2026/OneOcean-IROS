#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--nc",
        type=str,
        default="OceanEnv/Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc",
        help="Path to combined_environment.nc (NetCDF4/HDF5).",
    )
    ap.add_argument("--out", type=str, required=True, help="Output .npz path")
    ap.add_argument("--u-var", type=str, default="utotal")
    ap.add_argument("--v-var", type=str, default="vtotal")
    ap.add_argument("--time-index", type=int, default=0)
    ap.add_argument("--depth-index", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    nc_path = Path(args.nc).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(nc_path, "r") as f:
        lat = np.asarray(f["latitude"][:], dtype=np.float64)
        lon = np.asarray(f["longitude"][:], dtype=np.float64)
        time = np.asarray(f["time"][:], dtype=np.float64) if "time" in f else None
        depth = np.asarray(f["depth"][:], dtype=np.float64) if "depth" in f else None

        ti = int(np.clip(args.time_index, 0, (f[args.u_var].shape[0] - 1)))
        di = int(np.clip(args.depth_index, 0, (f[args.u_var].shape[1] - 1)))

        u = np.asarray(f[args.u_var][ti, di, :, :], dtype=np.float64)
        v = np.asarray(f[args.v_var][ti, di, :, :], dtype=np.float64)
        payload = {"latitude": lat, "longitude": lon, "u": u, "v": v}
        if "elevation" in f:
            payload["elevation"] = np.asarray(f["elevation"][:], dtype=np.float64)
        if "land_mask" in f:
            payload["land_mask"] = np.asarray(f["land_mask"][:], dtype=np.float64)

        np.savez_compressed(out_path, **payload)

        meta = {
            "nc": str(nc_path),
            "u_var": str(args.u_var),
            "v_var": str(args.v_var),
            "time_index": int(ti),
            "depth_index": int(di),
            "time_value": float(time[ti]) if time is not None else None,
            "depth_value": float(depth[di]) if depth is not None else None,
            "out_npz": str(out_path),
            "grid": {"lat_n": int(lat.size), "lon_n": int(lon.size)},
        }
        (out_path.with_suffix(".json")).write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

