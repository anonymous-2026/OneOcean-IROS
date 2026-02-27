from __future__ import annotations

import argparse
import json
from pathlib import Path


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset",
        default="/data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/combined_environment.nc",
        help="Path to combined_environment.nc (or a variant).",
    )
    ap.add_argument("--out_npz", required=True, help="Output .npz path.")
    ap.add_argument("--lat", type=float, default=None, help="Optional latitude to sample (nearest). Default: dataset center.")
    ap.add_argument("--lon", type=float, default=None, help="Optional longitude to sample (nearest). Default: dataset center.")
    ap.add_argument("--u_var", default="uo")
    ap.add_argument("--v_var", default="vo")
    args = ap.parse_args()

    import numpy as np
    import xarray as xr

    ds_path = Path(args.dataset).expanduser().resolve()
    if not ds_path.exists():
        raise FileNotFoundError(ds_path)

    out_npz = Path(args.out_npz).expanduser().resolve()
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    out_meta = out_npz.with_suffix(".json")

    ds = xr.open_dataset(ds_path)
    if "time" not in ds.coords or "depth" not in ds.coords or "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise ValueError("Dataset missing one of required coords: time, depth, latitude, longitude")
    if args.u_var not in ds or args.v_var not in ds:
        raise ValueError(f"Dataset missing required vars: {args.u_var!r}, {args.v_var!r}")

    lat = float(args.lat) if args.lat is not None else float(ds["latitude"].mean().item())
    lon = float(args.lon) if args.lon is not None else float(ds["longitude"].mean().item())

    sub = ds[[args.u_var, args.v_var]].sel(latitude=lat, longitude=lon, method="nearest")
    lat_sel = float(sub["latitude"].item())
    lon_sel = float(sub["longitude"].item())

    u = sub[args.u_var].astype("float32").values  # (time, depth)
    v = sub[args.v_var].astype("float32").values  # (time, depth)
    if u.ndim != 2 or v.ndim != 2:
        raise ValueError(f"Expected (time, depth) arrays; got u{u.shape}, v{v.shape}")

    time_ns = sub["time"].values.astype("datetime64[ns]").astype("int64")
    depth_m = sub["depth"].values.astype("float32")

    np.savez_compressed(
        out_npz,
        time_ns=time_ns,
        depth_m=depth_m,
        latitude=float(lat_sel),
        longitude=float(lon_sel),
        uo=u,
        vo=v,
        source_dataset=str(ds_path),
    )

    meta = {
        "tool": "tracks/h3_oceangym/export_current_series_npz.py",
        "dataset": str(ds_path),
        "selected": {"lat": lat_sel, "lon": lon_sel},
        "requested": {"lat": lat, "lon": lon},
        "vars": {"u": args.u_var, "v": args.v_var},
        "shapes": {"uo": list(u.shape), "vo": list(v.shape)},
        "out_npz": str(out_npz),
    }
    _write_json(out_meta, meta)
    print("[h3] wrote:", out_npz)
    print("[h3] wrote:", out_meta)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

