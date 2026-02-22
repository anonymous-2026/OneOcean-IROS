from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np


def _load_var(file: h5py.File, name: str, required: bool = True):
    if name not in file:
        if required:
            raise KeyError(f"Missing variable '{name}' in dataset")
        return None
    return file[name]


def prepare_drift_cache(
    dataset_path: str | Path,
    output_path: str | Path,
    time_index: int = 0,
    depth_index: int = 0,
    include_tides: bool = True,
    include_bathymetry: bool = True,
) -> dict[str, str]:
    ds_path = Path(dataset_path).expanduser().resolve()
    if not ds_path.exists():
        raise FileNotFoundError(f"Dataset not found: {ds_path}")
    out_path = Path(output_path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(ds_path, "r") as file:
        latitude = np.asarray(_load_var(file, "latitude")[:], dtype=np.float64)
        longitude = np.asarray(_load_var(file, "longitude")[:], dtype=np.float64)
        uo = _load_var(file, "uo")
        vo = _load_var(file, "vo")
        if not (0 <= time_index < uo.shape[0]):
            raise IndexError(f"time_index out of range: {time_index}")
        if not (0 <= depth_index < uo.shape[1]):
            raise IndexError(f"depth_index out of range: {depth_index}")
        u = np.asarray(uo[time_index, depth_index, :, :], dtype=np.float64)
        v = np.asarray(vo[time_index, depth_index, :, :], dtype=np.float64)

        if include_tides:
            utide = _load_var(file, "utide", required=False)
            vtide = _load_var(file, "vtide", required=False)
            if utide is not None and vtide is not None:
                u = u + np.asarray(utide[time_index, depth_index, :, :], dtype=np.float64)
                v = v + np.asarray(vtide[time_index, depth_index, :, :], dtype=np.float64)

        land_mask = None
        elevation = None
        if include_bathymetry:
            land_mask_raw = _load_var(file, "land_mask", required=False)
            elevation_raw = _load_var(file, "elevation", required=False)
            if land_mask_raw is not None:
                land_mask = np.asarray(land_mask_raw[:, :], dtype=np.float64)
            if elevation_raw is not None:
                elevation = np.asarray(elevation_raw[:, :], dtype=np.float64)

    u = np.nan_to_num(u, nan=0.0)
    v = np.nan_to_num(v, nan=0.0)
    if land_mask is not None:
        land_mask = np.nan_to_num(land_mask, nan=0.0)
    if elevation is not None:
        elevation = np.nan_to_num(elevation, nan=0.0)

    payload = {
        "latitude": latitude,
        "longitude": longitude,
        "u": u,
        "v": v,
        "time_index": np.asarray([time_index], dtype=np.int32),
        "depth_index": np.asarray([depth_index], dtype=np.int32),
    }
    if land_mask is not None:
        payload["land_mask"] = land_mask
    if elevation is not None:
        payload["elevation"] = elevation
    np.savez_compressed(out_path, **payload)

    meta = {
        "dataset_path": str(ds_path),
        "cache_path": str(out_path),
        "time_index": int(time_index),
        "depth_index": int(depth_index),
        "include_tides": bool(include_tides),
        "include_bathymetry": bool(include_bathymetry),
        "has_land_mask": bool(land_mask is not None),
        "has_elevation": bool(elevation is not None),
        "shape": {"lat": int(latitude.size), "lon": int(longitude.size)},
    }
    meta_path = out_path.with_suffix(".json")
    with meta_path.open("w", encoding="utf-8") as file:
        json.dump(meta, file, indent=2)

    return {
        "cache_npz": str(out_path),
        "cache_meta_json": str(meta_path),
    }
