from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from oneocean_sim_habitat.drift import CachedDriftField

from .mapping import GridMapping, try_load_adjacent_json


@dataclass(frozen=True)
class DriftCacheInfo:
    npz_path: str
    meta_json: dict[str, Any] | None
    mapping: GridMapping

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["mapping"] = self.mapping.to_dict()
        return payload


def load_drift_cache(npz_path: str | Path) -> tuple[CachedDriftField, DriftCacheInfo]:
    cache_path = Path(npz_path).expanduser().resolve()
    field = CachedDriftField(cache_path)
    mapping = GridMapping.from_latlon(field.latitude, field.longitude)
    info = DriftCacheInfo(
        npz_path=str(cache_path),
        meta_json=try_load_adjacent_json(cache_path),
        mapping=mapping,
    )
    return field, info


def resample_uv_to_model_grid(
    u_latlon: np.ndarray,
    v_latlon: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    nx: int,
    ny: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear resample from (lat, lon) to (ny, nx) in array order (y, x)."""
    lat = np.asarray(latitude, dtype=np.float64).reshape(-1)
    lon = np.asarray(longitude, dtype=np.float64).reshape(-1)
    u = np.asarray(u_latlon, dtype=np.float64)
    v = np.asarray(v_latlon, dtype=np.float64)
    if u.shape != (lat.size, lon.size) or v.shape != (lat.size, lon.size):
        raise ValueError("u/v must have shape (lat, lon)")
    if nx < 2 or ny < 2:
        raise ValueError("nx/ny must be >= 2")

    lat_q = np.linspace(float(lat.min()), float(lat.max()), ny, dtype=np.float64)
    lon_q = np.linspace(float(lon.min()), float(lon.max()), nx, dtype=np.float64)

    # Precompute fractional indices.
    lat_idx = np.interp(lat_q, lat, np.arange(lat.size, dtype=np.float64))
    lon_idx = np.interp(lon_q, lon, np.arange(lon.size, dtype=np.float64))

    y0 = np.floor(lat_idx).astype(int)
    x0 = np.floor(lon_idx).astype(int)
    y1 = np.clip(y0 + 1, 0, lat.size - 1)
    x1 = np.clip(x0 + 1, 0, lon.size - 1)
    y0 = np.clip(y0, 0, lat.size - 1)
    x0 = np.clip(x0, 0, lon.size - 1)

    wy = (lat_idx - y0).astype(np.float64)  # (ny,)
    wx = (lon_idx - x0).astype(np.float64)  # (nx,)

    # Vectorized bilinear (build via outer products).
    U = np.zeros((ny, nx), dtype=np.float64)
    V = np.zeros((ny, nx), dtype=np.float64)
    for yi in range(ny):
        y0i = int(y0[yi])
        y1i = int(y1[yi])
        wyi = float(wy[yi])
        u_y0 = (1.0 - wyi) * u[y0i, :] + wyi * u[y1i, :]
        v_y0 = (1.0 - wyi) * v[y0i, :] + wyi * v[y1i, :]
        # interpolate along lon
        U[yi, :] = (1.0 - wx) * u_y0[x0] + wx * u_y0[x1]
        V[yi, :] = (1.0 - wx) * v_y0[x0] + wx * v_y0[x1]

    U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    return U, V

