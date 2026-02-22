from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

METERS_PER_DEG_LAT = 111_132.0


@dataclass(frozen=True)
class DriftConfig:
    mode: str = "synthetic_wave"
    amplitude_mps: float = 0.35
    spatial_scale_m: float = 8.0
    temporal_scale_steps: float = 20.0
    bias_x_mps: float = 0.0
    bias_z_mps: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def meters_per_deg_lon(latitude_deg: float) -> float:
    return METERS_PER_DEG_LAT * max(0.1, float(np.cos(np.radians(latitude_deg))))


def xz_to_latlon(x_m: float, z_m: float, lat0: float, lon0: float) -> tuple[float, float]:
    lat = lat0 + (z_m / METERS_PER_DEG_LAT)
    lon = lon0 + (x_m / meters_per_deg_lon(lat0))
    return float(lat), float(lon)


class CachedDriftField:
    """Lightweight drift sampler backed by a pre-exported npz cache."""

    def __init__(self, cache_path: str | Path) -> None:
        path = Path(cache_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Drift cache not found: {path}")
        payload = np.load(path)
        self.path = path
        self.latitude = np.asarray(payload["latitude"], dtype=np.float64)
        self.longitude = np.asarray(payload["longitude"], dtype=np.float64)
        self.u = np.asarray(payload["u"], dtype=np.float64)
        self.v = np.asarray(payload["v"], dtype=np.float64)
        if self.u.shape != self.v.shape:
            raise ValueError(f"u/v shape mismatch in drift cache: {self.u.shape} vs {self.v.shape}")
        if self.u.shape != (self.latitude.size, self.longitude.size):
            raise ValueError(
                f"drift field shape mismatch: {(self.latitude.size, self.longitude.size)} expected, got {self.u.shape}"
            )
        self.elevation: np.ndarray | None = None
        if "elevation" in payload.files:
            elevation = np.asarray(payload["elevation"], dtype=np.float64)
            if elevation.shape != self.u.shape:
                raise ValueError(
                    f"elevation shape mismatch: {elevation.shape} vs {self.u.shape}"
                )
            self.elevation = elevation
        self.land_mask: np.ndarray | None = None
        if "land_mask" in payload.files:
            land_mask = np.asarray(payload["land_mask"], dtype=np.float64)
            if land_mask.shape != self.u.shape:
                raise ValueError(
                    f"land_mask shape mismatch: {land_mask.shape} vs {self.u.shape}"
                )
            self.land_mask = land_mask

    def center_latlon(self) -> tuple[float, float]:
        return float(np.mean(self.latitude)), float(np.mean(self.longitude))

    def _indices_from_xz(
        self,
        x_m: float,
        z_m: float,
        origin_lat: float,
        origin_lon: float,
    ) -> tuple[int, int]:
        lat, lon = xz_to_latlon(x_m=x_m, z_m=z_m, lat0=origin_lat, lon0=origin_lon)
        lat_idx = int(np.argmin(np.abs(self.latitude - lat)))
        lon_idx = int(np.argmin(np.abs(self.longitude - lon)))
        return lat_idx, lon_idx

    def sample_xz(
        self,
        x_m: float,
        z_m: float,
        origin_lat: float,
        origin_lon: float,
    ) -> tuple[float, float]:
        lat_idx, lon_idx = self._indices_from_xz(
            x_m=x_m,
            z_m=z_m,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
        )
        drift_x = float(np.nan_to_num(self.u[lat_idx, lon_idx], nan=0.0))
        drift_z = float(np.nan_to_num(self.v[lat_idx, lon_idx], nan=0.0))
        return drift_x, drift_z

    def sample_land_mask_xz(
        self,
        x_m: float,
        z_m: float,
        origin_lat: float,
        origin_lon: float,
    ) -> float | None:
        if self.land_mask is None:
            return None
        lat_idx, lon_idx = self._indices_from_xz(
            x_m=x_m,
            z_m=z_m,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
        )
        return float(np.nan_to_num(self.land_mask[lat_idx, lon_idx], nan=0.0))

    def sample_elevation_xz(
        self,
        x_m: float,
        z_m: float,
        origin_lat: float,
        origin_lon: float,
    ) -> float | None:
        if self.elevation is None:
            return None
        lat_idx, lon_idx = self._indices_from_xz(
            x_m=x_m,
            z_m=z_m,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
        )
        return float(np.nan_to_num(self.elevation[lat_idx, lon_idx], nan=0.0))

    def is_blocked_xz(
        self,
        x_m: float,
        z_m: float,
        origin_lat: float,
        origin_lon: float,
        land_mask_threshold: float = 0.5,
        elevation_threshold: float | None = None,
    ) -> bool:
        mask_value = self.sample_land_mask_xz(
            x_m=x_m,
            z_m=z_m,
            origin_lat=origin_lat,
            origin_lon=origin_lon,
        )
        if mask_value is not None and mask_value >= float(land_mask_threshold):
            return True
        if elevation_threshold is not None:
            elevation_value = self.sample_elevation_xz(
                x_m=x_m,
                z_m=z_m,
                origin_lat=origin_lat,
                origin_lon=origin_lon,
            )
            if elevation_value is not None and elevation_value >= float(elevation_threshold):
                return True
        return False


def sample_drift_xz(position_xyz: np.ndarray, step_index: int, config: DriftConfig) -> tuple[float, float]:
    """Return a coarse current/drift vector on Habitat's ground plane (x, z)."""
    if config.mode != "synthetic_wave":
        return config.bias_x_mps, config.bias_z_mps

    x = float(position_xyz[0])
    z = float(position_xyz[2])
    wave_t = float(step_index) / max(1.0, config.temporal_scale_steps)
    wave_x = np.sin((x / max(1e-3, config.spatial_scale_m)) + wave_t)
    wave_z = np.cos((z / max(1e-3, config.spatial_scale_m)) - wave_t)
    drift_x = config.bias_x_mps + config.amplitude_mps * float(wave_x)
    drift_z = config.bias_z_mps + config.amplitude_mps * float(wave_z)
    return drift_x, drift_z
