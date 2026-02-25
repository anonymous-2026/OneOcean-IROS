from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# Keep this module lightweight: avoid importing `oneocean_sim_habitat`, which pulls in
# optional heavyweight rendering deps (e.g., `cv2`). We only need basic geo conversion.
METERS_PER_DEG_LAT = 111_132.0


def meters_per_deg_lon(latitude_deg: float) -> float:
    return METERS_PER_DEG_LAT * max(0.1, float(np.cos(np.radians(latitude_deg))))


@dataclass(frozen=True)
class GridMapping:
    """Map local sim XZ (meters) to dataset lat/lon, and estimate sim bounds.

    Convention:
    - sim x: east (meters)
    - sim z: north (meters)
    - mapping origin corresponds to (lat_min, lon_min)
    """

    lat_min: float
    lon_min: float
    lat_max: float
    lon_max: float
    meters_per_deg_lon_at_lat_mid: float

    @staticmethod
    def from_latlon(latitude: np.ndarray, longitude: np.ndarray) -> "GridMapping":
        lat = np.asarray(latitude, dtype=np.float64).reshape(-1)
        lon = np.asarray(longitude, dtype=np.float64).reshape(-1)
        if lat.size < 2 or lon.size < 2:
            raise ValueError("latitude/longitude arrays must have at least 2 values")
        lat_min = float(np.min(lat))
        lat_max = float(np.max(lat))
        lon_min = float(np.min(lon))
        lon_max = float(np.max(lon))
        lat_mid = 0.5 * (lat_min + lat_max)
        return GridMapping(
            lat_min=lat_min,
            lon_min=lon_min,
            lat_max=lat_max,
            lon_max=lon_max,
            meters_per_deg_lon_at_lat_mid=float(meters_per_deg_lon(lat_mid)),
        )

    def bounds_xz_m(self) -> tuple[tuple[float, float], tuple[float, float]]:
        x_max = (self.lon_max - self.lon_min) * self.meters_per_deg_lon_at_lat_mid
        z_max = (self.lat_max - self.lat_min) * METERS_PER_DEG_LAT
        return (0.0, float(max(0.0, x_max))), (0.0, float(max(0.0, z_max)))

    def xz_to_latlon(self, x_m: float, z_m: float) -> tuple[float, float]:
        lat = self.lat_min + (float(z_m) / METERS_PER_DEG_LAT)
        lon = self.lon_min + (float(x_m) / self.meters_per_deg_lon_at_lat_mid)
        return float(lat), float(lon)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def try_load_adjacent_json(npz_path: str | Path) -> dict[str, Any] | None:
    path = Path(npz_path).expanduser().resolve()
    meta_path = path.with_suffix(".json")
    if not meta_path.exists():
        return None
    try:
        import json

        return json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None
