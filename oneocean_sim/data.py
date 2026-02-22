from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import os

import h5py
import numpy as np


REQUIRED_VARS = ("uo", "vo", "time", "depth", "latitude", "longitude", "elevation")


@dataclass(frozen=True)
class DatasetContext:
    dataset_path: Path
    variant: str
    time_index: int
    depth_index: int


def _default_variants_root() -> Path:
    repo_root = Path(__file__).resolve().parents[1]
    in_repo = repo_root / "OceanEnv" / "Data_pipeline" / "Data" / "Combined" / "variants"
    workspace_sibling = repo_root.parent / "OceanEnv" / "Data_pipeline" / "Data" / "Combined" / "variants"

    if in_repo.exists():
        return in_repo
    return workspace_sibling


def resolve_dataset_path(dataset_path: Optional[str], variant: str) -> Path:
    if dataset_path:
        path = Path(dataset_path).expanduser().resolve()
    else:
        variants_root = Path(
            os.environ.get("ONEOCEAN_VARIANTS_ROOT", str(_default_variants_root()))
        ).expanduser()
        path = (variants_root / variant / "combined" / "combined_environment.nc").resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    return path


class CombinedDataset:
    def __init__(self, dataset_path: Path) -> None:
        self.path = dataset_path
        self.file = h5py.File(dataset_path, "r")
        for var in REQUIRED_VARS:
            if var not in self.file:
                self.file.close()
                raise ValueError(f"Missing required variable/coord: {var}")

        self.time = self.file["time"]
        self.depth = self.file["depth"]
        self.latitude = self.file["latitude"]
        self.longitude = self.file["longitude"]
        self.uo = self.file["uo"]
        self.vo = self.file["vo"]
        self.elevation = self.file["elevation"] if "elevation" in self.file else None
        self.utide = self.file["utide"] if "utide" in self.file else None
        self.vtide = self.file["vtide"] if "vtide" in self.file else None
        self.land_mask = self.file["land_mask"] if "land_mask" in self.file else None

        self.latitude_values = np.asarray(self.latitude[:], dtype=np.float64)
        self.longitude_values = np.asarray(self.longitude[:], dtype=np.float64)
        self._elevation_grid: Optional[np.ndarray] = None

    @property
    def sizes(self) -> dict[str, int]:
        return {
            "time": int(self.time.shape[0]),
            "depth": int(self.depth.shape[0]),
            "latitude": int(self.latitude.shape[0]),
            "longitude": int(self.longitude.shape[0]),
        }

    def close(self) -> None:
        if self.file:
            self.file.close()

    def elevation_grid(self) -> np.ndarray:
        if self.elevation is None:
            raise ValueError("Dataset missing required variable: elevation")
        if self._elevation_grid is None:
            self._elevation_grid = np.asarray(self.elevation[:, :], dtype=np.float32)
        return self._elevation_grid

    def nearest_latlon_indices(self, latitude: float, longitude: float) -> tuple[int, int]:
        lat_idx = int(np.argmin(np.abs(self.latitude_values - latitude)))
        lon_idx = int(np.argmin(np.abs(self.longitude_values - longitude)))
        return lat_idx, lon_idx

    def nearest_uv(
        self, latitude: float, longitude: float, time_index: int, depth_index: int, include_tides: bool
    ) -> tuple[float, float]:
        lat_idx, lon_idx = self.nearest_latlon_indices(latitude, longitude)
        u = float(self.uo[time_index, depth_index, lat_idx, lon_idx])
        v = float(self.vo[time_index, depth_index, lat_idx, lon_idx])
        if include_tides and self.utide is not None and self.vtide is not None:
            u += float(self.utide[time_index, depth_index, lat_idx, lon_idx])
            v += float(self.vtide[time_index, depth_index, lat_idx, lon_idx])
        if np.isnan(u):
            u = 0.0
        if np.isnan(v):
            v = 0.0
        return u, v

    def invalid_region(self, latitude: float, longitude: float) -> bool:
        if self.land_mask is None:
            return False
        lat_idx, lon_idx = self.nearest_latlon_indices(latitude, longitude)
        value = float(self.land_mask[lat_idx, lon_idx])
        if np.isnan(value):
            return True
        return int(round(value)) != 0

    def center_latlon(self) -> tuple[float, float]:
        return float(np.mean(self.latitude_values)), float(np.mean(self.longitude_values))

    def uv_grid(self, time_index: int, depth_index: int, include_tides: bool) -> tuple[np.ndarray, np.ndarray]:
        u = np.asarray(self.uo[time_index, depth_index, :, :], dtype=np.float64)
        v = np.asarray(self.vo[time_index, depth_index, :, :], dtype=np.float64)
        if include_tides and self.utide is not None and self.vtide is not None:
            u = u + np.asarray(self.utide[time_index, depth_index, :, :], dtype=np.float64)
            v = v + np.asarray(self.vtide[time_index, depth_index, :, :], dtype=np.float64)
        u = np.nan_to_num(u, nan=0.0)
        v = np.nan_to_num(v, nan=0.0)
        return u, v


def open_dataset(dataset_path: Path) -> CombinedDataset:
    return CombinedDataset(dataset_path)


def validate_time_depth_indices(ds: CombinedDataset, time_index: int, depth_index: int) -> tuple[int, int]:
    if not (0 <= time_index < ds.sizes["time"]):
        raise IndexError(f"time_index {time_index} out of range [0, {ds.sizes['time'] - 1}]")
    if not (0 <= depth_index < ds.sizes["depth"]):
        raise IndexError(f"depth_index {depth_index} out of range [0, {ds.sizes['depth'] - 1}]")
    return time_index, depth_index


class CurrentSampler:
    def __init__(self, ds: CombinedDataset, include_tides: bool = True) -> None:
        self.ds = ds
        self.include_tides = include_tides

    @property
    def bounds(self) -> dict[str, float]:
        return {
            "lat_min": float(np.min(self.ds.latitude_values)),
            "lat_max": float(np.max(self.ds.latitude_values)),
            "lon_min": float(np.min(self.ds.longitude_values)),
            "lon_max": float(np.max(self.ds.longitude_values)),
        }

    def center_latlon(self) -> tuple[float, float]:
        return self.ds.center_latlon()

    def sample_uv(
        self, latitude: float, longitude: float, time_index: int, depth_index: int
    ) -> tuple[float, float]:
        return self.ds.nearest_uv(
            latitude=latitude,
            longitude=longitude,
            time_index=time_index,
            depth_index=depth_index,
            include_tides=self.include_tides,
        )

    def invalid_region(self, latitude: float, longitude: float) -> bool:
        return self.ds.invalid_region(latitude, longitude)
