from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class FieldMapping:
    width_m: float
    height_m: float
    origin_x: float = 0.0
    origin_y: float = 0.0
    clamp: bool = True


class DatasetCurrentField:
    def __init__(
        self,
        nc_path: str | Path,
        mapping: FieldMapping,
        time_index: int = 0,
        depth_index: int = 0,
        u_name: str = "uo",
        v_name: str = "vo",
    ):
        import xarray as xr

        self._mapping = mapping
        self._ds = xr.open_dataset(Path(nc_path))
        self._u = self._ds[u_name].isel(time=time_index, depth=depth_index).values.astype(np.float32)
        self._v = self._ds[v_name].isel(time=time_index, depth=depth_index).values.astype(np.float32)

        # For interpolation we operate purely in index space; lat/lon are not assumed to match AirSim meters.
        self._h, self._w = self._u.shape

    def close(self) -> None:
        self._ds.close()

    def _xy_to_ij(self, x: float, y: float) -> Tuple[float, float]:
        mx = (x - self._mapping.origin_x) / max(1e-6, self._mapping.width_m)
        my = (y - self._mapping.origin_y) / max(1e-6, self._mapping.height_m)
        if self._mapping.clamp:
            mx = float(np.clip(mx, 0.0, 1.0))
            my = float(np.clip(my, 0.0, 1.0))
        # y maps to row (i), x maps to col (j)
        i = my * (self._h - 1)
        j = mx * (self._w - 1)
        return i, j

    def sample_uv(self, x: float, y: float) -> Tuple[float, float]:
        i, j = self._xy_to_ij(x, y)
        i0 = int(np.floor(i))
        j0 = int(np.floor(j))
        i1 = min(i0 + 1, self._h - 1)
        j1 = min(j0 + 1, self._w - 1)
        di = float(i - i0)
        dj = float(j - j0)

        def bilinear(a):
            return (
                (1 - di) * (1 - dj) * a[i0, j0]
                + (1 - di) * dj * a[i0, j1]
                + di * (1 - dj) * a[i1, j0]
                + di * dj * a[i1, j1]
            )

        u = float(bilinear(self._u))
        v = float(bilinear(self._v))
        if not np.isfinite(u):
            u = 0.0
        if not np.isfinite(v):
            v = 0.0
        return u, v


class AdvectedGaussianPlume:
    """A lightweight, dataset-driven plume proxy.

    This is not a full diffusion PDE solve; it is an engineering proxy that:
    - maintains a Gaussian concentration field,
    - advects the center using the dataset current field,
    - is deterministic and cheap to query per step for multi-agent tasks.
    """

    def __init__(
        self,
        current_field: DatasetCurrentField,
        center_xy: Tuple[float, float],
        sigma_m: float = 6.0,
        advect_gain: float = 1.0,
    ):
        self._field = current_field
        self._cx, self._cy = float(center_xy[0]), float(center_xy[1])
        self._sigma = float(sigma_m)
        self._advect_gain = float(advect_gain)

    @property
    def center_xy(self) -> Tuple[float, float]:
        return self._cx, self._cy

    def step(self, dt_s: float) -> None:
        u, v = self._field.sample_uv(self._cx, self._cy)
        self._cx += self._advect_gain * u * dt_s
        self._cy += self._advect_gain * v * dt_s

    def concentration(self, x: float, y: float) -> float:
        dx = float(x - self._cx)
        dy = float(y - self._cy)
        r2 = dx * dx + dy * dy
        s2 = max(1e-6, self._sigma * self._sigma)
        return float(np.exp(-0.5 * r2 / s2))

