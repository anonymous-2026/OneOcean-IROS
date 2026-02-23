from __future__ import annotations

from dataclasses import dataclass
from math import floor
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class DiffusionPlumeField2D:
    x_min_m: float
    y_min_m: float
    dx_m: float
    dy_m: float
    concentration: np.ndarray  # shape: (ny, nx), normalized to [0, 1]

    def sample(self, x_m: float, y_m: float) -> float:
        ny, nx = self.concentration.shape
        fx = (float(x_m) - self.x_min_m) / max(1e-9, self.dx_m)
        fy = (float(y_m) - self.y_min_m) / max(1e-9, self.dy_m)
        if fx < 0.0 or fy < 0.0 or fx > (nx - 1) or fy > (ny - 1):
            return 0.0
        x0 = int(floor(fx))
        y0 = int(floor(fy))
        x1 = min(nx - 1, x0 + 1)
        y1 = min(ny - 1, y0 + 1)
        wx = fx - float(x0)
        wy = fy - float(y0)
        c00 = float(self.concentration[y0, x0])
        c10 = float(self.concentration[y0, x1])
        c01 = float(self.concentration[y1, x0])
        c11 = float(self.concentration[y1, x1])
        c0 = (1.0 - wx) * c00 + wx * c10
        c1 = (1.0 - wx) * c01 + wx * c11
        return float((1.0 - wy) * c0 + wy * c1)


def build_diffusion_plume_field_2d(
    *,
    x_half_m: float,
    y_half_m: float,
    source_xy_m: tuple[float, float],
    current_uv_at_xy: Callable[[float, float], tuple[float, float]],
    nx: int = 96,
    ny: int = 96,
    dt_sec: float = 0.35,
    steps: int = 160,
    diffusion_m2ps: float = 4.5,
    decay_rate_per_sec: float = 0.004,
    source_rate: float = 1.0,
    domain_scale: float = 0.88,
) -> DiffusionPlumeField2D:
    """Build a coarse 2D advection-diffusion plume field driven by dataset currents.

    This is a lightweight proxy used for robotics tasks (probe-based localization).
    It is NOT a measured pollution field.
    """
    nx = int(max(24, nx))
    ny = int(max(24, ny))
    scale = float(np.clip(domain_scale, 0.4, 0.98))
    x_min = -scale * float(x_half_m)
    x_max = scale * float(x_half_m)
    y_min = -scale * float(y_half_m)
    y_max = scale * float(y_half_m)
    dx = (x_max - x_min) / max(1, nx - 1)
    dy = (y_max - y_min) / max(1, ny - 1)

    xs = x_min + dx * np.arange(nx, dtype=np.float64)
    ys = y_min + dy * np.arange(ny, dtype=np.float64)

    u = np.zeros((ny, nx), dtype=np.float64)
    v = np.zeros((ny, nx), dtype=np.float64)
    for yi, y in enumerate(ys):
        for xi, x in enumerate(xs):
            uu, vv = current_uv_at_xy(float(x), float(y))
            u[yi, xi] = float(uu)
            v[yi, xi] = float(vv)

    conc = np.zeros((ny, nx), dtype=np.float64)
    sx, sy = source_xy_m
    sx_idx = int(np.clip(round((float(sx) - x_min) / max(1e-9, dx)), 0, nx - 1))
    sy_idx = int(np.clip(round((float(sy) - y_min) / max(1e-9, dy)), 0, ny - 1))

    dt = float(max(1e-6, dt_sec))
    D = float(max(0.0, diffusion_m2ps))
    decay = float(max(0.0, decay_rate_per_sec))
    src = float(max(0.0, source_rate))

    for _ in range(int(max(1, steps))):
        conc[sy_idx, sx_idx] += src * dt

        u_plus = np.maximum(u, 0.0)
        u_minus = np.minimum(u, 0.0)
        v_plus = np.maximum(v, 0.0)
        v_minus = np.minimum(v, 0.0)

        adv = np.zeros_like(conc)
        adv[:, 1:-1] += (
            -u_plus[:, 1:-1] * (conc[:, 1:-1] - conc[:, :-2]) / max(1e-9, dx)
            -u_minus[:, 1:-1] * (conc[:, 2:] - conc[:, 1:-1]) / max(1e-9, dx)
        )
        adv[1:-1, :] += (
            -v_plus[1:-1, :] * (conc[1:-1, :] - conc[:-2, :]) / max(1e-9, dy)
            -v_minus[1:-1, :] * (conc[2:, :] - conc[1:-1, :]) / max(1e-9, dy)
        )

        diff = np.zeros_like(conc)
        diff[:, 1:-1] += (conc[:, 2:] - 2.0 * conc[:, 1:-1] + conc[:, :-2]) / max(1e-9, dx * dx)
        diff[1:-1, :] += (conc[2:, :] - 2.0 * conc[1:-1, :] + conc[:-2, :]) / max(1e-9, dy * dy)
        diff *= D

        conc = conc + dt * (adv + diff - decay * conc)
        conc = np.clip(conc, 0.0, None)

        # Zero-gradient boundaries.
        conc[:, 0] = conc[:, 1]
        conc[:, -1] = conc[:, -2]
        conc[0, :] = conc[1, :]
        conc[-1, :] = conc[-2, :]

    max_c = float(np.max(conc))
    if max_c > 1e-12:
        conc = conc / max_c

    return DiffusionPlumeField2D(
        x_min_m=float(x_min),
        y_min_m=float(y_min),
        dx_m=float(dx),
        dy_m=float(dy),
        concentration=conc.astype(np.float32),
    )

