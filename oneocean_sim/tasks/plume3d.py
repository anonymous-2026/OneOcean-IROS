from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin, sqrt
from typing import Sequence

import numpy as np


def _norm2(x: float, y: float) -> float:
    return sqrt(x * x + y * y)


def _unit2(x: float, y: float) -> tuple[float, float]:
    n = _norm2(x, y)
    if n <= 1e-9:
        return 1.0, 0.0
    return x / n, y / n


def plume_concentration(
    *,
    xy_m: tuple[float, float],
    source_xy_m: tuple[float, float],
    current_uv_mps: tuple[float, float],
    sigma_cross_m: float,
    sigma_down_m: float,
    sigma_up_m: float,
    upstream_scale: float,
) -> float:
    x, y = xy_m
    sx, sy = source_xy_m
    rx = x - sx
    ry = y - sy

    cu, cv = current_uv_mps
    ux, uy = _unit2(cu, cv)
    s = rx * ux + ry * uy  # downstream coordinate (+ downstream)
    tx = rx - s * ux
    ty = ry - s * uy
    t = _norm2(tx, ty)  # cross-stream distance

    if s >= 0.0:
        # Downstream plume.
        return float(np.exp(-0.5 * (t / max(1e-6, sigma_cross_m)) ** 2) * np.exp(-0.5 * (s / max(1e-6, sigma_down_m)) ** 2))

    # Upstream: low concentration with exponential decay as you move upstream.
    return float(
        0.05
        * np.exp(-0.5 * (t / max(1e-6, sigma_cross_m)) ** 2)
        * np.exp(-0.5 * (s / max(1e-6, sigma_up_m)) ** 2)
        * np.exp(s / max(1e-6, upstream_scale))
    )


@dataclass
class PlumeMultiAgentTaskConfig:
    desired_depth_z_m: float = -4.0
    max_steps: int = 1100
    detection_threshold: float = 0.16
    source_tolerance_m: float = 10.0
    cruise_speed_mps: float = 1.8
    cast_speed_mps: float = 1.2
    cast_period_steps: int = 90
    cast_amplitude: float = 1.0
    # Plume geometry parameters (in sim meters).
    sigma_cross_m: float = 18.0
    sigma_down_m: float = 55.0
    sigma_up_m: float = 28.0
    upstream_scale_m: float = 30.0
    sensor_noise_std: float = 0.01


@dataclass
class CastAndSurgeController:
    cruise_speed_mps: float = 1.2
    cast_speed_mps: float = 0.9
    cast_period_steps: int = 90
    cast_sign: float = 1.0  # +1 or -1
    detection_threshold: float = 0.16
    compensate_current: bool = True

    def act(
        self,
        *,
        step_index: int,
        obs: dict[str, float],
        concentration: float,
        global_detected: bool,
    ) -> dict[str, float]:
        x = float(obs["x_m"])
        y = float(obs["y_m"])
        cu = float(obs.get("current_u_mps", 0.0))
        cv = float(obs.get("current_v_mps", 0.0))
        upx, upy = _unit2(-cu, -cv)
        crx, cry = _unit2(-upy, upx)  # cross-current (90deg)

        if global_detected or concentration >= self.detection_threshold:
            vx = self.cruise_speed_mps * upx
            vy = self.cruise_speed_mps * upy
        else:
            phase = (step_index % max(1, int(self.cast_period_steps))) / max(1.0, float(self.cast_period_steps))
            sweep = sin(2.0 * np.pi * phase)
            vx = 0.35 * self.cast_speed_mps * upx + (self.cast_sign * sweep) * self.cast_speed_mps * crx
            vy = 0.35 * self.cast_speed_mps * upy + (self.cast_sign * sweep) * self.cast_speed_mps * cry

        if self.compensate_current:
            vx -= cu
            vy -= cv

        yaw = atan2(vy, vx)
        return {"vx_mps": float(vx), "vy_mps": float(vy), "yaw_rad": float(yaw)}


def sample_source_xy(
    rng: np.random.Generator,
    *,
    x_half_m: float,
    y_half_m: float,
) -> tuple[float, float]:
    return (
        float(rng.uniform(-0.65 * x_half_m, 0.65 * x_half_m)),
        float(rng.uniform(-0.65 * y_half_m, 0.65 * y_half_m)),
    )
