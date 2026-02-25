from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import math
import numpy as np

from .drift_cache import resample_uv_to_model_grid


PollutionModelKind = Literal["gaussian", "ocpnet_3d"]


@dataclass(frozen=True)
class GaussianPlumeConfig:
    sigma_m: float = 35.0
    peak: float = 1.0


class GaussianPlumeField:
    def __init__(self, config: GaussianPlumeConfig) -> None:
        self.cfg = config
        self.source_xyz = np.zeros((3,), dtype=np.float64)
        self.center_xyz = np.zeros((3,), dtype=np.float64)
        self.mass = 1.0
        self._mass0 = 1.0

    def reset(self, rng: np.random.Generator, bounds_xyz: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
        lo, hi = bounds_xyz
        self.source_xyz = rng.uniform(lo, hi).astype(np.float64)
        self.center_xyz = self.source_xyz.copy()
        self.mass = 1.0
        self._mass0 = 1.0
        return {
            "kind": "gaussian",
            "source_xyz": self.source_xyz.tolist(),
            "config": asdict(self.cfg),
        }

    def step(self, dt_s: float) -> None:
        _ = dt_s

    def advect_center(self, drift_xz: np.ndarray, dt_s: float) -> None:
        d = np.asarray(drift_xz, dtype=np.float64).reshape(2)
        self.center_xyz[0] += float(d[0]) * float(dt_s)
        self.center_xyz[2] += float(d[1]) * float(dt_s)

    def apply_agent_sink(self, agent_xyz: np.ndarray, *, radius_m: float = 10.0, strength_per_s: float = 0.10, dt_s: float = 1.0) -> None:
        # If any agent is close to the center, reduce mass (proxy for cleanup).
        if self.mass <= 0.0:
            return
        pos = np.asarray(agent_xyz, dtype=np.float64).reshape(-1, 3)
        d = np.linalg.norm(pos - self.center_xyz[None, :], axis=1)
        if np.any(d <= float(radius_m)):
            keep = float(np.clip(1.0 - float(strength_per_s) * float(dt_s), 0.0, 1.0))
            self.mass *= keep

    def mass_fraction(self) -> float:
        return float(self.mass / max(1e-12, float(self._mass0)))

    def sample(self, xyz: np.ndarray) -> float:
        p = np.asarray(xyz, dtype=np.float64).reshape(3)
        d2 = float(np.sum((p - self.center_xyz) ** 2))
        s2 = float(self.cfg.sigma_m) ** 2
        return float(float(self.mass) * self.cfg.peak * np.exp(-0.5 * d2 / max(1e-9, s2)))


@dataclass(frozen=True)
class OCPNetConfig:
    pollutant_name: str = "microplastic"
    grid_resolution: tuple[int, int, int] = (28, 28, 10)  # (nx, ny, nz)
    time_step_s: float = 2.0
    diffusion_coefficient: float = 8e-8
    decay_rate: float = 0.0
    emission_rate: float = 0.02
    sink_radius_m: float = 8.0
    sink_strength_per_s: float = 0.15


class OCPNetPollutionField:
    """Wrapper around OCPNet PollutionModel3D with an extra agent-driven sink term.

    Coordinate convention (must match the headless env):
    - sim position is (x, y_depth, z)
    - OCPNet model uses (x, y, z_depth); we map sim z -> model y.
    """

    def __init__(
        self,
        config: OCPNetConfig,
        *,
        domain_size_m: tuple[float, float, float],
        drift_u_latlon: np.ndarray,
        drift_v_latlon: np.ndarray,
        latitude: np.ndarray,
        longitude: np.ndarray,
        output_dir: str | Path,
    ) -> None:
        from OCPNet.PollutionModel3D.src.model import PollutionModel3D

        self.cfg = config
        self.domain_size_m = tuple(float(x) for x in domain_size_m)
        nx, ny, nz = (int(config.grid_resolution[0]), int(config.grid_resolution[1]), int(config.grid_resolution[2]))
        self.model = PollutionModel3D(
            domain_size=self.domain_size_m,
            grid_resolution=(nx, ny, nz),
            time_step=float(config.time_step_s),
            output_dir=Path(output_dir),
        )

        # Velocity field from dataset (2D) resampled to model XY, then stacked in Z.
        U_yx, V_yx = resample_uv_to_model_grid(
            drift_u_latlon,
            drift_v_latlon,
            latitude=latitude,
            longitude=longitude,
            nx=nx,
            ny=ny,
        )  # (ny, nx)
        u = np.repeat(U_yx.T[:, :, None], nz, axis=2).astype(np.float64)  # (nx, ny, nz)
        v = np.repeat(V_yx.T[:, :, None], nz, axis=2).astype(np.float64)  # (nx, ny, nz)
        w = np.zeros((nx, ny, nz), dtype=np.float64)

        self.model.set_velocity_field(u, v, w)

        # Environmental defaults (kept simple but broadcast to full grid; record in run_meta).
        shape = (nx, ny, nz)
        self.model.set_environmental_field("temperature", np.full(shape, 292.0, dtype=np.float64))
        self.model.set_environmental_field("pH", np.full(shape, 7.8, dtype=np.float64))
        self.model.set_environmental_field("DO", np.full(shape, 7.5, dtype=np.float64))
        self.model.set_environmental_field("light_intensity", np.full(shape, 120.0, dtype=np.float64))
        self.model.set_environmental_field("wave_velocity", np.full(shape, 0.0, dtype=np.float64))
        self.model.set_environmental_field("salinity", np.full(shape, 34.5, dtype=np.float64))

        self.pollutant = str(config.pollutant_name)
        self.model.add_pollutant(
            name=self.pollutant,
            initial_concentration=0.0,
            molecular_weight=1.0,
            decay_rate=float(config.decay_rate) if float(config.decay_rate) > 0 else None,
            diffusion_coefficient=float(config.diffusion_coefficient),
        )

        self.source_xyz = np.zeros((3,), dtype=np.float64)
        self._rng: np.random.Generator | None = None

    def reset(self, rng: np.random.Generator, bounds_xyz: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
        # Re-initialize pollutant concentration to zeros and place a source.
        self._rng = rng
        for field in self.model.pollutant_fields.values():
            field.set_concentration(self.pollutant, np.zeros_like(field.get_concentration(self.pollutant)))

        lo, hi = bounds_xyz
        src = rng.uniform(lo, hi).astype(np.float64)
        self.source_xyz = src

        # Clear existing sources and add a point source.
        self.model.source_sink.point_sources.clear()
        self.model.source_sink.area_sources.clear()
        self.model.source_sink.line_sources.clear()
        self.model.add_source(
            type="point",
            pollutant=self.pollutant,
            position=(float(src[0]), float(src[2]), float(src[1])),  # (x, y, z) where y := sim z, z := depth
            emission_rate=float(self.cfg.emission_rate),
            time_function=None,
        )
        self.model.current_time = 0.0
        return {
            "kind": "ocpnet_3d",
            "source_xyz": self.source_xyz.tolist(),
            "config": asdict(self.cfg),
            "domain_size_m": list(self.domain_size_m),
        }

    def step(self, dt_s: float) -> None:
        # Advance the diffusion model by as many internal steps as needed.
        target = float(self.model.current_time + float(dt_s))
        while float(self.model.current_time) + 1e-9 < target:
            self.model.compute_time_step()

    def apply_agent_sink(self, agent_xyz: np.ndarray) -> None:
        """Approximate cleanup: reduce concentration locally around each agent."""
        if float(self.cfg.sink_strength_per_s) <= 0.0:
            return
        conc = self.model.pollutant_fields[self.pollutant].get_concentration(self.pollutant)
        grid = self.model.grid
        dx, dy, dz = grid.get_grid_spacing()
        rr = float(self.cfg.sink_radius_m)
        rx = max(1, int(math.ceil(rr / max(1e-6, float(dx)))))
        ry = max(1, int(math.ceil(rr / max(1e-6, float(dy)))))
        rz = max(1, int(math.ceil(rr / max(1e-6, float(dz)))))

        rate = float(self.cfg.sink_strength_per_s) * float(self.model.time_step)
        keep_factor = float(np.clip(1.0 - rate, 0.0, 1.0))

        for p in np.asarray(agent_xyz, dtype=np.float64).reshape(-1, 3):
            ix = int(np.clip(round(p[0] / dx), 0, grid.nx - 1))
            iy = int(np.clip(round(p[2] / dy), 0, grid.ny - 1))  # sim z maps to model y
            iz = int(np.clip(round(p[1] / dz), 0, grid.nz - 1))  # sim y(depth) maps to model z
            x0 = max(0, ix - rx)
            x1 = min(grid.nx, ix + rx + 1)
            y0 = max(0, iy - ry)
            y1 = min(grid.ny, iy + ry + 1)
            z0 = max(0, iz - rz)
            z1 = min(grid.nz, iz + rz + 1)
            conc[x0:x1, y0:y1, z0:z1] *= keep_factor

        self.model.pollutant_fields[self.pollutant].set_concentration(self.pollutant, conc)

    def sample(self, xyz: np.ndarray) -> float:
        p = np.asarray(xyz, dtype=np.float64).reshape(3)
        grid = self.model.grid
        dx, dy, dz = grid.get_grid_spacing()
        ix = int(np.clip(round(p[0] / dx), 0, grid.nx - 1))
        iy = int(np.clip(round(p[2] / dy), 0, grid.ny - 1))
        iz = int(np.clip(round(p[1] / dz), 0, grid.nz - 1))
        conc = self.model.pollutant_fields[self.pollutant].get_concentration(self.pollutant)
        return float(conc[ix, iy, iz])


def build_pollution_field(
    kind: PollutionModelKind,
    *,
    rng: np.random.Generator,
    bounds_xyz: tuple[np.ndarray, np.ndarray],
    output_dir: str | Path,
    drift_payload: dict[str, Any],
) -> tuple[Any, dict[str, Any]]:
    if kind == "gaussian":
        field = GaussianPlumeField(GaussianPlumeConfig())
        meta = field.reset(rng, bounds_xyz=bounds_xyz)
        return field, meta

    if kind != "ocpnet_3d":
        raise ValueError(f"Unknown pollution model kind: {kind}")

    cfg = OCPNetConfig()
    field = OCPNetPollutionField(
        cfg,
        domain_size_m=tuple(float(x) for x in drift_payload["domain_size_m"]),
        drift_u_latlon=np.asarray(drift_payload["u"], dtype=np.float64),
        drift_v_latlon=np.asarray(drift_payload["v"], dtype=np.float64),
        latitude=np.asarray(drift_payload["latitude"], dtype=np.float64),
        longitude=np.asarray(drift_payload["longitude"], dtype=np.float64),
        output_dir=Path(output_dir) / "pollution_ocpnet",
    )
    meta = field.reset(rng, bounds_xyz=bounds_xyz)
    return field, meta
