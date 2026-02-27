from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OCPNetCfg:
    domain_size_m: tuple[float, float, float] = (240.0, 240.0, 60.0)
    grid_resolution: tuple[int, int, int] = (24, 24, 12)
    time_step_s: float = 0.05
    initial_concentration: float = 0.0
    diffusion_coefficient: float = 8e-8
    decay_rate: float = 8e-7
    emission_rate: float = 0.015


class OCPNetPlume:
    """
    Lightweight wrapper around OCPNet PollutionModel3D for H3 tasks.

    Coordinate convention (H3-local):
    - We build a local frame centered at `world_center_xyz`.
    - PollutionModel3D grid coordinates are in meters with origin at (0,0,0).
    - A world point is mapped into the model by subtracting (center - domain/2).
    """

    def __init__(self, *, cfg: OCPNetCfg, work_dir: Path, world_center_xyz: tuple[float, float, float]):
        import numpy as np

        from OCPNet.PollutionModel3D.src.model import PollutionModel3D

        self.cfg = cfg
        self.work_dir = Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.center_world = tuple(float(x) for x in world_center_xyz)
        self._origin_world = (
            self.center_world[0] - cfg.domain_size_m[0] * 0.5,
            self.center_world[1] - cfg.domain_size_m[1] * 0.5,
            self.center_world[2] - cfg.domain_size_m[2] * 0.5,
        )

        self.model = PollutionModel3D(
            domain_size=cfg.domain_size_m,
            grid_resolution=cfg.grid_resolution,
            time_step=float(cfg.time_step_s),
            output_dir=self.work_dir,
        )

        nx, ny, nz = cfg.grid_resolution
        u = np.zeros((nx, ny, nz), dtype=np.float32)
        v = np.zeros((nx, ny, nz), dtype=np.float32)
        w = np.zeros((nx, ny, nz), dtype=np.float32)
        self.model.set_velocity_field(u, v, w)

        # Minimal environmental fields required by compute_time_step().
        z = np.linspace(0.0, 1.0, nz, dtype=np.float32)
        zz = np.tile(z.reshape(1, 1, nz), (nx, ny, 1))
        self.model.set_environmental_field("temperature", 292.0 + 3.0 * np.exp(-1.7 * zz))
        self.model.set_environmental_field("pH", 7.7 - 0.4 * zz)
        self.model.set_environmental_field("DO", 7.8 - 1.8 * zz)
        self.model.set_environmental_field("light_intensity", 900.0 * np.exp(-2.6 * zz))
        self.model.set_environmental_field("wave_velocity", 0.08 * np.exp(-2.1 * zz))
        self.model.set_environmental_field("salinity", 34.2 + 0.5 * zz)

        self.pollutant = "microplastic"
        self.model.add_pollutant(
            name=self.pollutant,
            initial_concentration=float(cfg.initial_concentration),
            molecular_weight=1.0,
            decay_rate=float(cfg.decay_rate),
            diffusion_coefficient=float(cfg.diffusion_coefficient),
        )

        self._source_model_xyz: tuple[float, float, float] | None = None

    def _world_to_model_xyz(self, world_xyz: tuple[float, float, float]) -> tuple[float, float, float]:
        return (
            float(world_xyz[0] - self._origin_world[0]),
            float(world_xyz[1] - self._origin_world[1]),
            float(world_xyz[2] - self._origin_world[2]),
        )

    def _model_to_grid_ijk(self, model_xyz: tuple[float, float, float]) -> tuple[int, int, int]:
        import numpy as np

        x, y, z = model_xyz
        i = int(round((x - float(self.model.grid.origin[0])) / float(self.model.grid.dx)))
        j = int(round((y - float(self.model.grid.origin[1])) / float(self.model.grid.dy)))
        k = int(round((z - float(self.model.grid.origin[2])) / float(self.model.grid.dz)))
        i = int(np.clip(i, 0, self.model.grid.nx - 1))
        j = int(np.clip(j, 0, self.model.grid.ny - 1))
        k = int(np.clip(k, 0, self.model.grid.nz - 1))
        return i, j, k

    def set_source_world(self, world_xyz: tuple[float, float, float]) -> None:
        src_model = self._world_to_model_xyz(world_xyz)
        self._source_model_xyz = src_model
        self.model.add_source(
            type="point",
            pollutant=self.pollutant,
            position=src_model,
            emission_rate=float(self.cfg.emission_rate),
            time_function=lambda t: 1.0,
        )

    def freeze_source(self) -> None:
        for s in getattr(self.model.source_sink, "point_sources", []):
            if s.get("pollutant") == self.pollutant and s.get("type") == "point":
                s["emission_rate"] = 0.0
                s["time_function"] = (lambda _t: 0.0)

    def step(self, *, u_mps: float, v_mps: float) -> None:
        import numpy as np

        nx, ny, nz = self.cfg.grid_resolution
        u = np.full((nx, ny, nz), float(u_mps), dtype=np.float32)
        v = np.full((nx, ny, nz), float(v_mps), dtype=np.float32)
        w = np.zeros((nx, ny, nz), dtype=np.float32)
        self.model.set_velocity_field(u, v, w)
        self.model.compute_time_step()

    def concentration_at_world(self, world_xyz: tuple[float, float, float]) -> float:
        c = self.model.pollutant_fields[self.pollutant].get_concentration(self.pollutant)
        i, j, k = self._model_to_grid_ijk(self._world_to_model_xyz(world_xyz))
        return float(c[i, j, k])

    def apply_sink_at_world(self, world_xyz: tuple[float, float, float], *, sink_radius_m: float, sink_strength: float) -> float:
        """
        Apply a simple cleanup sink by reducing concentration near the nearest grid cell.
        Returns approximate removed mass (kg) for bookkeeping.
        """
        import numpy as np

        sink_strength = float(np.clip(sink_strength, 0.0, 1.0))
        if sink_strength <= 0.0:
            return 0.0

        c = self.model.pollutant_fields[self.pollutant].get_concentration(self.pollutant)
        i0, j0, k0 = self._model_to_grid_ijk(self._world_to_model_xyz(world_xyz))

        # Convert radius to index window.
        ri = max(0, int(round(float(sink_radius_m) / float(self.model.grid.dx))))
        rj = max(0, int(round(float(sink_radius_m) / float(self.model.grid.dy))))
        rk = max(0, int(round(float(sink_radius_m) / float(self.model.grid.dz))))

        i1 = max(0, i0 - ri)
        i2 = min(self.model.grid.nx - 1, i0 + ri)
        j1 = max(0, j0 - rj)
        j2 = min(self.model.grid.ny - 1, j0 + rj)
        k1 = max(0, k0 - rk)
        k2 = min(self.model.grid.nz - 1, k0 + rk)

        before = c[i1 : i2 + 1, j1 : j2 + 1, k1 : k2 + 1].copy()
        c[i1 : i2 + 1, j1 : j2 + 1, k1 : k2 + 1] *= 1.0 - sink_strength
        self.model.pollutant_fields[self.pollutant].set_concentration(self.pollutant, c)

        removed_conc = before - c[i1 : i2 + 1, j1 : j2 + 1, k1 : k2 + 1]
        removed_mass = float(np.sum(removed_conc * self.model.grid.volumes[i1 : i2 + 1, j1 : j2 + 1, k1 : k2 + 1]))
        return removed_mass

    def mass_total(self) -> float:
        import numpy as np

        c = self.model.pollutant_fields[self.pollutant].get_concentration(self.pollutant)
        return float(np.sum(c * self.model.grid.volumes))

    def mass_leaked_world_xy(self, *, source_world_xy: tuple[float, float], leak_radius_m: float) -> float:
        """
        Approximate leaked mass as mass in cells whose (x,y) distance from the source exceeds leak_radius_m.
        """
        import numpy as np

        c = self.model.pollutant_fields[self.pollutant].get_concentration(self.pollutant)

        # Build model-frame XY grids.
        X = self.model.grid.X[:, :, 0]
        Y = self.model.grid.Y[:, :, 0]
        src_model_xy = self._world_to_model_xyz((source_world_xy[0], source_world_xy[1], self.center_world[2]))
        dx = X - float(src_model_xy[0])
        dy = Y - float(src_model_xy[1])
        far = (dx * dx + dy * dy) >= float(leak_radius_m) ** 2
        far3 = np.repeat(far[:, :, None], self.model.grid.nz, axis=2)
        return float(np.sum(c[far3] * self.model.grid.volumes[far3]))
