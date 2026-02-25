from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from oneocean_sim_habitat.drift import CachedDriftField
from oneocean_sim_habitat.drift import METERS_PER_DEG_LAT, meters_per_deg_lon

from .controllers import ControllerConfig, compute_actions
from .drift_cache import DriftCacheInfo, load_drift_cache
from .pollution import PollutionModelKind, build_pollution_field
from .recorder import HeadlessRecorder, RecorderConfig
from .tasks import TaskConfig, TaskState, compute_success, reset_task


@dataclass(frozen=True)
class EnvConfig:
    drift_cache_npz: str
    pollution_model: PollutionModelKind = "gaussian"
    dt_s: float = 1.0
    y_depth_range_m: tuple[float, float] = (2.0, 18.0)  # y is depth (positive down)
    tile_size_x_m: float = 600.0
    tile_size_z_m: float = 600.0
    land_mask_threshold: float = 0.5
    max_speed_mps: float = 1.2
    current_gain: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class HeadlessOceanEnv:
    def __init__(self, cfg: EnvConfig, *, out_dir: str | Path, seed: int, n_agents: int) -> None:
        self.cfg = cfg
        self.out_dir = Path(out_dir).expanduser().resolve()
        self.seed = int(seed)
        self.n_agents = int(n_agents)
        self.rng = np.random.default_rng(self.seed)

        self.drift_field, self.drift_info = load_drift_cache(cfg.drift_cache_npz)
        meta_json = self.drift_info.meta_json or {}
        self.dataset_time_index: int | None = int(meta_json["time_index"]) if "time_index" in meta_json else None
        self.dataset_depth_index: int | None = int(meta_json["depth_index"]) if "depth_index" in meta_json else None

        # Define a manageable simulation tile (meters) over the cached lat/lon grid.
        lat_min = float(np.min(self.drift_field.latitude))
        lat_max = float(np.max(self.drift_field.latitude))
        lon_min = float(np.min(self.drift_field.longitude))
        lon_max = float(np.max(self.drift_field.longitude))
        lat_mid = 0.5 * (lat_min + lat_max)
        max_x = (lon_max - lon_min) * meters_per_deg_lon(lat_mid)
        max_z = (lat_max - lat_min) * METERS_PER_DEG_LAT
        self.tile_size_x_m = float(np.clip(float(cfg.tile_size_x_m), 10.0, max(10.0, float(max_x))))
        self.tile_size_z_m = float(np.clip(float(cfg.tile_size_z_m), 10.0, max(10.0, float(max_z))))
        self.bounds_xyz = (
            np.array([0.0, float(cfg.y_depth_range_m[0]), 0.0], dtype=np.float64),
            np.array([self.tile_size_x_m, float(cfg.y_depth_range_m[1]), self.tile_size_z_m], dtype=np.float64),
        )

        # Tile origin on the dataset grid (chosen per-episode in reset; recorded in run_meta).
        self.origin_lat = lat_min
        self.origin_lon = lon_min
        self._dataset_lat_min = lat_min
        self._dataset_lat_max = lat_max
        self._dataset_lon_min = lon_min
        self._dataset_lon_max = lon_max

        self._positions = np.zeros((self.n_agents, 3), dtype=np.float64)
        self._yaws = np.zeros((self.n_agents,), dtype=np.float64)
        self._t = 0.0
        self._energy = 0.0
        self._constraint_violations = 0
        self._time_to_success_s: float | None = None

        # pollution model expects access to drift payload arrays for OCPNet velocity mapping.
        drift_payload = {
            "latitude": self.drift_field.latitude,
            "longitude": self.drift_field.longitude,
            "u": self.drift_field.u,
            "v": self.drift_field.v,
            "domain_size_m": [
                float(self.bounds_xyz[1][0] - self.bounds_xyz[0][0]),
                float(self.bounds_xyz[1][2] - self.bounds_xyz[0][2]),
                float(self.bounds_xyz[1][1] - self.bounds_xyz[0][1]),
            ],
        }
        self.pollution, self.pollution_meta = build_pollution_field(
            cfg.pollution_model,
            rng=self.rng,
            bounds_xyz=self.bounds_xyz,
            output_dir=self.out_dir,
            drift_payload=drift_payload,
        )

        self.task_cfg: TaskConfig | None = None
        self.task_state: TaskState | None = None
        self.controller_cfg: ControllerConfig | None = None

        self.rec = HeadlessRecorder(self.out_dir, n_agents=self.n_agents, config=RecorderConfig())
        self._initial_pollution_mass: float | None = None

    @property
    def positions_xyz(self) -> np.ndarray:
        return self._positions.copy()

    @property
    def energy_proxy(self) -> float:
        return float(self._energy)

    @property
    def constraint_violations(self) -> int:
        return int(self._constraint_violations)

    @property
    def time_to_success_s(self) -> float | None:
        return None if self._time_to_success_s is None else float(self._time_to_success_s)

    def _sample_current(self, x_m: float, z_m: float) -> tuple[float, float]:
        drift_x, drift_z = self.drift_field.sample_xz(
            x_m=x_m,
            z_m=z_m,
            origin_lat=self.origin_lat,
            origin_lon=self.origin_lon,
        )
        return float(drift_x) * float(self.cfg.current_gain), float(drift_z) * float(self.cfg.current_gain)

    def _is_blocked(self, x_m: float, z_m: float) -> bool:
        return self.drift_field.is_blocked_xz(
            x_m=x_m,
            z_m=z_m,
            origin_lat=self.origin_lat,
            origin_lon=self.origin_lon,
            land_mask_threshold=float(self.cfg.land_mask_threshold),
            elevation_threshold=None,
        )

    def reset(self, *, task: TaskConfig, controller: ControllerConfig) -> dict[str, Any]:
        self.task_cfg = task
        self.controller_cfg = controller
        self.task_state = reset_task(self.rng, self.bounds_xyz, task)

        # Pick a dataset tile origin such that the sim tile maps inside the cached lat/lon extent.
        lat_span_deg = float(self.tile_size_z_m) / METERS_PER_DEG_LAT
        lon_span_deg = float(self.tile_size_x_m) / meters_per_deg_lon(self._dataset_lat_min)
        lat0_min = self._dataset_lat_min
        lat0_max = self._dataset_lat_max - lat_span_deg
        if lat0_max < lat0_min:
            lat0_max = lat0_min
        self.origin_lat = float(self.rng.uniform(lat0_min, lat0_max))
        lon0_min = self._dataset_lon_min
        lon0_max = self._dataset_lon_max - lon_span_deg
        if lon0_max < lon0_min:
            lon0_max = lon0_min
        self.origin_lon = float(self.rng.uniform(lon0_min, lon0_max))

        lo, hi = self.bounds_xyz
        for i in range(self.n_agents):
            for _ in range(200):
                p = self.rng.uniform(lo, hi).astype(np.float64)
                if not self._is_blocked(float(p[0]), float(p[2])):
                    self._positions[i] = p
                    break
            self._yaws[i] = float(self.rng.uniform(-math.pi, math.pi))

        # Task-specific goal/anchor choices.
        if task.kind == "station_keeping":
            self.task_state.goal_xyz = np.mean(self._positions, axis=0)
        elif task.kind == "go_to_goal_current":
            # Choose a reachable goal offset from the initial centroid (task-difficulty dependent).
            center = np.mean(self._positions, axis=0)
            dist = 90.0 if task.difficulty == "easy" else 160.0 if task.difficulty == "medium" else 240.0
            for _ in range(200):
                ang = float(self.rng.uniform(0, 2 * math.pi))
                off = np.array([math.cos(ang) * dist, 0.0, math.sin(ang) * dist], dtype=np.float64)
                goal = np.minimum(np.maximum(center + off, lo), hi)
                if not self._is_blocked(float(goal[0]), float(goal[2])):
                    self.task_state.goal_xyz = goal
                    break
        elif task.kind == "pollution_containment_multiagent":
            self.task_state.goal_xyz = 0.5 * (lo + hi)

        self._t = 0.0
        self._energy = 0.0
        self._constraint_violations = 0
        self._time_to_success_s = None
        self.pollution_meta = self.pollution.reset(self.rng, bounds_xyz=self.bounds_xyz)

        # Track an initial mass proxy (for containment task success definition).
        self._initial_pollution_mass = None
        if hasattr(self.pollution, "model"):
            try:
                conc = self.pollution.model.pollutant_fields[self.pollution.pollutant].get_concentration(self.pollution.pollutant)
                self._initial_pollution_mass = float(np.sum(conc))
            except Exception:
                self._initial_pollution_mass = None

        meta = {
            "seed": int(self.seed),
            "n_agents": int(self.n_agents),
            "env_config": self.cfg.to_dict(),
            "task": task.to_dict(),
            "controller": controller.to_dict(),
            "drift_cache": self.drift_info.to_dict(),
            "pollution": self.pollution_meta,
            "tile": {
                "tile_size_x_m": float(self.tile_size_x_m),
                "tile_size_z_m": float(self.tile_size_z_m),
                "origin_lat": float(self.origin_lat),
                "origin_lon": float(self.origin_lon),
                "projection_note": "x=east meters, z=north meters; lat/lon from tangent-plane approx around origin_lat/lon (nearest-neighbor sampling on cached grid).",
            },
            "bounds_xyz": {"lo": lo.tolist(), "hi": hi.tolist()},
        }
        self.rec.write_run_meta(meta)
        return meta

    def step(self) -> tuple[bool, dict[str, Any]]:
        assert self.task_cfg is not None and self.task_state is not None and self.controller_cfg is not None

        # Observations for controller.
        probe = np.array([float(self.pollution.sample(self._positions[i])) for i in range(self.n_agents)], dtype=np.float64)
        goal = self.task_state.goal_xyz

        act = compute_actions(
            self.controller_cfg,
            step_index=int(round(self._t / max(1e-9, float(self.cfg.dt_s)))),
            positions_xyz=self._positions,
            goal_xyz=goal,
            pollution_probe=probe,
            rng=self.rng,
        )
        # Clip actions to max speed.
        for i in range(self.n_agents):
            sp = float(np.linalg.norm(act[i]))
            if sp > float(self.cfg.max_speed_mps):
                act[i] = act[i] * (float(self.cfg.max_speed_mps) / sp)

        # Currents and state update.
        currents = np.zeros((self.n_agents, 3), dtype=np.float64)
        lat_arr = np.zeros((self.n_agents,), dtype=np.float64)
        lon_arr = np.zeros((self.n_agents,), dtype=np.float64)
        elev_arr = np.zeros((self.n_agents,), dtype=np.float64)
        mask_arr = np.zeros((self.n_agents,), dtype=np.float64)
        lo, hi = self.bounds_xyz
        for i in range(self.n_agents):
            cx, cz = self._sample_current(float(self._positions[i, 0]), float(self._positions[i, 2]))
            currents[i] = np.array([cx, 0.0, cz], dtype=np.float64)
            new_p = self._positions[i] + (act[i] + currents[i]) * float(self.cfg.dt_s)
            new_p = np.minimum(np.maximum(new_p, lo), hi)
            if self._is_blocked(float(new_p[0]), float(new_p[2])):
                # simple rejection: stay put if blocked
                new_p = self._positions[i]
                self._constraint_violations += 1
            self._positions[i] = new_p
            if float(np.linalg.norm(act[i])) > 1e-6:
                self._yaws[i] = float(math.atan2(act[i, 2], act[i, 0]))

            # extra audit-friendly observables
            lat = float(self.origin_lat + (float(self._positions[i, 2]) / METERS_PER_DEG_LAT))
            lon = float(self.origin_lon + (float(self._positions[i, 0]) / meters_per_deg_lon(self.origin_lat)))
            lat_arr[i] = float(lat)
            lon_arr[i] = float(lon)
            elev = self.drift_field.sample_elevation_xz(
                x_m=float(self._positions[i, 0]),
                z_m=float(self._positions[i, 2]),
                origin_lat=self.origin_lat,
                origin_lon=self.origin_lon,
            )
            elev_arr[i] = float(0.0 if elev is None else elev)
            m = self.drift_field.sample_land_mask_xz(
                x_m=float(self._positions[i, 0]),
                z_m=float(self._positions[i, 2]),
                origin_lat=self.origin_lat,
                origin_lon=self.origin_lon,
            )
            mask_arr[i] = float(0.0 if m is None else m)

            self._energy += float(np.dot(act[i], act[i])) * float(self.cfg.dt_s)

        # Update pollution field.
        if hasattr(self.pollution, "advect_center"):
            # Advect the Gaussian plume center using the current at its center.
            cpos = getattr(self.pollution, "center_xyz", None)
            if cpos is not None:
                cx, cz = self._sample_current(float(cpos[0]), float(cpos[2]))
                self.pollution.advect_center(np.array([cx, cz], dtype=np.float64), float(self.cfg.dt_s))
        self.pollution.step(float(self.cfg.dt_s))
        if hasattr(self.pollution, "apply_agent_sink") and self.task_cfg.kind == "pollution_containment_multiagent":
            # both Gaussian and OCPNet expose this method (signature differs; use kwargs best-effort)
            try:
                self.pollution.apply_agent_sink(self._positions, dt_s=float(self.cfg.dt_s))
            except TypeError:
                self.pollution.apply_agent_sink(self._positions)

        # Recompute probes for logging/metrics.
        probe2 = np.array([float(self.pollution.sample(self._positions[i])) for i in range(self.n_agents)], dtype=np.float64)
        self.rec.step(
            self._t,
            positions_xyz=self._positions,
            yaws_rad=self._yaws,
            actions_xyz=act,
            currents_xyz=currents,
            pollution_probe=probe2,
            latitude=lat_arr,
            longitude=lon_arr,
            elevation=elev_arr,
            land_mask=mask_arr,
        )
        self.rec.write_environment_sample(self._t, dataset_time_index=self.dataset_time_index, dataset_depth_index=self.dataset_depth_index)

        # Success checks.
        pollution_src = np.array(self.pollution_meta.get("source_xyz"), dtype=np.float64) if "source_xyz" in self.pollution_meta else None
        mass_frac = None
        if hasattr(self.pollution, "mass_fraction"):
            try:
                mass_frac = float(self.pollution.mass_fraction())
            except Exception:
                mass_frac = None
        elif hasattr(self.pollution, "model") and self._initial_pollution_mass is not None and self._initial_pollution_mass > 1e-12:
            conc = self.pollution.model.pollutant_fields[self.pollution.pollutant].get_concentration(self.pollution.pollutant)
            mass_frac = float(np.sum(conc) / self._initial_pollution_mass)

        success, extra = compute_success(
            self.task_cfg,
            step_index=int(round(self._t / max(1e-9, float(self.cfg.dt_s)))),
            positions_xyz=self._positions,
            task_state=self.task_state,
            pollution_source_xyz=pollution_src,
            pollution_total_mass=mass_frac,
        )
        info = {
            "t": float(self._t),
            "probe_max": float(np.max(probe2)),
            "probe_mean": float(np.mean(probe2)),
            "energy_proxy": float(self._energy),
            "constraint_violations": int(self._constraint_violations),
            **extra,
        }
        self._t += float(self.cfg.dt_s)
        # Termination by time horizon.
        done = bool(success)
        if not done and int(round(self._t / max(1e-9, float(self.cfg.dt_s)))) >= int(self.task_cfg.max_steps):
            done = True
            info["time_limit"] = True
        info["success"] = bool(success) and not bool(info.get("time_limit", False))
        if info["success"] and self._time_to_success_s is None:
            self._time_to_success_s = float(self._t)
        return bool(done), info

    def close(self) -> None:
        self.rec.close()
