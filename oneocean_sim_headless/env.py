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
from .tasks import TaskConfig, TaskState, compute_success, required_n_agents, reset_task


@dataclass(frozen=True)
class EnvConfig:
    drift_cache_npz: str
    pollution_model: PollutionModelKind = "gaussian"
    dt_s: float = 1.0
    y_depth_range_m: tuple[float, float] = (2.0, 18.0)  # y is depth (positive down)
    tile_size_x_m: float = 600.0
    tile_size_z_m: float = 600.0
    land_mask_threshold: float = 0.5
    constraint_mode: Literal["off", "hard"] = "hard"
    bathy_mode: Literal["off", "hard"] = "off"
    seafloor_clearance_m: float = 1.0
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

    def _violates_constraints(self, p_xyz: np.ndarray) -> bool:
        if str(self.cfg.constraint_mode) == "off":
            return False
        x_m = float(p_xyz[0])
        y_depth_m = float(p_xyz[1])
        z_m = float(p_xyz[2])

        m = self.drift_field.sample_land_mask_xz(
            x_m=x_m,
            z_m=z_m,
            origin_lat=self.origin_lat,
            origin_lon=self.origin_lon,
        )
        if m is not None and float(m) >= float(self.cfg.land_mask_threshold):
            return True

        if str(self.cfg.bathy_mode) != "hard":
            return False

        elev = self.drift_field.sample_elevation_xz(
            x_m=x_m,
            z_m=z_m,
            origin_lat=self.origin_lat,
            origin_lon=self.origin_lon,
        )
        if elev is None or not np.isfinite(float(elev)):
            return True
        elev_m = float(elev)
        # Convention: elevation is seafloor height relative to sea surface (meters),
        # typically negative for underwater seabed.
        if elev_m >= 0.0:
            return True
        water_depth_m = -elev_m
        if (y_depth_m + float(self.cfg.seafloor_clearance_m)) > water_depth_m:
            return True
        return False

    def reset(self, *, task: TaskConfig, controller: ControllerConfig) -> dict[str, Any]:
        self.task_cfg = task
        self.controller_cfg = controller
        req_n = required_n_agents(task.kind)
        if req_n is not None and int(self.n_agents) != int(req_n):
            raise ValueError(f"Task {task.kind!r} requires n_agents={req_n}, got n_agents={self.n_agents}")
        self.task_state = reset_task(self.rng, self.bounds_xyz, task, n_agents=int(self.n_agents))

        if str(self.cfg.constraint_mode) != "off" and not bool(self.drift_info.has_land_mask):
            raise ValueError(
                "constraint_mode is enabled but drift cache is missing 'land_mask'. "
                "Re-export the drift cache with land_mask or set constraint_mode=off."
            )
        if str(self.cfg.bathy_mode) == "hard" and not bool(self.drift_info.has_elevation):
            raise ValueError(
                "bathy_mode=hard but drift cache is missing 'elevation'. "
                "Re-export the drift cache with elevation or set bathy_mode=off."
            )

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
                if not self._violates_constraints(p):
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
                if not self._violates_constraints(goal):
                    self.task_state.goal_xyz = goal
                    break
        elif task.kind == "pollution_containment_multiagent":
            self.task_state.goal_xyz = 0.5 * (lo + hi)
        elif task.kind == "formation_transit_multiagent":
            # Formation center goal is a reachable offset from the initial centroid.
            center = np.mean(self._positions, axis=0)
            dist = 120.0 if task.difficulty == "easy" else 200.0 if task.difficulty == "medium" else 280.0
            for _ in range(200):
                ang = float(self.rng.uniform(0, 2 * math.pi))
                off = np.array([math.cos(ang) * dist, 0.0, math.sin(ang) * dist], dtype=np.float64)
                goal = np.minimum(np.maximum(center + off, lo), hi)
                if not self._violates_constraints(goal):
                    self.task_state.goal_xyz = goal
                    break
        elif task.kind == "fish_herding_8uuv":
            # Deep corner target (XZ); keep depth near shallow for goal semantics.
            self.task_state.goal_xyz = np.array([hi[0] - 20.0, lo[1] + 0.8, hi[2] - 20.0], dtype=np.float64)
        elif task.kind == "area_scan_terrain_recon":
            # Precompute a lawnmower scan path over the scan grid.
            if self.task_state.scan_grid_origin_xz is not None and self.task_state.scan_grid_hw is not None:
                ox, oz = [float(x) for x in np.asarray(self.task_state.scan_grid_origin_xz, dtype=np.float64).reshape(2)]
                h, w = self.task_state.scan_grid_hw
                cell = float(task.scan_cell_size_m)
                pts = []
                y = float(np.mean(self._positions[:, 1]))
                for i in range(h):
                    xs = range(w) if (i % 2 == 0) else range(w - 1, -1, -1)
                    for j in xs:
                        x = ox + (float(j) + 0.5) * cell
                        z = oz + (float(i) + 0.5) * cell
                        pts.append([float(np.clip(x, lo[0], hi[0])), y, float(np.clip(z, lo[2], hi[2]))])
                self.task_state.waypoints_xyz = np.asarray(pts, dtype=np.float64)
                self.task_state.waypoint_index = 0
                if self.task_state.waypoints_xyz.size:
                    self.task_state.goal_xyz = np.asarray(self.task_state.waypoints_xyz[0], dtype=np.float64)
        elif task.kind == "depth_profile_tracking":
            # If waypoints exist, encode a depth profile into waypoint y (depth positive down).
            if self.task_state.waypoints_xyz is not None:
                wps = np.asarray(self.task_state.waypoints_xyz, dtype=np.float64)
                amp = 2.0 if task.difficulty == "easy" else 3.5 if task.difficulty == "medium" else 5.0
                base = float(np.clip(np.mean(self._positions[:, 1]), lo[1] + 0.5, hi[1] - 0.5))
                for i in range(wps.shape[0]):
                    wps[i, 1] = float(np.clip(base + amp * math.sin(2.0 * math.pi * (i / max(1, wps.shape[0] - 1))), lo[1] + 0.2, hi[1] - 0.2))
                self.task_state.waypoints_xyz = wps
                self.task_state.goal_xyz = wps[int(np.clip(self.task_state.waypoint_index, 0, wps.shape[0] - 1))].copy()

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
        step_i = int(round(self._t / max(1e-9, float(self.cfg.dt_s))))

        if self.task_cfg.kind == "pollution_containment_multiagent":
            # Estimate plume center from probes (weighted centroid). If probes are near-flat, keep last goal.
            w = np.maximum(probe, 0.0)
            s = float(np.sum(w))
            if s > 1e-12:
                est = np.sum(self._positions * (w[:, None] / s), axis=0)
                self.task_state.goal_xyz = est.astype(np.float64)
        # Default goal is broadcast; some tasks override with per-agent targets.
        goal: np.ndarray
        goal = np.asarray(self.task_state.goal_xyz, dtype=np.float64)
        if self.task_cfg.kind == "formation_transit_multiagent" and self.task_state.formation_offsets_xyz is not None:
            offsets = np.asarray(self.task_state.formation_offsets_xyz, dtype=np.float64).reshape(self.n_agents, 3)
            goal = goal.reshape(1, 3) + offsets
        elif self.task_cfg.kind == "surface_pollution_cleanup_multiagent" and self.task_state.cleanup_sources_xyz is not None and self.task_state.cleanup_done is not None:
            srcs = np.asarray(self.task_state.cleanup_sources_xyz, dtype=np.float64)
            done = np.asarray(self.task_state.cleanup_done, dtype=bool)
            if self.task_state.cleanup_assigned_source is None:
                self.task_state.cleanup_assigned_source = np.full((self.n_agents,), -1, dtype=np.int64)
            # Pick nearest unfinished per agent.
            for ai in range(self.n_agents):
                cur = int(self.task_state.cleanup_assigned_source[ai])
                if cur >= 0 and cur < int(done.size) and not bool(done[cur]):
                    continue
                if np.all(done):
                    self.task_state.cleanup_assigned_source[ai] = -1
                    continue
                cand = np.where(~done)[0]
                dists = np.linalg.norm(srcs[cand] - self._positions[ai][None, :], axis=1)
                self.task_state.cleanup_assigned_source[ai] = int(cand[int(np.argmin(dists))])
            goals = np.repeat(goal.reshape(1, 3), self.n_agents, axis=0)
            for ai in range(self.n_agents):
                si = int(self.task_state.cleanup_assigned_source[ai])
                if si >= 0:
                    goals[ai] = srcs[si]
            goal = goals
        elif self.task_cfg.kind == "underwater_pollution_lift_5uuv" and self.task_state.lift_barrel_xyz is not None:
            barrel = np.asarray(self.task_state.lift_barrel_xyz, dtype=np.float64).reshape(3)
            goals = np.repeat(barrel.reshape(1, 3), self.n_agents, axis=0)
            # Side attachments for the first 4 agents; the 5th aims from below.
            r = 5.0
            offsets = [
                np.array([+r, 0.0, 0.0], dtype=np.float64),
                np.array([-r, 0.0, 0.0], dtype=np.float64),
                np.array([0.0, 0.0, +r], dtype=np.float64),
                np.array([0.0, 0.0, -r], dtype=np.float64),
                np.array([0.0, +r, 0.0], dtype=np.float64),
            ]
            for i in range(min(self.n_agents, len(offsets))):
                goals[i] = barrel + offsets[i]
            # During lift phases, bias upward to approach surface.
            if str(self.task_state.lift_phase) in {"lift_off", "join5", "to_surface"}:
                goals[:, 1] = max(float(self.bounds_xyz[0][1]) + 0.6, float(barrel[1]) - 0.9)
            goal = goals
        elif self.task_cfg.kind == "fish_herding_8uuv" and self.task_state.fish_xyz is not None:
            fish = np.asarray(self.task_state.fish_xyz, dtype=np.float64).reshape(-1, 3)
            cen = np.mean(fish, axis=0)
            tgt = np.asarray(self.task_state.goal_xyz, dtype=np.float64).reshape(3)
            push = tgt - cen
            push[1] = 0.0
            nrm = float(np.linalg.norm(push))
            if not np.isfinite(nrm) or nrm < 1e-9:
                push = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            else:
                push = push / nrm
            # Place agents on a ring "behind" the fish (opposite push direction).
            back = -push
            ring_center = cen + 12.0 * back
            rr = 10.0
            goals = np.zeros((self.n_agents, 3), dtype=np.float64)
            for i in range(self.n_agents):
                ang = 2.0 * math.pi * (i / max(1, self.n_agents))
                off = np.array([rr * math.cos(ang), 0.0, rr * math.sin(ang)], dtype=np.float64)
                goals[i] = ring_center + off
                goals[i, 1] = float(cen[1])
            goal = goals
        elif self.task_cfg.kind == "area_scan_terrain_recon" and self.task_state.waypoints_xyz is not None:
            wps = np.asarray(self.task_state.waypoints_xyz, dtype=np.float64)
            i = int(np.clip(int(self.task_state.waypoint_index), 0, wps.shape[0] - 1))
            # Advance lawnmower target when agent0 reaches the current cell center.
            if i < (wps.shape[0] - 1):
                if float(np.linalg.norm(self._positions[0] - wps[i])) <= float(self.task_cfg.success_radius_m):
                    self.task_state.waypoint_index = i + 1
                    i = int(self.task_state.waypoint_index)
            goal = wps[i]
        elif self.task_cfg.kind == "pipeline_inspection_leak_detection" and self.task_state.waypoints_xyz is not None:
            wps = np.asarray(self.task_state.waypoints_xyz, dtype=np.float64)
            i = int(np.clip(int(self.task_state.waypoint_index), 0, wps.shape[0] - 1))
            if i < (wps.shape[0] - 1):
                if float(np.linalg.norm(self._positions[0] - wps[i])) <= float(self.task_cfg.success_radius_m):
                    self.task_state.waypoint_index = i + 1
                    i = int(self.task_state.waypoint_index)
            self.task_state.goal_xyz = wps[i].copy()
            goal = self.task_state.goal_xyz

        act = compute_actions(
            self.controller_cfg,
            step_index=step_i,
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
            if self._violates_constraints(new_p):
                # Hard constraint: reject invalid states (land/NoData/touchdown).
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
            elev_arr[i] = float(np.nan if elev is None else elev)
            m = self.drift_field.sample_land_mask_xz(
                x_m=float(self._positions[i, 0]),
                z_m=float(self._positions[i, 2]),
                origin_lat=self.origin_lat,
                origin_lon=self.origin_lon,
            )
            mask_arr[i] = float(np.nan if m is None else m)

            self._energy += float(np.dot(act[i], act[i])) * float(self.cfg.dt_s)

        # Update pollution field.
        if hasattr(self.pollution, "advect_center"):
            # Advect the Gaussian plume center using the current at its center.
            cpos = getattr(self.pollution, "center_xyz", None)
            if cpos is not None:
                cx, cz = self._sample_current(float(cpos[0]), float(cpos[2]))
                self.pollution.advect_center(np.array([cx, cz], dtype=np.float64), float(self.cfg.dt_s))
        self.pollution.step(float(self.cfg.dt_s))
        if hasattr(self.pollution, "apply_agent_sink") and self.task_cfg.kind in ("pollution_containment_multiagent",):
            # both Gaussian and OCPNet expose this method (signature differs; use kwargs best-effort)
            try:
                self.pollution.apply_agent_sink(self._positions, dt_s=float(self.cfg.dt_s))
            except TypeError:
                self.pollution.apply_agent_sink(self._positions)

        # Update task-side semantics (fish/barrel) after the physics step.
        if self.task_cfg.kind == "fish_herding_8uuv" and self.task_state.fish_xyz is not None:
            fish = np.asarray(self.task_state.fish_xyz, dtype=np.float64).reshape(-1, 3)
            # Drift fish with a small current at their centroid and repulsion from agents.
            cen = np.mean(fish, axis=0)
            cx, cz = self._sample_current(float(cen[0]), float(cen[2]))
            drift = np.array([cx, 0.0, cz], dtype=np.float64) * float(self.cfg.dt_s)
            noise = self.rng.normal(scale=0.35, size=fish.shape).astype(np.float64)
            noise[:, 1] = 0.0
            fish = fish + 0.25 * drift[None, :] + 0.15 * noise
            # Repel from agents (simple herding proxy).
            for p in self._positions:
                d = fish - p[None, :]
                d[:, 1] = 0.0
                r2 = np.sum(d[:, [0, 2]] ** 2, axis=1)
                w = np.exp(-r2 / (2.0 * (18.0**2)))
                fish[:, 0] += 0.6 * w * (d[:, 0] / (np.sqrt(r2) + 1e-6))
                fish[:, 2] += 0.6 * w * (d[:, 2] / (np.sqrt(r2) + 1e-6))
            lo, hi = self.bounds_xyz
            fish = np.minimum(np.maximum(fish, lo[None, :]), hi[None, :])
            self.task_state.fish_xyz = fish

        if self.task_cfg.kind == "underwater_pollution_lift_5uuv" and self.task_state.lift_barrel_xyz is not None and self.task_state.lift_attached is not None:
            attached = np.asarray(self.task_state.lift_attached, dtype=bool).reshape(self.n_agents)
            if np.any(attached):
                barrel = np.asarray(self.task_state.lift_barrel_xyz, dtype=np.float64).reshape(3)
                mean = np.mean(self._positions[attached], axis=0)
                barrel[0] = float(mean[0])
                barrel[2] = float(mean[2])
                barrel[1] = float(mean[1])
                self.task_state.lift_barrel_xyz = barrel

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

        # Optional semantic stream (downsample to reduce file size).
        if step_i % 5 == 0:
            payload: dict[str, Any] = {"task": str(self.task_cfg.kind)}
            if self.task_cfg.kind == "surface_pollution_cleanup_multiagent" and self.task_state.cleanup_sources_xyz is not None and self.task_state.cleanup_done is not None:
                payload["cleanup_sources_xyz"] = np.asarray(self.task_state.cleanup_sources_xyz, dtype=np.float64).tolist()
                payload["cleanup_done"] = np.asarray(self.task_state.cleanup_done, dtype=bool).astype(int).tolist()
                if self.task_state.cleanup_assigned_source is not None:
                    payload["cleanup_assigned_source"] = np.asarray(self.task_state.cleanup_assigned_source, dtype=np.int64).tolist()
            if self.task_cfg.kind == "underwater_pollution_lift_5uuv" and self.task_state.lift_barrel_xyz is not None:
                payload["barrel_xyz"] = np.asarray(self.task_state.lift_barrel_xyz, dtype=np.float64).tolist()
                payload["lift_phase"] = str(self.task_state.lift_phase)
                if self.task_state.lift_attached is not None:
                    payload["lift_attached"] = np.asarray(self.task_state.lift_attached, dtype=bool).astype(int).tolist()
            if self.task_cfg.kind == "fish_herding_8uuv" and self.task_state.fish_xyz is not None:
                fish = np.asarray(self.task_state.fish_xyz, dtype=np.float64).reshape(-1, 3)
                payload["fish_centroid_xyz"] = np.mean(fish, axis=0).tolist()
                payload["fish_stage"] = int(self.task_state.fish_stage)
            if self.task_cfg.kind == "pipeline_inspection_leak_detection" and self.task_state.leak_xyz is not None and self.task_state.leak_detected is not None:
                payload["leak_xyz"] = np.asarray(self.task_state.leak_xyz, dtype=np.float64).tolist()
                payload["leak_detected"] = np.asarray(self.task_state.leak_detected, dtype=bool).astype(int).tolist()
            self.rec.write_semantics(self._t, payload)

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
            dt_s=float(self.cfg.dt_s),
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
