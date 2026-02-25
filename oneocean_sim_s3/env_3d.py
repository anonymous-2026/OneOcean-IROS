from __future__ import annotations

from dataclasses import asdict, dataclass
from math import cos, pi, sin
from pathlib import Path
from typing import Optional

import numpy as np
import sapien.core as sapien

from oneocean_sim.data import CurrentSampler

from .software_renderer import (
    CameraConfig,
    CameraPose,
    RenderMesh,
    RenderSphere,
    RenderVehicle,
    render_scene,
    yaw_from_velocity,
)
from .world_3d import TerrainMesh, TerrainSpec, build_obstacles, build_terrain_mesh, xy_to_latlon
from .external_scenes.polyhaven_models import ensure_polyhaven_model_obj


_GLOBAL_ENGINE: Optional[sapien.Engine] = None


def _get_engine() -> sapien.Engine:
    global _GLOBAL_ENGINE
    if _GLOBAL_ENGINE is None:
        _GLOBAL_ENGINE = sapien.Engine()
    return _GLOBAL_ENGINE


def _yaw_to_quat(yaw_rad: float) -> list[float]:
    half = 0.5 * float(yaw_rad)
    return [cos(half), 0.0, 0.0, sin(half)]


def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm = np.where(norm > 1e-9, norm, 1.0)
    return (n / norm).astype(np.float64)


@dataclass(frozen=True)
class S3WorldConfig3D:
    dt_sec: float = 0.1
    max_steps: int = 240
    max_rel_speed_mps: float = 1.6
    velocity_tau_sec: float = 0.6
    terrain_grid_size: int = 33
    terrain_z_min_m: float = -30.0
    terrain_z_max_m: float = -5.0
    terrain_xy_scale: float = 0.02
    obstacle_count: int = 14
    depth_clearance_m: float = 4.0
    current_scale: float = 1.0
    terminate_on_collision: bool = True
    terminate_on_invalid_region: bool = True


@dataclass
class WorldArtifacts:
    terrain_obj: Optional[str]
    external_scene: Optional[dict[str, object]] = None


class OceanWorldS3_3D:
    def __init__(
        self,
        *,
        sampler: CurrentSampler,
        dataset_path: Path,
        time_index: int,
        depth_index: int,
        include_tides: bool,
        seed: int,
        agents: int,
        config: S3WorldConfig3D,
        output_dir: Path,
        external_scene: Optional[str] = None,
        external_scene_resolution: str = "1k",
        external_scene_max_faces: int = 12000,
    ) -> None:
        if agents < 1:
            raise ValueError("agents must be >= 1")
        self.sampler = sampler
        self.dataset_path = dataset_path
        self.time_index = int(time_index)
        self.depth_index = int(depth_index)
        self.include_tides = bool(include_tides)
        self.config = config
        self.output_dir = output_dir
        self.rng = np.random.default_rng(int(seed))

        self.engine = _get_engine()
        scene_cfg = sapien.SceneConfig()
        scene_cfg.gravity = [0.0, 0.0, 0.0]
        self.scene = self.engine.create_scene(scene_cfg)
        self.scene.set_timestep(float(config.dt_sec))

        # --- build terrain mesh (dataset-grounded) ---
        lat_values = np.asarray(self.sampler.ds.latitude_values, dtype=np.float64)
        lon_values = np.asarray(self.sampler.ds.longitude_values, dtype=np.float64)
        if "elevation" not in self.sampler.ds.file:
            raise ValueError("Dataset missing required variable: elevation")
        elev = np.asarray(self.sampler.ds.file["elevation"][:, :], dtype=np.float64)
        land_mask = (
            np.asarray(self.sampler.ds.file["land_mask"][:, :], dtype=np.float64)
            if "land_mask" in self.sampler.ds.file
            else None
        )

        center_lat, center_lon = self.sampler.center_latlon()
        terrain_spec = TerrainSpec(
            grid_size=int(config.terrain_grid_size),
            z_min_m=float(config.terrain_z_min_m),
            z_max_m=float(config.terrain_z_max_m),
            xy_scale=float(config.terrain_xy_scale),
        )
        terrain_obj_path = output_dir / "assets" / "terrain.obj"
        self.terrain = build_terrain_mesh(
            latitude_values=lat_values,
            longitude_values=lon_values,
            elevation_grid=elev,
            land_mask=land_mask,
            center_lat=float(center_lat),
            center_lon=float(center_lon),
            spec=terrain_spec,
            output_obj=terrain_obj_path,
        )
        self.terrain_face_normals = _compute_face_normals(self.terrain.vertices_m, self.terrain.faces)

        self.artifacts = WorldArtifacts(terrain_obj=str(terrain_obj_path), external_scene=None)

        # physics collision terrain
        builder = self.scene.create_actor_builder()
        builder.add_nonconvex_collision_from_file(str(terrain_obj_path))
        self.terrain_actor = builder.build_static(name="seafloor")

        # --- optional external scene mesh (third-party assets, cached locally) ---
        self.external_meshes: list[RenderMesh] = []
        self.external_actors: list[sapien.Actor] = []
        self._external_scene = str(external_scene) if external_scene else None
        if external_scene and external_scene != "none":
            # Currently supported: polyhaven:<asset_id>
            if not external_scene.startswith("polyhaven:"):
                raise ValueError(f"Unsupported external_scene: {external_scene}")
            asset_id = external_scene.split(":", 1)[1].strip()
            if not asset_id:
                raise ValueError(f"Invalid external_scene: {external_scene}")

            assets = ensure_polyhaven_model_obj(
                asset_id=asset_id,
                cache_root=Path("runs") / "_cache",
                resolution=str(external_scene_resolution),
                max_faces=int(external_scene_max_faces),
                # PolyHaven models are often large; shrink a bit to fit our small terrain patch.
                scale=0.12,
                center=True,
            )
            self.artifacts.external_scene = {
                "provider": "polyhaven",
                "asset_id": str(asset_id),
                "resolution": str(external_scene_resolution),
                "max_faces": int(external_scene_max_faces),
                "obj_path": str(assets.obj_path),
                "sources_md": str(assets.sources_md),
            }

            self.external_meshes = [
                RenderMesh(
                    mesh_path=str(assets.obj_path),
                    position_m=(0.0, 0.0, 0.0),
                    yaw_rad=0.0,
                    scale_m=1.0,
                    color_bgr=(65, 90, 115),
                )
            ]
            # Collision: optional and coarse; keep but ensure it is always underwater and visible.
            b = self.scene.create_actor_builder()
            b.add_nonconvex_collision_from_file(str(assets.obj_path))
            actor = b.build_static(name="external_scene_asset")
            actor.set_pose(sapien.Pose([0.0, 0.0, float(self.config.terrain_z_min_m + 4.0)]))
            self.external_actors = [actor]

        # --- task-specific fields ---
        self.goal_xy_m: tuple[float, float] = (30.0, 0.0)
        self.goal_tolerance_m: float = 6.0
        self.formation_offset_m: tuple[float, float] = (0.0, 8.0)
        self.formation_tolerance_m: float = 4.0

        # obstacles depend on the task goal; set default then rebuild on reset_task
        self.obstacles: list[RenderSphere] = []
        self.obstacle_actors: list[sapien.Actor] = []

        self.agents = int(agents)
        self.vehicle_actors: list[sapien.Actor] = []
        self.vehicle_colors_bgr: list[tuple[int, int, int]] = [(40, 200, 255), (80, 255, 120), (255, 180, 80)]
        repo_root = Path(__file__).resolve().parents[1]
        mesh_dir = repo_root / "oneocean_sim_habitat" / "assets" / "meshes"
        self.vehicle_mesh_paths: list[str | None] = [
            str(mesh_dir / "uuv_blue.obj"),
            str(mesh_dir / "uuv_green.obj"),
            str(mesh_dir / "uuv_red.obj"),
        ]
        self._build_vehicles()

        self._time_sec = 0.0
        self._particles_xyz_m = np.zeros((0, 3), dtype=np.float64)
        self._particles_vel_mps = np.zeros((0, 3), dtype=np.float64)

    def _build_vehicles(self) -> None:
        self.vehicle_actors = []
        for idx in range(self.agents):
            builder = self.scene.create_actor_builder()
            builder.add_capsule_collision(radius=0.45, half_length=0.8)
            vehicle = builder.build(name=f"uuv_{idx}")
            vehicle.set_damping(1.4, 1.4)
            vehicle.set_pose(sapien.Pose([0.0, 0.0, -12.0]))
            vehicle.set_velocity([0.0, 0.0, 0.0])
            vehicle.set_angular_velocity([0.0, 0.0, 0.0])
            self.vehicle_actors.append(vehicle)

    def _reset_particles(self) -> None:
        x_min, x_max, y_min, y_max = self.terrain.bounds_xy_m
        z_min = float(self.config.terrain_z_min_m)
        z_max = 2.0
        count = int(np.clip(120 * float(self.config.terrain_grid_size), 800, 2400))
        self._particles_xyz_m = np.column_stack(
            [
                self.rng.uniform(x_min, x_max, size=count),
                self.rng.uniform(y_min, y_max, size=count),
                self.rng.uniform(z_min, z_max, size=count),
            ]
        ).astype(np.float64)
        vel = self.rng.normal(loc=0.0, scale=0.12, size=(count, 3)).astype(np.float64)
        vel[:, 2] = self.rng.normal(loc=0.02, scale=0.03, size=(count,)).astype(np.float64)
        self._particles_vel_mps = vel

    def _clear_obstacles(self) -> None:
        for actor in self.obstacle_actors:
            self.scene.remove_actor(actor)
        self.obstacle_actors = []
        self.obstacles = []

    def reset_task(
        self,
        *,
        task: str,
        seed: int,
        goal_distance_m: float,
        goal_bearing_deg: Optional[float],
        goal_tolerance_m: float,
        formation_offset_m: tuple[float, float],
        formation_tolerance_m: float,
    ) -> None:
        self.rng = np.random.default_rng(int(seed))
        if goal_bearing_deg is None:
            bearing = float(self.rng.uniform(0.0, 360.0))
        else:
            bearing = float(goal_bearing_deg)
        heading = bearing * pi / 180.0
        gx = float(goal_distance_m * cos(heading))
        gy = float(goal_distance_m * sin(heading))
        x_min, x_max, y_min, y_max = self.terrain.bounds_xy_m
        margin = float(self.terrain.x_coords_m[1] - self.terrain.x_coords_m[0]) if len(self.terrain.x_coords_m) > 1 else 8.0
        gx = float(np.clip(gx, x_min + margin, x_max - margin))
        gy = float(np.clip(gy, y_min + margin, y_max - margin))
        self.goal_xy_m = (gx, gy)
        self.goal_tolerance_m = float(goal_tolerance_m)
        self.formation_offset_m = (float(formation_offset_m[0]), float(formation_offset_m[1]))
        self.formation_tolerance_m = float(formation_tolerance_m)

        self._clear_obstacles()
        obstacle_specs = build_obstacles(
            rng=self.rng,
            terrain=self.terrain,
            count=int(self.config.obstacle_count),
            start_xy=(0.0, 0.0),
            goal_xy=self.goal_xy_m,
        )
        self.obstacles = []
        for spec in obstacle_specs:
            base_center = np.asarray(spec.position_m, dtype=np.float64)
            base_radius = float(spec.radius_m)
            parts = int(self.rng.integers(3, 6))
            for _ in range(parts):
                frac = float(self.rng.uniform(0.45, 0.9))
                radius = base_radius * frac
                dx, dy = self.rng.normal(loc=0.0, scale=0.42 * base_radius, size=(2,)).astype(np.float64)
                dz = float(self.rng.uniform(-0.25 * base_radius, 0.35 * base_radius))
                center = (float(base_center[0] + dx), float(base_center[1] + dy), float(base_center[2] + dz))
                self.obstacles.append(RenderSphere(center_m=center, radius_m=float(radius), color_bgr=(70, 85, 95)))

                builder = self.scene.create_actor_builder()
                builder.add_sphere_collision(radius=float(radius))
                actor = builder.build_static(name="rock_part")
                actor.set_pose(sapien.Pose(list(center)))
                self.obstacle_actors.append(actor)

        start_positions = []
        if task == "reef_navigation":
            start_positions = [(0.0, 0.0)]
        elif task == "formation_navigation":
            start_positions = [(0.0, -0.5 * self.formation_offset_m[1]), (0.0, +0.5 * self.formation_offset_m[1])]
        else:
            raise ValueError(f"Unknown task: {task}")

        for idx, actor in enumerate(self.vehicle_actors):
            if idx >= len(start_positions):
                sx, sy = start_positions[0]
            else:
                sx, sy = start_positions[idx]
            floor_z = float(self.terrain.height_at_xy(float(sx), float(sy)))
            z = float(floor_z + self.config.depth_clearance_m)
            actor.set_pose(sapien.Pose([float(sx), float(sy), z], _yaw_to_quat(0.0)))
            actor.set_velocity([0.0, 0.0, 0.0])
            actor.set_angular_velocity([0.0, 0.0, 0.0])

        self._time_sec = 0.0
        self._reset_particles()

        # Reposition the external scene mesh relative to the goal so it appears in the rollout.
        if self.external_meshes and self.external_actors:
            gx, gy = float(self.goal_xy_m[0]), float(self.goal_xy_m[1])
            # Keep the asset visible early in the rollout (near start but along the path to goal).
            px = 0.35 * gx
            py = 0.35 * gy
            floor_z = float(self.terrain.height_at_xy(px, py))
            pz = float(floor_z + 0.6)
            yaw = float(self.rng.uniform(-1.0, 1.0))
            self.external_meshes[0] = RenderMesh(
                mesh_path=self.external_meshes[0].mesh_path,
                position_m=(float(px), float(py), float(pz)),
                yaw_rad=yaw,
                scale_m=float(self.external_meshes[0].scale_m),
                color_bgr=self.external_meshes[0].color_bgr,
            )
            self.external_actors[0].set_pose(sapien.Pose([float(px), float(py), float(pz)], _yaw_to_quat(yaw)))

    def _sample_current_xy(self, x_m: float, y_m: float) -> tuple[float, float]:
        lat, lon = xy_to_latlon(
            float(x_m),
            float(y_m),
            self.terrain.origin_lat,
            self.terrain.origin_lon,
            xy_scale=float(self.terrain.xy_scale),
        )
        u, v = self.sampler.ds.nearest_uv(
            latitude=float(lat),
            longitude=float(lon),
            time_index=self.time_index,
            depth_index=self.depth_index,
            include_tides=self.include_tides,
        )
        scale = float(self.config.current_scale)
        return float(u) * scale, float(v) * scale

    def observe(self) -> dict[str, object]:
        agent_obs = []
        for idx, actor in enumerate(self.vehicle_actors):
            pose = actor.get_pose()
            vel = actor.get_velocity()
            u, v = self._sample_current_xy(float(pose.p[0]), float(pose.p[1]))
            floor_z = float(self.terrain.height_at_xy(float(pose.p[0]), float(pose.p[1])))
            agent_obs.append(
                {
                    "agent": int(idx),
                    "x_m": float(pose.p[0]),
                    "y_m": float(pose.p[1]),
                    "z_m": float(pose.p[2]),
                    "vx_mps": float(vel[0]),
                    "vy_mps": float(vel[1]),
                    "vz_mps": float(vel[2]),
                    "current_u_mps": float(u),
                    "current_v_mps": float(v),
                    "floor_z_m": float(floor_z),
                }
            )
        return {
            "agents": agent_obs,
            "goal_x_m": float(self.goal_xy_m[0]),
            "goal_y_m": float(self.goal_xy_m[1]),
        }

    def step(self, cmd_rel_vel_mps: list[tuple[float, float, float]]) -> dict[str, object]:
        if len(cmd_rel_vel_mps) != self.agents:
            raise ValueError(f"Expected {self.agents} actions, got {len(cmd_rel_vel_mps)}")

        alpha = float(self.config.dt_sec) / max(1e-6, float(self.config.velocity_tau_sec))
        alpha = float(np.clip(alpha, 0.0, 1.0))

        per_agent = []
        for idx, actor in enumerate(self.vehicle_actors):
            pose = actor.get_pose()
            vel = actor.get_velocity()
            u, v = self._sample_current_xy(float(pose.p[0]), float(pose.p[1]))

            cmd = np.asarray(cmd_rel_vel_mps[idx], dtype=np.float64)
            speed = float(np.linalg.norm(cmd))
            if speed > float(self.config.max_rel_speed_mps):
                cmd = cmd * (float(self.config.max_rel_speed_mps) / max(1e-9, speed))
            target = np.asarray([cmd[0] + u, cmd[1] + v, cmd[2]], dtype=np.float64)
            v_new = (1.0 - alpha) * np.asarray(vel, dtype=np.float64) + alpha * target
            actor.set_velocity(v_new.tolist())
            actor.set_angular_velocity([0.0, 0.0, 0.0])

            yaw = yaw_from_velocity(float(v_new[0]), float(v_new[1]), default=0.0)
            actor.set_pose(sapien.Pose(pose.p, _yaw_to_quat(yaw)))

            per_agent.append(
                {
                    "agent": int(idx),
                    "cmd_rel_vx_mps": float(cmd[0]),
                    "cmd_rel_vy_mps": float(cmd[1]),
                    "cmd_rel_vz_mps": float(cmd[2]),
                    "current_u_mps": float(u),
                    "current_v_mps": float(v),
                    "yaw_rad": float(yaw),
                }
            )

        self.scene.step()
        self._time_sec += float(self.config.dt_sec)

        # Drift suspended particles (simple visual cue; not a physical simulation).
        if self._particles_xyz_m.size:
            self._particles_xyz_m = self._particles_xyz_m + self._particles_vel_mps * float(self.config.dt_sec)
            x_min, x_max, y_min, y_max = self.terrain.bounds_xy_m
            z_min = float(self.config.terrain_z_min_m)
            z_max = 2.0
            self._particles_xyz_m[:, 0] = np.where(
                self._particles_xyz_m[:, 0] < x_min,
                self._particles_xyz_m[:, 0] + (x_max - x_min),
                self._particles_xyz_m[:, 0],
            )
            self._particles_xyz_m[:, 0] = np.where(
                self._particles_xyz_m[:, 0] > x_max,
                self._particles_xyz_m[:, 0] - (x_max - x_min),
                self._particles_xyz_m[:, 0],
            )
            self._particles_xyz_m[:, 1] = np.where(
                self._particles_xyz_m[:, 1] < y_min,
                self._particles_xyz_m[:, 1] + (y_max - y_min),
                self._particles_xyz_m[:, 1],
            )
            self._particles_xyz_m[:, 1] = np.where(
                self._particles_xyz_m[:, 1] > y_max,
                self._particles_xyz_m[:, 1] - (y_max - y_min),
                self._particles_xyz_m[:, 1],
            )
            self._particles_xyz_m[:, 2] = np.where(
                self._particles_xyz_m[:, 2] < z_min,
                self._particles_xyz_m[:, 2] + (z_max - z_min),
                self._particles_xyz_m[:, 2],
            )
            self._particles_xyz_m[:, 2] = np.where(
                self._particles_xyz_m[:, 2] > z_max,
                self._particles_xyz_m[:, 2] - (z_max - z_min),
                self._particles_xyz_m[:, 2],
            )

        contacts = self.scene.get_contacts()
        collided = False
        if contacts and self.config.terminate_on_collision:
            for c in contacts:
                actors = {c.actor0, c.actor1}
                if any(v in actors for v in self.vehicle_actors) and (self.terrain_actor in actors or any(o in actors for o in self.obstacle_actors)):
                    collided = True
                    break

        invalid = False
        if self.config.terminate_on_invalid_region:
            x_min, x_max, y_min, y_max = self.terrain.bounds_xy_m
            margin = float(self.terrain.x_coords_m[1] - self.terrain.x_coords_m[0]) if len(self.terrain.x_coords_m) > 1 else 8.0
            for actor in self.vehicle_actors:
                p = actor.get_pose().p
                if not (x_min + margin <= float(p[0]) <= x_max - margin and y_min + margin <= float(p[1]) <= y_max - margin):
                    invalid = True
                    break

        return {
            "per_agent": per_agent,
            "collided": bool(collided),
            "invalid_region": bool(invalid),
        }

    def render(
        self,
        *,
        camera: CameraConfig,
        pose: CameraPose,
    ) -> np.ndarray:
        vehicles = []
        for idx, actor in enumerate(self.vehicle_actors):
            p = actor.get_pose().p
            v = actor.get_velocity()
            yaw = yaw_from_velocity(float(v[0]), float(v[1]), default=0.0)
            color = self.vehicle_colors_bgr[min(idx, len(self.vehicle_colors_bgr) - 1)]
            vehicles.append(
                RenderVehicle(
                    position_m=(float(p[0]), float(p[1]), float(p[2])),
                    yaw_rad=float(yaw),
                    color_bgr=color,
                    scale_m=1.1,
                    mesh_path=self.vehicle_mesh_paths[min(idx, len(self.vehicle_mesh_paths) - 1)],
                )
            )

        img = render_scene(
            terrain_vertices_m=self.terrain.vertices_m,
            terrain_faces=self.terrain.faces,
            terrain_face_normals=self.terrain_face_normals,
            meshes=self.external_meshes,
            obstacles=self.obstacles,
            vehicles=vehicles,
            camera=camera,
            pose=pose,
            particles_xyz_m=self._particles_xyz_m,
            time_sec=float(self._time_sec),
            water_color_bgr=(155, 105, 40),
        )
        return img

    def dump_world_metadata(self) -> dict[str, object]:
        x_min, x_max, y_min, y_max = self.terrain.bounds_xy_m
        return {
            "dataset_path": str(self.dataset_path),
            "time_index": int(self.time_index),
            "depth_index": int(self.depth_index),
            "include_tides": bool(self.include_tides),
            "terrain": {
                "origin_lat": float(self.terrain.origin_lat),
                "origin_lon": float(self.terrain.origin_lon),
                "xy_scale": float(self.terrain.xy_scale),
                "bounds_xy_m": [float(x_min), float(x_max), float(y_min), float(y_max)],
                "lat_slice": [int(self.terrain.lat_slice[0]), int(self.terrain.lat_slice[1])],
                "lon_slice": [int(self.terrain.lon_slice[0]), int(self.terrain.lon_slice[1])],
                "grid_size": int(self.config.terrain_grid_size),
                "z_min_m": float(self.config.terrain_z_min_m),
                "z_max_m": float(self.config.terrain_z_max_m),
                "elevation_min": float(self.terrain.elevation_min),
                "elevation_max": float(self.terrain.elevation_max),
                "obj_path": self.artifacts.terrain_obj,
            },
            "external_scene": self.artifacts.external_scene,
            "current_scale": float(self.config.current_scale),
            "config": asdict(self.config),
        }
