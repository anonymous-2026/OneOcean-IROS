from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

from .data import CombinedDataset, CurrentSampler
from .mujoco3d_scene import AgentSpec, ObstacleSpec, OceanSceneSpec, build_ocean_scene_xml, write_heightfield_png


METERS_PER_DEG_LAT = 111_320.0


def meters_per_deg_lon(latitude_deg: float) -> float:
    return METERS_PER_DEG_LAT * max(0.1, cos(radians(latitude_deg)))


@dataclass(frozen=True)
class GeoMapper:
    lat_min: float
    lat_max: float
    lon_min: float
    lon_max: float
    x_half_m: float
    y_half_m: float

    def xy_to_latlon(self, x_m: float, y_m: float) -> tuple[float, float]:
        fx = (x_m + self.x_half_m) / max(1e-9, 2.0 * self.x_half_m)
        fy = (y_m + self.y_half_m) / max(1e-9, 2.0 * self.y_half_m)
        lon = self.lon_min + fx * (self.lon_max - self.lon_min)
        lat = self.lat_min + fy * (self.lat_max - self.lat_min)
        return float(lat), float(lon)

    def latlon_to_xy(self, lat: float, lon: float) -> tuple[float, float]:
        fx = (lon - self.lon_min) / max(1e-9, self.lon_max - self.lon_min)
        fy = (lat - self.lat_min) / max(1e-9, self.lat_max - self.lat_min)
        x = fx * (2.0 * self.x_half_m) - self.x_half_m
        y = fy * (2.0 * self.y_half_m) - self.y_half_m
        return float(x), float(y)


@dataclass
class Mujoco3DConfig:
    dt_sec: float = 0.05
    max_steps: int = 900
    vel_gain: float = 55.0
    yaw_gain: float = 6.0
    max_force_n: float = 160.0
    max_torque_yaw: float = 25.0
    drag_gain: float = 22.0
    current_speed_scale: float = 80.0
    target_domain_size_m: float = 1000.0
    meters_per_sim_meter: Optional[float] = None
    terrain_height_m: float = 7.0
    terrain_base_z_m: float = -14.0
    water_depth_m: float = 22.0
    camera_distance_m: float = 40.0
    camera_elevation_m: float = 20.0


@dataclass(frozen=True)
class AgentHandles:
    name: str
    joint_x: int
    joint_y: int
    joint_z: int
    joint_yaw: int
    dof_x: int
    dof_y: int
    dof_z: int
    dof_yaw: int
    act_fx: int
    act_fy: int
    act_fz: int
    act_tau_yaw: int


def _name2id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj, name)
    if idx < 0:
        raise KeyError(f"MuJoCo name not found: obj={obj} name={name}")
    return int(idx)


def _default_heightfield_cache(variant: str) -> Path:
    return (Path("runs") / "_cache" / "heightfields" / variant).resolve()

def _dataset_real_extents_m(ds: CombinedDataset) -> tuple[float, float]:
    lat_min = float(np.min(ds.latitude_values))
    lat_max = float(np.max(ds.latitude_values))
    lon_min = float(np.min(ds.longitude_values))
    lon_max = float(np.max(ds.longitude_values))
    lat0 = float(np.mean(ds.latitude_values))
    width_m = (lon_max - lon_min) * meters_per_deg_lon(lat0)
    height_m = (lat_max - lat_min) * METERS_PER_DEG_LAT
    return float(width_m), float(height_m)


class OceanMujoco3DEnv:
    def __init__(
        self,
        ds: CombinedDataset,
        sampler: CurrentSampler,
        *,
        variant: str,
        time_index: int,
        depth_index: int,
        seed: int,
        agent_count: int,
        obstacles: tuple[ObstacleSpec, ...],
        config: Optional[Mujoco3DConfig] = None,
    ) -> None:
        if agent_count < 1:
            raise ValueError("agent_count must be >= 1")
        self.ds = ds
        self.sampler = sampler
        self.variant = variant
        self.time_index = int(time_index)
        self.depth_index = int(depth_index)
        self.rng = np.random.default_rng(seed)
        self.cfg = config or Mujoco3DConfig()

        real_w_m, real_h_m = _dataset_real_extents_m(ds)
        default_mpsm = max(1e-6, max(real_w_m, real_h_m) / max(1e-6, self.cfg.target_domain_size_m))
        meters_per_sim_meter = float(self.cfg.meters_per_sim_meter) if self.cfg.meters_per_sim_meter else float(default_mpsm)
        x_half = 0.5 * real_w_m / meters_per_sim_meter
        y_half = 0.5 * real_h_m / meters_per_sim_meter

        self.x_half_m = float(x_half)
        self.y_half_m = float(y_half)
        self.meters_per_sim_meter = float(meters_per_sim_meter)
        self.geo = GeoMapper(
            lat_min=float(np.min(ds.latitude_values)),
            lat_max=float(np.max(ds.latitude_values)),
            lon_min=float(np.min(ds.longitude_values)),
            lon_max=float(np.max(ds.longitude_values)),
            x_half_m=self.x_half_m,
            y_half_m=self.y_half_m,
        )

        heightfield_dir = _default_heightfield_cache(variant)
        heightfield_png = heightfield_dir / "bathy_heightfield.png"
        if not heightfield_png.exists():
            write_heightfield_png(ds.elevation_grid(), heightfield_png, flipud=True)

        agents = tuple(AgentSpec(name=f"agent{i}") for i in range(agent_count))
        scene_spec = OceanSceneSpec(
            model_name="oneocean_ocean3d",
            dt_sec=self.cfg.dt_sec,
            x_half_m=self.x_half_m,
            y_half_m=self.y_half_m,
            terrain_height_m=self.cfg.terrain_height_m,
            terrain_base_z_m=self.cfg.terrain_base_z_m,
            heightfield_png=heightfield_png,
            heightfield_rows=int(ds.elevation_grid().shape[0]),
            heightfield_cols=int(ds.elevation_grid().shape[1]),
            agents=agents,
            obstacles=obstacles,
            water_depth_m=self.cfg.water_depth_m,
            camera_distance_m=self.cfg.camera_distance_m,
            camera_elevation_m=self.cfg.camera_elevation_m,
        )
        xml = build_ocean_scene_xml(scene_spec)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.agents: list[AgentHandles] = []
        for i in range(agent_count):
            name = f"agent{i}"
            jx = _name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_x")
            jy = _name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_y")
            jz = _name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_z")
            jyaw = _name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_yaw")
            ax = _name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_fx")
            ay = _name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_fy")
            az = _name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_fz")
            atawa = _name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_tau_yaw")
            dof_x = int(self.model.jnt_dofadr[jx])
            dof_y = int(self.model.jnt_dofadr[jy])
            dof_z = int(self.model.jnt_dofadr[jz])
            dof_yaw = int(self.model.jnt_dofadr[jyaw])
            self.agents.append(
                AgentHandles(
                    name=name,
                    joint_x=jx,
                    joint_y=jy,
                    joint_z=jz,
                    joint_yaw=jyaw,
                    dof_x=dof_x,
                    dof_y=dof_y,
                    dof_z=dof_z,
                    dof_yaw=dof_yaw,
                    act_fx=ax,
                    act_fy=ay,
                    act_fz=az,
                    act_tau_yaw=atawa,
                )
            )

        goal_body = _name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal_marker")
        source_body = _name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "source_marker")
        self.goal_mocap_id = int(self.model.body_mocapid[goal_body])
        self.source_mocap_id = int(self.model.body_mocapid[source_body])
        if self.goal_mocap_id < 0 or self.source_mocap_id < 0:
            raise RuntimeError("Expected goal/source markers to be mocap bodies")

        self.obstacles = obstacles

    def close(self) -> None:
        pass

    def sim_uv_mps(self, latitude: float, longitude: float) -> tuple[float, float]:
        u_real, v_real = self.sampler.sample_uv(latitude, longitude, self.time_index, self.depth_index)
        scale = 1.0 / max(1e-9, self.meters_per_sim_meter)
        return float(u_real * scale * self.cfg.current_speed_scale), float(v_real * scale * self.cfg.current_speed_scale)

    def reset(
        self,
        *,
        agent_xyz_m: list[tuple[float, float, float]],
        agent_yaw_rad: Optional[list[float]] = None,
        goal_xyz_m: tuple[float, float, float] = (0.0, 0.0, -3.5),
        source_xyz_m: tuple[float, float, float] = (0.0, 0.0, -3.5),
    ) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        if agent_yaw_rad is None:
            agent_yaw_rad = [0.0 for _ in agent_xyz_m]
        if len(agent_xyz_m) != len(self.agents):
            raise ValueError("agent_xyz_m length mismatch")
        for handle, xyz, yaw in zip(self.agents, agent_xyz_m, agent_yaw_rad, strict=True):
            qpos_x = int(self.model.jnt_qposadr[handle.joint_x])
            qpos_y = int(self.model.jnt_qposadr[handle.joint_y])
            qpos_z = int(self.model.jnt_qposadr[handle.joint_z])
            qpos_yaw = int(self.model.jnt_qposadr[handle.joint_yaw])
            self.data.qpos[qpos_x] = float(xyz[0])
            self.data.qpos[qpos_y] = float(xyz[1])
            self.data.qpos[qpos_z] = float(xyz[2])
            self.data.qpos[qpos_yaw] = float(yaw)

        self.data.mocap_pos[self.goal_mocap_id] = np.asarray(goal_xyz_m, dtype=float)
        self.data.mocap_pos[self.source_mocap_id] = np.asarray(source_xyz_m, dtype=float)
        mujoco.mj_forward(self.model, self.data)

    def agent_state(self, agent_index: int) -> dict[str, float]:
        handle = self.agents[agent_index]
        qpos_x = int(self.model.jnt_qposadr[handle.joint_x])
        qpos_y = int(self.model.jnt_qposadr[handle.joint_y])
        qpos_z = int(self.model.jnt_qposadr[handle.joint_z])
        qpos_yaw = int(self.model.jnt_qposadr[handle.joint_yaw])
        x = float(self.data.qpos[qpos_x])
        y = float(self.data.qpos[qpos_y])
        z = float(self.data.qpos[qpos_z])
        yaw = float(self.data.qpos[qpos_yaw])
        vx = float(self.data.qvel[handle.dof_x])
        vy = float(self.data.qvel[handle.dof_y])
        vz = float(self.data.qvel[handle.dof_z])
        vyaw = float(self.data.qvel[handle.dof_yaw])
        lat, lon = self.geo.xy_to_latlon(x, y)
        u_sim, v_sim = self.sim_uv_mps(lat, lon)
        invalid = self.sampler.invalid_region(lat, lon)
        return {
            "agent_index": float(agent_index),
            "x_m": x,
            "y_m": y,
            "z_m": z,
            "yaw_rad": yaw,
            "vx_mps": vx,
            "vy_mps": vy,
            "vz_mps": vz,
            "yaw_rate_rps": vyaw,
            "latitude": float(lat),
            "longitude": float(lon),
            "current_u_mps": float(u_sim),
            "current_v_mps": float(v_sim),
            "invalid_region": float(invalid),
        }

    def set_markers(
        self,
        *,
        goal_xyz_m: Optional[tuple[float, float, float]] = None,
        source_xyz_m: Optional[tuple[float, float, float]] = None,
    ) -> None:
        if goal_xyz_m is not None:
            self.data.mocap_pos[self.goal_mocap_id] = np.asarray(goal_xyz_m, dtype=float)
        if source_xyz_m is not None:
            self.data.mocap_pos[self.source_mocap_id] = np.asarray(source_xyz_m, dtype=float)

    def _apply_current_drag(self) -> None:
        self.data.qfrc_applied[:] = 0.0
        for idx, handle in enumerate(self.agents):
            state = self.agent_state(idx)
            fx = self.cfg.drag_gain * (state["current_u_mps"] - state["vx_mps"])
            fy = self.cfg.drag_gain * (state["current_v_mps"] - state["vy_mps"])
            self.data.qfrc_applied[handle.dof_x] += float(fx)
            self.data.qfrc_applied[handle.dof_y] += float(fy)

    def step(self, desired: list[dict[str, float]]) -> None:
        if len(desired) != len(self.agents):
            raise ValueError("desired list length mismatch")
        self._apply_current_drag()

        for idx, handle in enumerate(self.agents):
            state = self.agent_state(idx)
            vx_des = float(desired[idx].get("vx_mps", 0.0))
            vy_des = float(desired[idx].get("vy_mps", 0.0))
            z_des = float(desired[idx].get("z_m", state["z_m"]))
            yaw_des = float(desired[idx].get("yaw_rad", state["yaw_rad"]))

            fx = self.cfg.vel_gain * (vx_des - state["vx_mps"])
            fy = self.cfg.vel_gain * (vy_des - state["vy_mps"])
            fz = self.cfg.vel_gain * (0.0 - state["vz_mps"]) + 14.0 * (z_des - state["z_m"])
            tau = self.cfg.yaw_gain * (yaw_des - state["yaw_rad"]) - 0.4 * state["yaw_rate_rps"]

            self.data.ctrl[handle.act_fx] = float(np.clip(fx, -self.cfg.max_force_n, self.cfg.max_force_n))
            self.data.ctrl[handle.act_fy] = float(np.clip(fy, -self.cfg.max_force_n, self.cfg.max_force_n))
            self.data.ctrl[handle.act_fz] = float(np.clip(fz, -self.cfg.max_force_n, self.cfg.max_force_n))
            self.data.ctrl[handle.act_tau_yaw] = float(np.clip(tau, -self.cfg.max_torque_yaw, self.cfg.max_torque_yaw))

        mujoco.mj_step(self.model, self.data)

    def render(self, renderer: mujoco.Renderer, *, camera: str = "cam_main") -> np.ndarray:
        renderer.update_scene(self.data, camera=camera)
        return renderer.render()
