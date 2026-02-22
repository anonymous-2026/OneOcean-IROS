from __future__ import annotations

from dataclasses import dataclass
from math import cos, radians, sin, sqrt, tanh, pi
from typing import Optional

import mujoco
import numpy as np

from .data import CurrentSampler


METERS_PER_DEG_LAT = 111_132.0


def meters_per_deg_lon(latitude_deg: float) -> float:
    return METERS_PER_DEG_LAT * max(0.1, cos(radians(latitude_deg)))


def xy_to_latlon(x_m: float, y_m: float, lat0: float, lon0: float) -> tuple[float, float]:
    lat = lat0 + (y_m / METERS_PER_DEG_LAT)
    lon = lon0 + (x_m / meters_per_deg_lon(lat0))
    return lat, lon


def latlon_to_xy(lat: float, lon: float, lat0: float, lon0: float) -> tuple[float, float]:
    x_m = (lon - lon0) * meters_per_deg_lon(lat0)
    y_m = (lat - lat0) * METERS_PER_DEG_LAT
    return x_m, y_m


@dataclass
class NavigationConfig:
    dt_sec: float = 0.5
    max_steps: int = 600
    max_speed_mps: float = 1.8
    goal_tolerance_m: float = 20.0
    control_gain: float = 25.0
    max_force_n: float = 120.0
    terminate_on_invalid_region: bool = True


class OceanNavigationEnv:
    def __init__(
        self,
        sampler: CurrentSampler,
        time_index: int,
        depth_index: int,
        seed: int,
        config: Optional[NavigationConfig] = None,
    ) -> None:
        self.sampler = sampler
        self.time_index = time_index
        self.depth_index = depth_index
        self.config = config or NavigationConfig()
        self.rng = np.random.default_rng(seed)
        self.model, self.data = self._build_model(self.config)

        self.origin_lat = 0.0
        self.origin_lon = 0.0
        self.goal_x_m = 0.0
        self.goal_y_m = 0.0
        self.step_count = 0
        self.trajectory: list[dict[str, float]] = []

    def _build_model(self, cfg: NavigationConfig) -> tuple[mujoco.MjModel, mujoco.MjData]:
        xml = f"""
        <mujoco model="oneocean_nav">
          <option timestep="{cfg.dt_sec}" gravity="0 0 0" integrator="Euler"/>
          <worldbody>
            <body name="vehicle" pos="0 0 0.1">
              <joint name="x" type="slide" axis="1 0 0" damping="2.0"/>
              <joint name="y" type="slide" axis="0 1 0" damping="2.0"/>
              <geom type="sphere" size="0.25" mass="50.0" rgba="0.8 0.2 0.2 1"/>
            </body>
          </worldbody>
          <actuator>
            <motor name="fx" joint="x" gear="1" ctrlrange="-{cfg.max_force_n} {cfg.max_force_n}" ctrllimited="true"/>
            <motor name="fy" joint="y" gear="1" ctrlrange="-{cfg.max_force_n} {cfg.max_force_n}" ctrllimited="true"/>
          </actuator>
        </mujoco>
        """.strip()
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        return model, data

    def _distance_to_goal(self) -> float:
        dx = self.goal_x_m - float(self.data.qpos[0])
        dy = self.goal_y_m - float(self.data.qpos[1])
        return sqrt(dx * dx + dy * dy)

    def _observation(self) -> dict[str, float]:
        x_m = float(self.data.qpos[0])
        y_m = float(self.data.qpos[1])
        latitude, longitude = xy_to_latlon(x_m, y_m, self.origin_lat, self.origin_lon)
        current_u, current_v = self.sampler.sample_uv(
            latitude, longitude, self.time_index, self.depth_index
        )
        distance = self._distance_to_goal()
        return {
            "x_m": x_m,
            "y_m": y_m,
            "vx_mps": float(self.data.qvel[0]),
            "vy_mps": float(self.data.qvel[1]),
            "goal_x_m": self.goal_x_m,
            "goal_y_m": self.goal_y_m,
            "distance_to_goal_m": distance,
            "latitude": latitude,
            "longitude": longitude,
            "current_u_mps": current_u,
            "current_v_mps": current_v,
        }

    def observe(self) -> dict[str, float]:
        return self._observation()

    def reset(
        self,
        start_latlon: Optional[tuple[float, float]] = None,
        goal_latlon: Optional[tuple[float, float]] = None,
        goal_distance_m: float = 600.0,
        goal_bearing_deg: Optional[float] = None,
    ) -> dict[str, float]:
        self.step_count = 0
        self.trajectory = []
        mujoco.mj_resetData(self.model, self.data)

        if start_latlon is None:
            self.origin_lat, self.origin_lon = self.sampler.center_latlon()
        else:
            self.origin_lat, self.origin_lon = start_latlon
        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qvel[:] = 0.0

        if goal_latlon is not None:
            self.goal_x_m, self.goal_y_m = latlon_to_xy(
                goal_latlon[0], goal_latlon[1], self.origin_lat, self.origin_lon
            )
        else:
            if goal_bearing_deg is None:
                goal_bearing_deg = float(self.rng.uniform(0.0, 360.0))
            heading = radians(goal_bearing_deg)
            self.goal_x_m = goal_distance_m * cos(heading)
            self.goal_y_m = goal_distance_m * sin(heading)

        obs = self._observation()
        self._append_trajectory(obs, cmd_speed_mps=0.0, heading_rad=0.0, invalid_region=False)
        return obs

    def _append_trajectory(
        self, obs: dict[str, float], cmd_speed_mps: float, heading_rad: float, invalid_region: bool
    ) -> None:
        self.trajectory.append(
            {
                "step": float(self.step_count),
                "x_m": obs["x_m"],
                "y_m": obs["y_m"],
                "vx_mps": obs["vx_mps"],
                "vy_mps": obs["vy_mps"],
                "latitude": obs["latitude"],
                "longitude": obs["longitude"],
                "current_u_mps": obs["current_u_mps"],
                "current_v_mps": obs["current_v_mps"],
                "goal_x_m": obs["goal_x_m"],
                "goal_y_m": obs["goal_y_m"],
                "distance_to_goal_m": obs["distance_to_goal_m"],
                "cmd_speed_mps": cmd_speed_mps,
                "heading_rad": heading_rad,
                "invalid_region": float(invalid_region),
            }
        )

    def step(
        self, action: dict[str, float], terminate_on_goal: bool = True
    ) -> tuple[dict[str, float], float, bool, dict[str, float]]:
        requested_speed = float(action.get("speed_mps", 0.0))
        heading_rad = float(action.get("heading_rad", 0.0))
        speed_mps = float(np.clip(requested_speed, 0.0, self.config.max_speed_mps))
        heading_rad = ((heading_rad + pi) % (2 * pi)) - pi

        obs_before = self._observation()
        current_u = obs_before["current_u_mps"]
        current_v = obs_before["current_v_mps"]

        cmd_vx = speed_mps * cos(heading_rad)
        cmd_vy = speed_mps * sin(heading_rad)
        target_vx = cmd_vx + current_u
        target_vy = cmd_vy + current_v

        force_x = self.config.control_gain * (target_vx - float(self.data.qvel[0]))
        force_y = self.config.control_gain * (target_vy - float(self.data.qvel[1]))
        self.data.ctrl[0] = float(np.clip(force_x, -self.config.max_force_n, self.config.max_force_n))
        self.data.ctrl[1] = float(np.clip(force_y, -self.config.max_force_n, self.config.max_force_n))
        mujoco.mj_step(self.model, self.data)

        self.step_count += 1
        obs = self._observation()
        invalid_region = self.sampler.invalid_region(obs["latitude"], obs["longitude"])
        reached_goal = obs["distance_to_goal_m"] <= self.config.goal_tolerance_m
        timeout = self.step_count >= self.config.max_steps
        done = (reached_goal and terminate_on_goal) or timeout or (
            self.config.terminate_on_invalid_region and invalid_region
        )
        reward = 1.0 if reached_goal else -tanh(obs["distance_to_goal_m"] / 500.0)

        self._append_trajectory(
            obs, cmd_speed_mps=speed_mps, heading_rad=heading_rad, invalid_region=invalid_region
        )
        info = {
            "success": float(reached_goal),
            "timeout": float(timeout),
            "invalid_region": float(invalid_region),
            "time_sec": self.step_count * self.config.dt_sec,
            "energy_step_proxy": speed_mps * speed_mps * self.config.dt_sec,
        }
        return obs, reward, done, info
