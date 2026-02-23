from __future__ import annotations

from dataclasses import dataclass
from math import atan2, sqrt
from typing import Iterable, Sequence

import numpy as np

from ..mujoco3d_scene import ObstacleSpec


def _norm2(x: float, y: float) -> float:
    return sqrt(x * x + y * y)


def _unit2(x: float, y: float) -> tuple[float, float]:
    n = _norm2(x, y)
    if n <= 1e-9:
        return 1.0, 0.0
    return x / n, y / n


def _obstacle_bounding_radius(obstacle: ObstacleSpec) -> float:
    if obstacle.shape == "sphere":
        return float(obstacle.size_xyz_m[0])
    if obstacle.shape == "box":
        hx, hy, hz = obstacle.size_xyz_m
        return float(sqrt(hx * hx + hy * hy + hz * hz))
    if obstacle.shape == "cylinder":
        r, half_h, _ = obstacle.size_xyz_m
        return float(sqrt(r * r + half_h * half_h))
    if obstacle.shape == "ellipsoid":
        return float(max(obstacle.size_xyz_m))
    return float(max(obstacle.size_xyz_m))


def count_collisions(
    agent_xyz_m: tuple[float, float, float],
    agent_radius_m: float,
    obstacles: Sequence[ObstacleSpec],
) -> int:
    ax, ay, az = agent_xyz_m
    count = 0
    for obstacle in obstacles:
        ox, oy, oz = obstacle.pos_xyz_m
        rr = agent_radius_m + _obstacle_bounding_radius(obstacle)
        dx = ax - ox
        dy = ay - oy
        dz = az - oz
        if (dx * dx + dy * dy + dz * dz) <= rr * rr:
            count += 1
    return int(count)


@dataclass
class Nav3DTaskConfig:
    goal_distance_m: float = 60.0
    goal_tolerance_m: float = 12.0
    cruise_speed_mps: float = 3.0
    slowdown_radius_m: float = 30.0
    obstacle_influence_m: float = 10.0
    obstacle_repulsion_gain: float = 7.0
    desired_depth_z_m: float = -4.0
    max_steps: int = 900


@dataclass
class Nav3DController:
    max_speed_mps: float = 1.4
    slowdown_radius_m: float = 90.0
    compensate_current: bool = True
    obstacle_influence_m: float = 10.0
    obstacle_repulsion_gain: float = 7.0

    def act(
        self,
        *,
        obs: dict[str, float],
        goal_xyz_m: tuple[float, float, float],
        obstacles: Sequence[ObstacleSpec],
        desired_depth_z_m: float,
    ) -> dict[str, float]:
        x = float(obs["x_m"])
        y = float(obs["y_m"])
        z = float(obs["z_m"])
        gx, gy, _gz = goal_xyz_m
        dx = gx - x
        dy = gy - y
        dist = max(1e-6, _norm2(dx, dy))

        base_speed = self.max_speed_mps * min(1.0, dist / max(1e-6, self.slowdown_radius_m))
        ux, uy = _unit2(dx, dy)
        vx = base_speed * ux
        vy = base_speed * uy

        # Potential-field obstacle avoidance (repulsive term).
        rep_x = 0.0
        rep_y = 0.0
        for obstacle in obstacles:
            ox, oy, _oz = obstacle.pos_xyz_m
            ddx = x - ox
            ddy = y - oy
            d = _norm2(ddx, ddy)
            if d <= 1e-6 or d > self.obstacle_influence_m:
                continue
            away_x, away_y = ddx / d, ddy / d
            strength = self.obstacle_repulsion_gain * (1.0 / d - 1.0 / self.obstacle_influence_m) / (d * d)
            rep_x += strength * away_x
            rep_y += strength * away_y
        vx += rep_x
        vy += rep_y

        current_u = float(obs.get("current_u_mps", 0.0))
        current_v = float(obs.get("current_v_mps", 0.0))
        if self.compensate_current:
            vx_cmd = vx - current_u
            vy_cmd = vy - current_v
        else:
            vx_cmd = vx
            vy_cmd = vy

        speed = _norm2(vx_cmd, vy_cmd)
        if speed > self.max_speed_mps:
            s = self.max_speed_mps / max(1e-9, speed)
            vx_cmd *= s
            vy_cmd *= s

        yaw = atan2(vy_cmd, vx_cmd)
        return {"vx_mps": float(vx_cmd), "vy_mps": float(vy_cmd), "z_m": float(desired_depth_z_m), "yaw_rad": float(yaw)}


def sample_obstacles(
    rng: np.random.Generator,
    *,
    count: int,
    x_half_m: float,
    y_half_m: float,
    z_center_m: float,
) -> tuple[ObstacleSpec, ...]:
    obstacles: list[ObstacleSpec] = []
    for i in range(int(count)):
        shape = "ellipsoid" if (i % 4) != 0 else "cylinder"
        x = float(rng.uniform(-0.85 * x_half_m, 0.85 * x_half_m))
        y = float(rng.uniform(-0.85 * y_half_m, 0.85 * y_half_m))
        z = float(z_center_m + rng.uniform(-2.4, 1.4))
        if shape == "ellipsoid":
            a = float(rng.uniform(1.0, 2.8))
            b = float(rng.uniform(0.9, 2.4))
            c = float(rng.uniform(1.2, 4.8))
            size = (a, b, c)
            yaw = float(rng.uniform(0.0, 2.0 * np.pi))
            quat = (float(np.cos(0.5 * yaw)), 0.0, 0.0, float(np.sin(0.5 * yaw)))
        else:
            r = float(rng.uniform(0.7, 1.6))
            half_h = float(rng.uniform(1.6, 5.2))
            size = (r, half_h, 0.0)
            quat = None
        obstacles.append(
            ObstacleSpec(
                name=f"obstacle{i:02d}",
                shape=shape,
                pos_xyz_m=(x, y, z),
                size_xyz_m=size,
                rgba=(1.0, 1.0, 1.0, 1.0),
                material="mat_rock",
                quat_wxyz=quat,
            )
        )
    return tuple(obstacles)
