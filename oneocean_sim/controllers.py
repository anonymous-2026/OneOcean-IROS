from __future__ import annotations

from dataclasses import dataclass
from math import atan2


@dataclass
class GoalSeekingController:
    max_speed_mps: float = 1.8
    slowdown_radius_m: float = 180.0
    compensate_current: bool = True

    def act(self, observation: dict[str, float]) -> dict[str, float]:
        dx = observation["goal_x_m"] - observation["x_m"]
        dy = observation["goal_y_m"] - observation["y_m"]
        distance = max(1e-6, observation["distance_to_goal_m"])
        desired_speed = self.max_speed_mps * min(1.0, distance / self.slowdown_radius_m)
        desired_vx = desired_speed * (dx / distance)
        desired_vy = desired_speed * (dy / distance)
        current_u = observation.get("current_u_mps", 0.0)
        current_v = observation.get("current_v_mps", 0.0)
        if self.compensate_current:
            cmd_vx = desired_vx - current_u
            cmd_vy = desired_vy - current_v
        else:
            cmd_vx = desired_vx
            cmd_vy = desired_vy
        cmd_norm = (cmd_vx * cmd_vx + cmd_vy * cmd_vy) ** 0.5
        if cmd_norm > self.max_speed_mps:
            scale = self.max_speed_mps / cmd_norm
            cmd_vx *= scale
            cmd_vy *= scale
        speed = (cmd_vx * cmd_vx + cmd_vy * cmd_vy) ** 0.5
        heading = atan2(cmd_vy, cmd_vx)
        return {"speed_mps": speed, "heading_rad": heading}


@dataclass
class StationKeepingController:
    max_speed_mps: float = 1.8
    position_gain: float = 0.02
    compensate_current: bool = True

    def act(self, observation: dict[str, float]) -> dict[str, float]:
        x = observation["x_m"]
        y = observation["y_m"]
        current_u = observation.get("current_u_mps", 0.0)
        current_v = observation.get("current_v_mps", 0.0)

        desired_vx = -self.position_gain * x
        desired_vy = -self.position_gain * y
        if self.compensate_current:
            cmd_vx = desired_vx - current_u
            cmd_vy = desired_vy - current_v
        else:
            cmd_vx = desired_vx
            cmd_vy = desired_vy
        norm = (cmd_vx * cmd_vx + cmd_vy * cmd_vy) ** 0.5
        if norm > self.max_speed_mps:
            scale = self.max_speed_mps / norm
            cmd_vx *= scale
            cmd_vy *= scale
        speed = (cmd_vx * cmd_vx + cmd_vy * cmd_vy) ** 0.5
        heading = atan2(cmd_vy, cmd_vx)
        return {"speed_mps": speed, "heading_rad": heading}
