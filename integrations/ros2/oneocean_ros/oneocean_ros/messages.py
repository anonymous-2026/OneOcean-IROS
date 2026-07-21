from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .frames import core_quaternion_to_enu, core_vector_to_enu, enu_vector_to_core


@dataclass(frozen=True)
class AgentSnapshot:
    agent_index: int
    position_enu_m: np.ndarray
    orientation_enu_xyzw: np.ndarray
    linear_velocity_enu_mps: np.ndarray
    angular_velocity_enu_rps: np.ndarray
    current_enu_mps: np.ndarray
    pollution: float
    bathymetry_depth_m: float
    land_mask: float
    latitude_deg: float
    longitude_deg: float


def commands_enu_to_core(commands_enu: np.ndarray, n_agents: int) -> np.ndarray:
    commands = np.asarray(commands_enu, dtype=np.float64)
    if commands.shape != (int(n_agents), 3):
        raise ValueError(f"commands_enu must have shape ({int(n_agents)}, 3), got {commands.shape}")
    if not np.all(np.isfinite(commands)):
        raise ValueError("commands_enu must contain only finite values")
    return enu_vector_to_core(commands)


def snapshots_from_observation(observation: dict[str, Any]) -> list[AgentSnapshot]:
    positions = np.asarray(observation["positions_xyz"], dtype=np.float64)
    quaternions = np.asarray(observation["quaternions_xyzw"], dtype=np.float64)
    body_velocity = np.asarray(observation["body_velocity_uvw_pqr"], dtype=np.float64)
    currents = np.asarray(observation["currents_xyz"], dtype=np.float64)
    probes = np.asarray(observation["pollution_probe"], dtype=np.float64).reshape(-1)
    elevation = np.asarray(observation["elevation_m"], dtype=np.float64).reshape(-1)
    land_mask = np.asarray(observation["land_mask"], dtype=np.float64).reshape(-1)
    latitude = np.asarray(observation["latitude"], dtype=np.float64).reshape(-1)
    longitude = np.asarray(observation["longitude"], dtype=np.float64).reshape(-1)
    n_agents = int(positions.shape[0])
    if positions.shape != (n_agents, 3) or quaternions.shape != (n_agents, 4) or body_velocity.shape != (n_agents, 6) or currents.shape != (n_agents, 3):
        raise ValueError("observation arrays have incompatible agent dimensions")
    snapshots: list[AgentSnapshot] = []
    for agent_index in range(n_agents):
        invalid = np.isfinite(land_mask[agent_index]) and land_mask[agent_index] >= 0.5
        bathymetry = float("nan") if invalid or not np.isfinite(elevation[agent_index]) else max(0.0, -float(elevation[agent_index]))
        snapshots.append(
            AgentSnapshot(
                agent_index=agent_index,
                position_enu_m=core_vector_to_enu(positions[agent_index]),
                orientation_enu_xyzw=core_quaternion_to_enu(quaternions[agent_index]),
                linear_velocity_enu_mps=core_vector_to_enu(body_velocity[agent_index, :3]),
                angular_velocity_enu_rps=core_vector_to_enu(body_velocity[agent_index, 3:]),
                current_enu_mps=core_vector_to_enu(currents[agent_index]),
                pollution=float(probes[agent_index]),
                bathymetry_depth_m=bathymetry,
                land_mask=float(land_mask[agent_index]),
                latitude_deg=float(latitude[agent_index]),
                longitude_deg=float(longitude[agent_index]),
            )
        )
    return snapshots
