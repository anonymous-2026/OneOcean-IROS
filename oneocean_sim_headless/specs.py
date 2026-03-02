from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Literal

from .controllers import ControllerConfig

if TYPE_CHECKING:  # pragma: no cover
    from .env import EnvConfig


@dataclass(frozen=True)
class VehicleSpec:
    """Vehicle-level contract for paper-facing runs (H1 v2)."""

    # Integrator / cadence
    dt_s: float
    # Kinematic limits (proxy; headless uses high-level velocity actions)
    max_speed_mps: float
    # Depth convention: y is positive-down depth (meters)
    y_depth_range_m: tuple[float, float]
    # High-level action interface (paper-facing): desired relative velocity in world frame.
    action_space: str = "desired_relative_velocity_world_xyz"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObsSpec:
    """Observation/recording contract (what signals exist downstream)."""

    # Minimal streams that must exist for every run dir.
    streams: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {"streams": list(self.streams)}


@dataclass(frozen=True)
class ConstraintSpec:
    """Constraint contract for headless tasks (toggleable)."""

    constraint_mode: Literal["off", "hard"]
    bathy_mode: Literal["off", "hard"]
    land_mask_threshold: float
    seafloor_clearance_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DynamicsSpec:
    """Dynamics + low-level control contract (H1 v2)."""

    dynamics_model: Literal["kinematic", "3dof", "6dof"]
    current_handling: str  # e.g., "relative_velocity"

    # Minimal param snapshot (diagonal model).
    mass_linear: tuple[float, float, float]
    mass_angular: tuple[float, float, float]
    damping_linear: tuple[float, float, float]
    damping_angular: tuple[float, float, float]

    # Velocity tracking gains used to map nu_cmd -> tau (diagonal PID-lite).
    kp_linear: tuple[float, float, float]
    kp_angular: tuple[float, float, float]
    kd_angular: tuple[float, float, float]

    # Angle convention note (important for yaw axis).
    angle_convention: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_spec_snapshot(*, env_cfg: "EnvConfig", controller: ControllerConfig) -> dict[str, Any]:
    vehicle = VehicleSpec(
        dt_s=float(env_cfg.dt_s),
        max_speed_mps=float(env_cfg.max_speed_mps),
        y_depth_range_m=tuple(float(x) for x in env_cfg.y_depth_range_m),
    )
    obs = ObsSpec(
        streams=(
            "agents/*/pose_groundtruth/data.csv",
            "agents/*/body_velocity/data.csv",
            "agents/*/actions/data.csv",
            "agents/*/obs/local_current/data.csv",
            "agents/*/obs/pollution_probe/data.csv",
            "agents/*/obs/latlon/data.csv",
            "agents/*/obs/bathymetry/data.csv",
            "run_meta.json",
            "metrics.json",
            "metrics.csv",
        )
    )
    cons = ConstraintSpec(
        constraint_mode=str(env_cfg.constraint_mode),  # type: ignore[arg-type]
        bathy_mode=str(env_cfg.bathy_mode),  # type: ignore[arg-type]
        land_mask_threshold=float(env_cfg.land_mask_threshold),
        seafloor_clearance_m=float(env_cfg.seafloor_clearance_m),
    )
    dyn = DynamicsSpec(
        dynamics_model=str(env_cfg.dynamics_model),  # type: ignore[arg-type]
        current_handling="relative_velocity",
        mass_linear=tuple(float(x) for x in env_cfg.dyn_mass_linear),
        mass_angular=tuple(float(x) for x in env_cfg.dyn_mass_angular),
        damping_linear=tuple(float(x) for x in env_cfg.dyn_damping_linear),
        damping_angular=tuple(float(x) for x in env_cfg.dyn_damping_angular),
        kp_linear=tuple(float(x) for x in env_cfg.dyn_kp_linear),
        kp_angular=tuple(float(x) for x in env_cfg.dyn_kp_angular),
        kd_angular=tuple(float(x) for x in env_cfg.dyn_kd_angular),
        angle_convention="world(x=east,y=down,z=north); yaw is about +y (down) axis; roll about +x; pitch about +z",
    )
    return {
        "schema_version": "v2",
        "vehicle_spec": vehicle.to_dict(),
        "obs_spec": obs.to_dict(),
        "constraint_spec": cons.to_dict(),
        "dynamics_spec": dyn.to_dict(),
        "controller": controller.to_dict(),
    }
