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


def build_spec_snapshot(*, env_cfg: "EnvConfig", controller: ControllerConfig) -> dict[str, Any]:
    vehicle = VehicleSpec(
        dt_s=float(env_cfg.dt_s),
        max_speed_mps=float(env_cfg.max_speed_mps),
        y_depth_range_m=tuple(float(x) for x in env_cfg.y_depth_range_m),
    )
    obs = ObsSpec(
        streams=(
            "agents/*/pose_groundtruth/data.csv",
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
    return {
        "schema_version": "v2",
        "vehicle_spec": vehicle.to_dict(),
        "obs_spec": obs.to_dict(),
        "constraint_spec": cons.to_dict(),
        "controller": controller.to_dict(),
    }
