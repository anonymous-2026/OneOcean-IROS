from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ObstacleSpec:
    name: str
    shape: str  # "sphere" | "box" | "cylinder"
    pos_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = (0.55, 0.55, 0.55, 1.0)


@dataclass(frozen=True)
class AgentSpec:
    name: str
    rgba: tuple[float, float, float, float] = (0.85, 0.25, 0.25, 1.0)
    radius_m: float = 0.45
    length_m: float = 1.2
    mass_kg: float = 40.0


@dataclass(frozen=True)
class OceanSceneSpec:
    model_name: str
    dt_sec: float
    x_half_m: float
    y_half_m: float
    terrain_height_m: float
    terrain_base_z_m: float
    heightfield_png: Path
    heightfield_rows: int
    heightfield_cols: int
    agents: tuple[AgentSpec, ...]
    obstacles: tuple[ObstacleSpec, ...]
    water_depth_m: float = 20.0
    camera_distance_m: float = 35.0
    camera_elevation_m: float = 18.0


def write_heightfield_png(
    elevation_m: np.ndarray,
    output_png: Path,
    *,
    flipud: bool = True,
) -> Path:
    output_png.parent.mkdir(parents=True, exist_ok=True)

    elev = np.asarray(elevation_m, dtype=np.float32)
    finite = np.isfinite(elev)
    if not np.any(finite):
        raise ValueError("Elevation grid has no finite values")

    vmin = float(np.min(elev[finite]))
    vmax = float(np.max(elev[finite]))
    denom = max(1e-6, vmax - vmin)
    norm = (elev - vmin) / denom
    norm = np.clip(norm, 0.0, 1.0)
    img = (norm * 255.0).astype(np.uint8)
    if flipud:
        img = np.flipud(img)
    Image.fromarray(img, mode="L").save(output_png)
    return output_png


def _fmt_rgba(rgba: tuple[float, float, float, float]) -> str:
    return " ".join(f"{v:.4g}" for v in rgba)


def _fmt_vec(v: Iterable[float]) -> str:
    return " ".join(f"{float(x):.6g}" for x in v)


def build_ocean_scene_xml(spec: OceanSceneSpec) -> str:
    # MuJoCo requires all hfield size parameters to be positive; use geom `pos` to shift below z=0.
    hfield_size = (spec.x_half_m, spec.y_half_m, spec.terrain_height_m, 0.1)
    water_pos = (0.0, 0.0, -0.5 * spec.water_depth_m)
    water_size = (spec.x_half_m, spec.y_half_m, 0.5 * spec.water_depth_m)

    lines: list[str] = []
    lines.append(f'<mujoco model="{spec.model_name}">')
    lines.append(
        f'  <option timestep="{spec.dt_sec}" gravity="0 0 0" integrator="Euler" />'
    )
    lines.append("  <visual>")
    lines.append('    <global offwidth="1920" offheight="1080" />')
    lines.append('    <quality shadowsize="2048" />')
    lines.append("  </visual>")
    lines.append("  <asset>")
    lines.append(
        "    "
        f'<hfield name="bathy" '
        f'file="{spec.heightfield_png.as_posix()}" '
        f'nrow="{spec.heightfield_rows}" ncol="{spec.heightfield_cols}" '
        f'size="{_fmt_vec(hfield_size)}" />'
    )
    lines.append("  </asset>")
    lines.append("  <worldbody>")
    lines.append('    <light name="sun" diffuse="1 1 1" pos="0 0 30" dir="0 0 -1" />')
    lines.append(
        f'    <camera name="cam_main" pos="0 {-spec.camera_distance_m:.6g} {spec.camera_elevation_m:.6g}" '
        'xyaxes="1 0 0 0 0.65 0.76" />'
    )
    lines.append(
        f'    <geom name="seafloor" type="hfield" hfield="bathy" pos="0 0 {spec.terrain_base_z_m:.6g}" '
        'rgba="0.08 0.18 0.18 1" />'
    )
    lines.append(
        '    <geom name="water_volume" type="box" '
        f'pos="{_fmt_vec(water_pos)}" size="{_fmt_vec(water_size)}" '
        'rgba="0.05 0.25 0.5 0.12" contype="0" conaffinity="0" />'
    )

    for obstacle in spec.obstacles:
        if obstacle.shape not in {"sphere", "box", "cylinder"}:
            raise ValueError(f"Unsupported obstacle shape: {obstacle.shape}")
        lines.append(
            "    "
            f'<geom name="{obstacle.name}" type="{obstacle.shape}" '
            f'pos="{_fmt_vec(obstacle.pos_xyz_m)}" size="{_fmt_vec(obstacle.size_xyz_m)}" '
            f'rgba="{_fmt_rgba(obstacle.rgba)}" />'
        )

    for agent in spec.agents:
        body = agent.name
        lines.append(f'    <body name="{body}" pos="0 0 -3.5">')
        lines.append(
            f'      <joint name="{body}_x" type="slide" axis="1 0 0" '
            f'range="{-spec.x_half_m:.6g} {spec.x_half_m:.6g}" limited="true" damping="4" />'
        )
        lines.append(
            f'      <joint name="{body}_y" type="slide" axis="0 1 0" '
            f'range="{-spec.y_half_m:.6g} {spec.y_half_m:.6g}" limited="true" damping="4" />'
        )
        lines.append(
            f'      <joint name="{body}_z" type="slide" axis="0 0 1" '
            f'range="-12 -1.5" limited="true" damping="8" />'
        )
        lines.append(
            f'      <joint name="{body}_yaw" type="hinge" axis="0 0 1" damping="2" />'
        )
        # Vehicle shape: capsule aligned with local x-axis.
        lines.append(
            f'      <geom name="{body}_geom" type="capsule" '
            f'fromto="{-0.5 * agent.length_m:.6g} 0 0 {0.5 * agent.length_m:.6g} 0 0" '
            f'size="{agent.radius_m:.6g}" mass="{agent.mass_kg:.6g}" rgba="{_fmt_rgba(agent.rgba)}" />'
        )
        lines.append(f'      <site name="{body}_sensor" type="sphere" size="0.08" pos="0 0 0" rgba="1 1 0 0.8" />')
        lines.append("    </body>")

    # Movable, non-colliding markers.
    lines.append('    <body name="goal_marker" mocap="true" pos="0 0 -3.5">')
    lines.append('      <geom name="goal_geom" type="sphere" size="0.55" rgba="0.1 0.9 0.2 0.7" contype="0" conaffinity="0" />')
    lines.append("    </body>")
    lines.append('    <body name="source_marker" mocap="true" pos="0 0 -3.5">')
    lines.append('      <geom name="source_geom" type="sphere" size="0.55" rgba="0.95 0.25 0.1 0.7" contype="0" conaffinity="0" />')
    lines.append("    </body>")

    lines.append("  </worldbody>")
    lines.append("  <actuator>")
    for agent in spec.agents:
        body = agent.name
        lines.append(
            f'    <motor name="{body}_fx" joint="{body}_x" gear="1" ctrlrange="-160 160" ctrllimited="true" />'
        )
        lines.append(
            f'    <motor name="{body}_fy" joint="{body}_y" gear="1" ctrlrange="-160 160" ctrllimited="true" />'
        )
        lines.append(
            f'    <motor name="{body}_fz" joint="{body}_z" gear="1" ctrlrange="-220 220" ctrllimited="true" />'
        )
        lines.append(
            f'    <motor name="{body}_tau_yaw" joint="{body}_yaw" gear="1" ctrlrange="-25 25" ctrllimited="true" />'
        )
    lines.append("  </actuator>")
    lines.append("</mujoco>")
    return "\n".join(lines)
