from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class ObstacleSpec:
    name: str
    shape: str  # "sphere" | "box" | "cylinder" | "ellipsoid"
    pos_xyz_m: tuple[float, float, float]
    size_xyz_m: tuple[float, float, float]
    rgba: tuple[float, float, float, float] = (0.55, 0.55, 0.55, 1.0)
    material: str | None = None
    quat_wxyz: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class AgentSpec:
    name: str
    rgba: tuple[float, float, float, float] = (0.85, 0.25, 0.25, 1.0)
    radius_m: float = 0.45
    length_m: float = 1.2
    mass_kg: float = 40.0
    mesh_obj: Path | None = None
    mesh_scale_xyz: tuple[float, float, float] = (1.0, 1.0, 1.0)
    mesh_quat_wxyz: tuple[float, float, float, float] = (0.70710678, 0.0, 0.70710678, 0.0)


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
    sand_texture: Path | None = None
    rock_texture: Path | None = None
    particle_count: int = 180
    particle_seed: int = 0


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
    cam_main_z = -max(2.5, 0.18 * float(spec.water_depth_m))
    cam_low_z = -max(4.0, 0.28 * float(spec.water_depth_m))

    lines: list[str] = []
    lines.append(f'<mujoco model="{spec.model_name}">')
    lines.append(
        f'  <option timestep="{spec.dt_sec}" gravity="0 0 0" integrator="Euler" />'
    )
    lines.append("  <visual>")
    lines.append('    <global offwidth="1920" offheight="1080" />')
    lines.append('    <quality shadowsize="2048" offsamples="4" />')
    lines.append(
        '    <headlight ambient="0.16 0.20 0.24" diffuse="0.30 0.40 0.48" specular="0.02 0.03 0.04" active="1" />'
    )
    lines.append('    <map fogstart="0.10" fogend="1.10" haze="0.62" znear="0.01" zfar="4.0" />')
    lines.append('    <rgba fog="0.01 0.07 0.13 1" haze="0.02 0.09 0.18 1" />')
    lines.append("  </visual>")
    lines.append("  <asset>")
    lines.append(
        '    <texture name="skybox" type="skybox" builtin="gradient" '
        'rgb1="0.01 0.05 0.10" rgb2="0.00 0.22 0.34" width="512" height="512" />'
    )
    sand_tex = spec.sand_texture if spec.sand_texture is not None and spec.sand_texture.exists() else None
    rock_tex = spec.rock_texture if spec.rock_texture is not None and spec.rock_texture.exists() else None
    if sand_tex is not None:
        lines.append(
            f'    <texture name="tex_sand" type="2d" file="{sand_tex.as_posix()}" />'
        )
    else:
        lines.append(
            '    <texture name="tex_sand" type="2d" builtin="checker" '
            'rgb1="0.20 0.18 0.14" rgb2="0.32 0.28 0.20" width="512" height="512" />'
        )
    if rock_tex is not None:
        lines.append(
            f'    <texture name="tex_rock" type="2d" file="{rock_tex.as_posix()}" />'
        )
    else:
        lines.append(
            '    <texture name="tex_rock" type="2d" builtin="checker" '
            'rgb1="0.24 0.26 0.28" rgb2="0.35 0.38 0.42" width="512" height="512" />'
        )
    lines.append('    <material name="mat_sand" texture="tex_sand" texrepeat="16 16" rgba="1 1 1 1" />')
    lines.append('    <material name="mat_rock" texture="tex_rock" texrepeat="4 4" rgba="1 1 1 1" />')

    for agent in spec.agents:
        if agent.mesh_obj is None:
            continue
        if not agent.mesh_obj.exists():
            raise FileNotFoundError(f"Agent mesh not found: {agent.mesh_obj}")
        lines.append(
            "    "
            f'<mesh name="{agent.name}_mesh" file="{agent.mesh_obj.as_posix()}" '
            f'scale="{_fmt_vec(agent.mesh_scale_xyz)}" />'
        )

    lines.append(
        "    "
        f'<hfield name="bathy" '
        f'file="{spec.heightfield_png.as_posix()}" '
        f'nrow="{spec.heightfield_rows}" ncol="{spec.heightfield_cols}" '
        f'size="{_fmt_vec(hfield_size)}" />'
    )
    lines.append("  </asset>")
    lines.append("  <worldbody>")
    lines.append('    <light name="sun" diffuse="0.90 0.96 1.0" ambient="0.10 0.14 0.18" pos="0 0 30" dir="0 0 -1" />')
    lines.append('    <light name="fill" diffuse="0.35 0.55 0.65" ambient="0.06 0.08 0.10" pos="25 -20 12" dir="-1 1 -0.6" />')
    lines.append(
        f'    <camera name="cam_main" pos="0 {-spec.camera_distance_m:.6g} {cam_main_z:.6g}" '
        'xyaxes="1 0 0 0 0.65 0.76" />'
    )
    lines.append(
        f'    <camera name="cam_low" pos="{0.55 * spec.x_half_m:.6g} {-0.35 * spec.y_half_m:.6g} {cam_low_z:.6g}" '
        'xyaxes="-0.8 0.6 0 0 0.35 0.94" />'
    )
    lines.append(
        f'    <geom name="seafloor" type="hfield" hfield="bathy" pos="0 0 {spec.terrain_base_z_m:.6g}" '
        'material="mat_sand" rgba="0.9 0.9 0.9 1" />'
    )

    for obstacle in spec.obstacles:
        if obstacle.shape not in {"sphere", "box", "cylinder", "ellipsoid"}:
            raise ValueError(f"Unsupported obstacle shape: {obstacle.shape}")
        attrs = []
        if obstacle.material is not None:
            attrs.append(f'material="{obstacle.material}"')
        if obstacle.quat_wxyz is not None:
            attrs.append(f'quat="{_fmt_vec(obstacle.quat_wxyz)}"')
        if obstacle.shape == "sphere":
            size_str = f"{float(obstacle.size_xyz_m[0]):.6g}"
        elif obstacle.shape == "cylinder":
            size_str = f"{float(obstacle.size_xyz_m[0]):.6g} {float(obstacle.size_xyz_m[1]):.6g}"
        else:
            size_str = _fmt_vec(obstacle.size_xyz_m)
        lines.append(
            "    "
            f'<geom name="{obstacle.name}" type="{obstacle.shape}" '
            f'pos="{_fmt_vec(obstacle.pos_xyz_m)}" size="{size_str}" '
            f'rgba="{_fmt_rgba(obstacle.rgba)}" {" ".join(attrs)} />'
        )

    if int(spec.particle_count) > 0:
        rng = np.random.default_rng(int(spec.particle_seed))
        for i in range(int(spec.particle_count)):
            x = float(rng.uniform(-0.75 * spec.x_half_m, 0.75 * spec.x_half_m))
            y = float(rng.uniform(-0.75 * spec.y_half_m, 0.75 * spec.y_half_m))
            z = float(rng.uniform(-0.9 * spec.water_depth_m, -0.8))
            radius = float(rng.uniform(0.045, 0.12))
            alpha = float(rng.uniform(0.08, 0.22))
            lines.append(
                "    "
                f'<geom name="particle{i:03d}" type="sphere" pos="{_fmt_vec((x, y, z))}" '
                f'size="{radius:.6g}" rgba="0.82 0.92 1.0 {alpha:.4g}" contype="0" conaffinity="0" />'
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
        if agent.mesh_obj is not None:
            lines.append(
                f'      <geom name="{body}_visual" type="mesh" mesh="{body}_mesh" '
                f'quat="{_fmt_vec(agent.mesh_quat_wxyz)}" rgba="{_fmt_rgba(agent.rgba)}" '
                'contype="0" conaffinity="0" />'
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
