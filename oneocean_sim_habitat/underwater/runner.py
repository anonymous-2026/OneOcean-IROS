from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Optional

import cv2
import imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import habitat_sim
import magnum as mn
from habitat_sim.utils.common import (
    quat_from_angle_axis,
    quat_from_two_vectors,
    quat_to_magnum,
)

from ..drift import CachedDriftField, DriftConfig, sample_drift_xz


@dataclass
class CameraConfig:
    width: int = 960
    height: int = 540
    orbit_radius_m: float = 18.0
    orbit_height_m: float = 10.0
    orbit_period_steps: int = 240


@dataclass
class UnderwaterRunConfig:
    stage_obj: str
    stage_meta: str
    base_scene: Optional[str] = None
    output_dir: Optional[str] = None
    invocation: str = ""
    seed: int = 0
    dt_s: float = 1.0
    max_steps_task1: int = 240
    max_steps_task2: int = 320
    write_video: bool = True
    video_fps: float = 24.0
    gif_stride: int = 2
    camera: CameraConfig = field(default_factory=CameraConfig)

    drift_cache_path: Optional[str] = None
    drift_origin_lat: Optional[float] = None
    drift_origin_lon: Optional[float] = None
    synthetic_drift_amplitude_mps: float = 0.25
    current_gain: float = 1.0

    vehicle_speed_mps: float = 1.4
    vehicle_max_vertical_mps: float = 0.8
    vehicle_clearance_m: float = 0.6

    plume_sigma_m: float = 220.0
    plume_advect_gain: float = 0.8
    plume_diffusion_m2ps: float = 0.35
    plume_cloud_scale: float = 0.10
    success_radius_m: float = 2.5

    formation_radius_m: float = 7.0
    formation_kp: float = 0.55
    formation_success_error_m: float = 1.25
    formation_success_hold_steps: int = 60

    rock_model_gltf: Optional[str] = None
    rock_count: int = 10
    rock_scale_min: float = 0.9
    rock_scale_max: float = 1.7

    silt_particle_count: int = 90
    silt_particle_scale: float = 0.035
    silt_advect_gain: float = 0.35
    silt_rise_mps: float = 0.015

    plume_particle_count: int = 45
    plume_particle_scale: float = 0.060

    underwater_fog_density: float = 0.065
    underwater_water_rgb: tuple[int, int, int] = (10, 55, 80)
    underwater_attenuation_rgb: tuple[float, float, float] = (0.14, 0.07, 0.03)


def _default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"oneocean_habitat_s2_underwater_{stamp}"


def _load_json(path: str | Path) -> dict[str, Any]:
    p = Path(path).expanduser().resolve()
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _normalize_rgb(rgba: np.ndarray) -> np.ndarray:
    frame = np.asarray(rgba)
    if frame.ndim != 3:
        raise ValueError(f"Unexpected frame shape: {frame.shape}")
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0.0, 255.0).astype(np.uint8)
    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    return frame


def _normalize_depth(depth: np.ndarray) -> np.ndarray:
    d = np.asarray(depth)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[..., 0]
    if d.ndim != 2:
        raise ValueError(f"Unexpected depth shape: {d.shape}")
    d = d.astype(np.float32)
    return np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)


def _apply_underwater_postprocess(
    rgb: np.ndarray,
    depth_m: np.ndarray,
    *,
    fog_density: float,
    water_rgb: tuple[int, int, int],
    attenuation_rgb: tuple[float, float, float],
) -> np.ndarray:
    """Cheap underwater look: depth fog + wavelength attenuation (postprocess)."""
    rgb_f = rgb.astype(np.float32) / 255.0
    d = np.clip(depth_m.astype(np.float32), 0.0, 250.0)

    fog = 1.0 - np.exp(-d * float(fog_density))
    fog = fog[..., None]

    r_att, g_att, b_att = (float(attenuation_rgb[0]), float(attenuation_rgb[1]), float(attenuation_rgb[2]))
    att = np.stack(
        [
            np.exp(-d * r_att),
            np.exp(-d * g_att),
            np.exp(-d * b_att),
        ],
        axis=-1,
    )
    rgb_f = rgb_f * att

    water = np.array(water_rgb, dtype=np.float32) / 255.0
    out = rgb_f * (1.0 - fog) + water * fog

    out = np.clip(out * 255.0, 0.0, 255.0).astype(np.uint8)
    return out


def _draw_hud(
    rgb: np.ndarray,
    lines: list[str],
    drift_x_mps: float,
    drift_z_mps: float,
) -> np.ndarray:
    img = Image.fromarray(rgb)
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", size=16)
    except Exception:
        font = ImageFont.load_default()

    pad = 10
    x0, y0 = pad, pad
    box_w = 420
    box_h = pad + 20 * (len(lines) + 2)
    draw.rectangle([x0 - 6, y0 - 6, x0 + box_w, y0 + box_h], fill=(0, 0, 0))

    for i, line in enumerate(lines):
        draw.text((x0, y0 + 20 * i), line, font=font, fill=(235, 235, 235))

    # Drift arrow (in screen space)
    arrow_origin = (x0 + 40, y0 + 20 * len(lines) + 40)
    scale = 85.0
    dx = float(drift_x_mps) * scale
    dz = float(drift_z_mps) * scale
    arrow_tip = (arrow_origin[0] + dx, arrow_origin[1] + dz)
    draw.line([arrow_origin, arrow_tip], fill=(80, 200, 255), width=4)
    draw.ellipse(
        [arrow_origin[0] - 4, arrow_origin[1] - 4, arrow_origin[0] + 4, arrow_origin[1] + 4],
        fill=(80, 200, 255),
    )
    draw.text(
        (arrow_origin[0] + 10, arrow_origin[1] + 10),
        f"current (x,z)=({drift_x_mps:+.2f},{drift_z_mps:+.2f}) m/s",
        font=font,
        fill=(80, 200, 255),
    )
    return np.asarray(img)


def _setup_sim(scene_id: Path, camera: CameraConfig) -> tuple[habitat_sim.Simulator, habitat_sim.Agent]:
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = "default"
    sim_cfg.scene_id = str(scene_id)
    sim_cfg.enable_physics = True
    sim_cfg.override_scene_light_defaults = True
    sim_cfg.scene_light_setup = "oneocean_underwater_lights"

    rgb_spec = habitat_sim.CameraSensorSpec()
    rgb_spec.uuid = "rgb"
    rgb_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_spec.resolution = [int(camera.height), int(camera.width)]
    rgb_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    rgb_spec.position = [0.0, 0.0, 0.0]

    depth_spec = habitat_sim.CameraSensorSpec()
    depth_spec.uuid = "depth"
    depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_spec.resolution = [int(camera.height), int(camera.width)]
    depth_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
    depth_spec.position = [0.0, 0.0, 0.0]

    agent_cfg = habitat_sim.AgentConfiguration()
    agent_cfg.sensor_specifications = [rgb_spec, depth_spec]

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    key = str(sim_cfg.scene_light_setup)
    lights: list[habitat_sim.gfx.LightInfo] = []

    # Soft bluish key light (directional).
    key_light = habitat_sim.gfx.LightInfo()
    key_light.vector = mn.Vector4(0.35, -1.0, 0.25, 0.0)
    key_light.color = mn.Color3(0.78, 0.92, 1.00)
    key_light.model = habitat_sim.gfx.LightPositionModel.Global
    lights.append(key_light)

    # Fill light (directional, dimmer).
    fill = habitat_sim.gfx.LightInfo()
    fill.vector = mn.Vector4(-0.55, -1.0, -0.10, 0.0)
    fill.color = mn.Color3(0.40, 0.62, 0.78)
    fill.model = habitat_sim.gfx.LightPositionModel.Global
    lights.append(fill)

    # Point light to lift shadows around the origin.
    point = habitat_sim.gfx.LightInfo()
    point.vector = mn.Vector4(0.0, 2.5, 0.0, 1.0)
    point.color = mn.Color3(0.22, 0.40, 0.52)
    point.model = habitat_sim.gfx.LightPositionModel.Global
    lights.append(point)

    # Register the setup under a custom key (avoid overriding 'no_lights').
    sim.set_light_setup(lights, key=key)
    agent = sim.initialize_agent(0)
    return sim, agent


def _register_mesh_template(sim: habitat_sim.Simulator, handle: str, mesh_path: Path, scale: float) -> int:
    otm = sim.get_object_template_manager()
    attr = otm.create_new_template(handle)
    attr.render_asset_handle = str(mesh_path)
    attr.collision_asset_handle = str(mesh_path)
    attr.scale = np.array([scale, scale, scale], dtype=np.float32)
    return int(otm.register_template(attr))


def _spawn_object(
    sim: habitat_sim.Simulator,
    template_id: int,
    translation: np.ndarray,
    motion_type: habitat_sim.physics.MotionType = habitat_sim.physics.MotionType.KINEMATIC,
) -> habitat_sim.physics.ManagedRigidObject:
    rom = sim.get_rigid_object_manager()
    obj = rom.add_object_by_template_id(int(template_id))
    obj.motion_type = motion_type
    obj.translation = np.asarray(translation, dtype=np.float32)
    return obj


def _yaw_quat_magnum(yaw_rad: float) -> Any:
    q = quat_from_angle_axis(float(yaw_rad), np.array([0.0, 1.0, 0.0], dtype=np.float32))
    return quat_to_magnum(q)


def _camera_state_from_orbit(step: int, center: np.ndarray, camera: CameraConfig) -> habitat_sim.AgentState:
    theta = (2.0 * math.pi) * (float(step) / max(1, int(camera.orbit_period_steps)))
    cam_pos = np.asarray(
        [
            float(center[0] + camera.orbit_radius_m * math.cos(theta)),
            float(center[1] + camera.orbit_height_m),
            float(center[2] + camera.orbit_radius_m * math.sin(theta)),
        ],
        dtype=np.float32,
    )
    forward = np.asarray(center, dtype=np.float32) - cam_pos
    norm = float(np.linalg.norm(forward))
    if norm < 1e-6:
        forward = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    else:
        forward = forward / norm

    # Habitat cameras look along -Z by default; rotate that to the desired forward direction.
    q = quat_from_two_vectors(np.array([0.0, 0.0, -1.0], dtype=np.float32), forward)
    state = habitat_sim.AgentState()
    state.position = cam_pos
    state.rotation = q
    return state


def _sample_drift_sim_xz(
    drift_field: Optional[CachedDriftField],
    position_xyz_sim: np.ndarray,
    step: int,
    horizontal_scale: float,
    origin_lat: Optional[float],
    origin_lon: Optional[float],
    synthetic_amplitude_mps: float,
    current_gain: float,
) -> tuple[float, float, str]:
    if drift_field is None or origin_lat is None or origin_lon is None:
        # Fall back to synthetic drift in *sim units*.
        drift_x, drift_z = sample_drift_xz(
            position_xyz=position_xyz_sim,
            step_index=step,
            config=DriftConfig(
                mode="synthetic_wave",
                amplitude_mps=float(synthetic_amplitude_mps) * float(current_gain),
                spatial_scale_m=8.0,
                temporal_scale_steps=28.0,
                bias_x_mps=0.0,
                bias_z_mps=0.0,
            ),
        )
        return float(drift_x), float(drift_z), "synthetic"

    scale = float(horizontal_scale)
    inv = 1.0 / max(1e-9, scale)
    x_real = float(position_xyz_sim[0]) * inv
    z_real = float(position_xyz_sim[2]) * inv
    drift_x_real, drift_z_real = drift_field.sample_xz(
        x_m=x_real, z_m=z_real, origin_lat=float(origin_lat), origin_lon=float(origin_lon)
    )
    gain = float(current_gain)
    # Convert real m/s into sim units using the same horizontal_scale that mapped meters -> sim.
    drift_x_sim = float(drift_x_real * gain * scale)
    drift_z_sim = float(drift_z_real * gain * scale)
    return drift_x_sim, drift_z_sim, f"cache:{drift_field.path}"


def _seafloor_y_sim(
    drift_field: Optional[CachedDriftField],
    position_xyz_sim: np.ndarray,
    horizontal_scale: float,
    origin_lat: Optional[float],
    origin_lon: Optional[float],
    elevation_max_m: float,
    vertical_scale: float,
    floor_offset_m: float,
) -> Optional[float]:
    if drift_field is None or drift_field.elevation is None or origin_lat is None or origin_lon is None:
        return None
    inv = 1.0 / max(1e-9, float(horizontal_scale))
    elev_m = drift_field.sample_elevation_xz(
        x_m=float(position_xyz_sim[0]) * inv,
        z_m=float(position_xyz_sim[2]) * inv,
        origin_lat=float(origin_lat),
        origin_lon=float(origin_lon),
    )
    if elev_m is None:
        return None
    return float((float(elev_m) - float(elevation_max_m)) * float(vertical_scale) - float(floor_offset_m))


def _clamp_to_stage_bounds(pos: np.ndarray, ext: dict[str, Any]) -> np.ndarray:
    out = np.asarray(pos, dtype=np.float32).copy()
    out[0] = float(np.clip(out[0], float(ext["x_min"]), float(ext["x_max"])))
    out[2] = float(np.clip(out[2], float(ext["z_min"]), float(ext["z_max"])))
    return out


def _plume_concentration(pos: np.ndarray, center: np.ndarray, sigma_m: float) -> float:
    """Simple plume proxy: exponential falloff around a drifting hotspot."""
    sigma = max(1e-6, float(sigma_m))
    d = np.asarray(pos - center, dtype=np.float32)
    r = float(np.linalg.norm(d))
    return float(math.exp(-r / sigma))


def _plume_grad_xz(pos: np.ndarray, center: np.ndarray, sigma_m: float, delta: float = 0.6) -> np.ndarray:
    base = _plume_concentration(pos, center, sigma_m)
    dx_pos = pos + np.array([delta, 0.0, 0.0], dtype=np.float32)
    dz_pos = pos + np.array([0.0, 0.0, delta], dtype=np.float32)
    gx = (_plume_concentration(dx_pos, center, sigma_m) - base) / delta
    gz = (_plume_concentration(dz_pos, center, sigma_m) - base) / delta
    return np.array([gx, 0.0, gz], dtype=np.float32)


def _vehicle_controller_plume(
    pos: np.ndarray,
    plume_center: np.ndarray,
    speed: float,
    rng: np.random.Generator,
    sigma_m: float,
) -> np.ndarray:
    grad = _plume_grad_xz(pos, plume_center, sigma_m=sigma_m)
    gnorm = float(np.linalg.norm(grad[[0, 2]]))
    if gnorm < 1e-5:
        # Low signal; explore.
        angle = float(rng.uniform(0.0, 2.0 * math.pi))
        direction = np.array([math.cos(angle), 0.0, math.sin(angle)], dtype=np.float32)
    else:
        direction = grad / max(1e-6, float(np.linalg.norm(grad)))
    return np.asarray(direction, dtype=np.float32) * float(speed)


def _vehicle_controller_formation(
    pos: np.ndarray,
    desired: np.ndarray,
    kp: float,
    max_speed: float,
) -> np.ndarray:
    err = np.asarray(desired - pos, dtype=np.float32)
    vel = float(kp) * err
    speed = float(np.linalg.norm(vel[[0, 2]]))
    if speed > float(max_speed):
        vel = vel * (float(max_speed) / max(1e-6, speed))
    vel[1] = 0.0
    return vel


def run_underwater_tasks(config: UnderwaterRunConfig) -> dict[str, Any]:
    stage_obj = Path(config.stage_obj).expanduser().resolve()
    stage_meta_path = Path(config.stage_meta).expanduser().resolve()
    if not stage_obj.exists():
        raise FileNotFoundError(f"Stage OBJ not found: {stage_obj}")
    if not stage_meta_path.exists():
        raise FileNotFoundError(f"Stage meta not found: {stage_meta_path}")

    stage_meta = _load_json(stage_meta_path)
    horizontal_scale = float(stage_meta.get("horizontal_scale", 1.0))
    vertical_scale = float(stage_meta.get("vertical_scale", 1.0))
    floor_offset_m = float(stage_meta.get("floor_offset_m", 0.0))
    origin_lat, origin_lon = None, None
    if "origin_latlon_deg" in stage_meta and isinstance(stage_meta["origin_latlon_deg"], list):
        try:
            origin_lat = float(stage_meta["origin_latlon_deg"][0])
            origin_lon = float(stage_meta["origin_latlon_deg"][1])
        except Exception:
            origin_lat, origin_lon = None, None

    elevation_max_m = float(stage_meta.get("elevation_max_m", 0.0))
    extents = stage_meta.get("extents_sim_m") or {}
    if not extents:
        raise ValueError("Stage meta missing extents_sim_m")

    output_dir = Path(config.output_dir) if config.output_dir else _default_output_dir()
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    task1_dir = output_dir / "task_plume_localization"
    task2_dir = output_dir / "task_plume_containment_multiagent"
    task1_dir.mkdir(parents=True, exist_ok=True)
    task2_dir.mkdir(parents=True, exist_ok=True)

    scene_id = stage_obj
    if config.base_scene:
        base_scene = Path(config.base_scene).expanduser().resolve()
        if not base_scene.exists():
            raise FileNotFoundError(f"Base scene not found: {base_scene}")
        scene_id = base_scene

    sim, cam_agent = _setup_sim(scene_id=scene_id, camera=config.camera)

    drift_field: Optional[CachedDriftField] = None
    drift_origin_lat = config.drift_origin_lat if config.drift_origin_lat is not None else origin_lat
    drift_origin_lon = config.drift_origin_lon if config.drift_origin_lon is not None else origin_lon
    if config.drift_cache_path:
        drift_field = CachedDriftField(config.drift_cache_path)
        if drift_origin_lat is None or drift_origin_lon is None:
            drift_origin_lat, drift_origin_lon = drift_field.center_latlon()

    assets_root = Path(__file__).resolve().parent.parent / "assets" / "meshes"
    uuv_red = assets_root / "uuv_red.obj"
    uuv_green = assets_root / "uuv_green.obj"
    uuv_blue = assets_root / "uuv_blue.obj"
    marker = assets_root / "marker_yellow.obj"
    particle_silt = assets_root / "particle_silt.obj"
    particle_plume = assets_root / "particle_plume.obj"
    for p in (uuv_red, uuv_green, uuv_blue, marker, particle_silt, particle_plume):
        if not p.exists():
            sim.close()
            raise FileNotFoundError(f"Missing mesh asset: {p}")

    tpl_red = _register_mesh_template(sim, "uuv_red", uuv_red, scale=1.0)
    tpl_green = _register_mesh_template(sim, "uuv_green", uuv_green, scale=1.0)
    tpl_blue = _register_mesh_template(sim, "uuv_blue", uuv_blue, scale=1.0)
    tpl_marker = _register_mesh_template(sim, "marker_yellow", marker, scale=1.0)
    tpl_silt = _register_mesh_template(sim, "particle_silt", particle_silt, scale=float(config.silt_particle_scale))
    tpl_plume = _register_mesh_template(sim, "particle_plume", particle_plume, scale=float(config.plume_particle_scale))

    rng = np.random.default_rng(int(config.seed))

    def seafloor_y(pos_sim: np.ndarray) -> Optional[float]:
        return _seafloor_y_sim(
            drift_field=drift_field,
            position_xyz_sim=pos_sim,
            horizontal_scale=horizontal_scale,
            origin_lat=drift_origin_lat,
            origin_lon=drift_origin_lon,
            elevation_max_m=elevation_max_m,
            vertical_scale=vertical_scale,
            floor_offset_m=floor_offset_m,
        )

    def apply_depth_clamp(pos_sim: np.ndarray) -> tuple[np.ndarray, float]:
        y_floor = seafloor_y(pos_sim)
        collided = 0.0
        out = np.asarray(pos_sim, dtype=np.float32).copy()
        if y_floor is not None:
            min_y = float(y_floor + float(config.vehicle_clearance_m))
            if float(out[1]) < min_y:
                out[1] = min_y
                collided = 1.0
        # Keep underwater (negative y) for visuals.
        out[1] = float(min(out[1], -0.4))
        return out, collided

    def clamp_above_seafloor(pos_sim: np.ndarray, *, clearance_m: float) -> np.ndarray:
        out = np.asarray(pos_sim, dtype=np.float32).copy()
        y_floor = seafloor_y(out)
        if y_floor is not None:
            out[1] = float(max(out[1], float(y_floor) + float(clearance_m)))
        out[1] = float(min(out[1], -0.55))
        return out

    x_min, x_max = float(extents["x_min"]), float(extents["x_max"])
    z_min, z_max = float(extents["z_min"]), float(extents["z_max"])
    y_min, y_max = float(extents["y_min"]), float(extents["y_max"])

    stage_tpl: Optional[int] = None
    if config.base_scene:
        stage_tpl = _register_mesh_template(sim, "underwater_stage_obj", stage_obj, scale=1.0)

    rock_path: Optional[Path] = None
    if config.rock_model_gltf:
        rock_path = Path(config.rock_model_gltf).expanduser().resolve()
        if not rock_path.exists():
            sim.close()
            raise FileNotFoundError(f"Rock model not found: {rock_path}")

    env_rng = np.random.default_rng(int(config.seed) + 12345)
    rock_specs: list[dict[str, Any]] = []
    if rock_path is not None and int(config.rock_count) > 0:
        for i in range(int(config.rock_count)):
            scale = float(env_rng.uniform(float(config.rock_scale_min), float(config.rock_scale_max)))
            tpl = _register_mesh_template(sim, f"rock_{i:02d}", rock_path, scale=scale)
            x = float(env_rng.uniform(x_min, x_max))
            z = float(env_rng.uniform(z_min, z_max))
            pos = np.array([x, -2.0, z], dtype=np.float32)
            y_floor = seafloor_y(pos)
            base_y = float(y_floor) if y_floor is not None else float(y_min)
            pos[1] = float(min(base_y + 0.10, -0.85))
            rock_specs.append(
                {
                    "template_id": int(tpl),
                    "translation": pos,
                    "yaw_rad": float(env_rng.uniform(0.0, 2.0 * math.pi)),
                }
            )

    silt_init_positions: list[np.ndarray] = []
    silt_phases: list[float] = []
    for i in range(int(config.silt_particle_count)):
        x = float(env_rng.uniform(x_min, x_max))
        z = float(env_rng.uniform(z_min, z_max))
        pos = np.array([x, -2.0, z], dtype=np.float32)
        y_floor = seafloor_y(pos)
        base_y = float(y_floor) if y_floor is not None else float(y_min)
        pos[1] = float(min(base_y + float(env_rng.uniform(1.0, 9.0)), -0.85))
        silt_init_positions.append(pos)
        silt_phases.append(float(env_rng.uniform(0.0, 2.0 * math.pi)))

    def spawn_environment() -> dict[str, Any]:
        env: dict[str, Any] = {"stage_obj": None, "rocks": [], "silt": []}
        if stage_tpl is not None:
            env["stage_obj"] = _spawn_object(
                sim,
                int(stage_tpl),
                translation=np.array([0.0, 0.0, 0.0], dtype=np.float32),
                motion_type=habitat_sim.physics.MotionType.STATIC,
            )
        for spec in rock_specs:
            obj = _spawn_object(
                sim,
                int(spec["template_id"]),
                translation=np.asarray(spec["translation"], dtype=np.float32),
                motion_type=habitat_sim.physics.MotionType.STATIC,
            )
            obj.rotation = _yaw_quat_magnum(float(spec["yaw_rad"]))
            env["rocks"].append(obj)
        for pos in silt_init_positions:
            obj = _spawn_object(
                sim,
                int(tpl_silt),
                translation=np.asarray(pos, dtype=np.float32),
                motion_type=habitat_sim.physics.MotionType.KINEMATIC,
            )
            env["silt"].append(obj)
        return env

    def step_silt_particles(step: int, silt_objs: list[habitat_sim.physics.ManagedRigidObject]) -> None:
        for idx, obj in enumerate(silt_objs):
            pos = np.asarray(obj.translation, dtype=np.float32)
            drift_x, drift_z, _ = _sample_drift_sim_xz(
                drift_field,
                pos,
                step,
                horizontal_scale,
                drift_origin_lat,
                drift_origin_lon,
                config.synthetic_drift_amplitude_mps,
                config.current_gain,
            )
            pos[0] = float(pos[0] + float(drift_x) * float(config.dt_s) * float(config.silt_advect_gain))
            pos[2] = float(pos[2] + float(drift_z) * float(config.dt_s) * float(config.silt_advect_gain))
            pos[1] = float(
                pos[1]
                + float(config.silt_rise_mps) * float(config.dt_s)
                + 0.10 * math.sin(float(silt_phases[idx]) + 0.18 * float(step))
            )

            if float(pos[0]) < x_min:
                pos[0] = float(x_max)
            elif float(pos[0]) > x_max:
                pos[0] = float(x_min)
            if float(pos[2]) < z_min:
                pos[2] = float(z_max)
            elif float(pos[2]) > z_max:
                pos[2] = float(z_min)

            pos = clamp_above_seafloor(pos, clearance_m=0.25)
            if float(pos[1]) > -0.60:
                pos = np.asarray(silt_init_positions[idx], dtype=np.float32).copy()
            obj.translation = pos

    def spawn_plume_cloud(
        plume_center: np.ndarray,
        *,
        rng: np.random.Generator,
        sigma_m: float,
    ) -> tuple[list[habitat_sim.physics.ManagedRigidObject], np.ndarray]:
        n = max(0, int(config.plume_particle_count))
        if n == 0:
            return [], np.zeros((0, 3), dtype=np.float32)
        offsets = rng.normal(loc=0.0, scale=1.0, size=(n, 3)).astype(np.float32)
        offsets[:, 1] = offsets[:, 1] * 0.35
        norms = np.linalg.norm(offsets, axis=1, keepdims=True)
        norms = np.clip(norms, 1e-6, None)
        radii = rng.uniform(0.15, 1.0, size=(n, 1)).astype(np.float32)
        offsets = offsets / norms * radii

        objs: list[habitat_sim.physics.ManagedRigidObject] = []
        scale = float(float(config.plume_cloud_scale) * max(1e-6, float(sigma_m)))
        for i in range(n):
            pos = np.asarray(plume_center, dtype=np.float32) + offsets[i] * scale
            pos = _clamp_to_stage_bounds(pos, extents)
            pos = clamp_above_seafloor(pos, clearance_m=0.25)
            objs.append(
                _spawn_object(
                    sim,
                    int(tpl_plume),
                    translation=pos,
                    motion_type=habitat_sim.physics.MotionType.KINEMATIC,
                )
            )
        return objs, offsets

    def update_plume_cloud(
        plume_objs: list[habitat_sim.physics.ManagedRigidObject],
        offsets: np.ndarray,
        plume_center: np.ndarray,
        *,
        sigma_m: float,
        step: int,
    ) -> None:
        if not plume_objs:
            return
        theta = 0.12 * float(step)
        c = float(math.cos(theta))
        s = float(math.sin(theta))
        scale = float(float(config.plume_cloud_scale) * max(1e-6, float(sigma_m)))
        for i, obj in enumerate(plume_objs):
            off = offsets[i]
            ox = float(off[0] * c - off[2] * s)
            oz = float(off[0] * s + off[2] * c)
            oy = float(off[1] + 0.08 * math.sin(0.25 * float(step) + float(i)))
            pos = np.asarray(plume_center, dtype=np.float32) + np.array([ox, oy, oz], dtype=np.float32) * scale
            pos = _clamp_to_stage_bounds(pos, extents)
            pos = clamp_above_seafloor(pos, clearance_m=0.25)
            obj.translation = pos

    def update_camera(step: int, focus: np.ndarray) -> None:
        state = _camera_state_from_orbit(step=step, center=focus, camera=config.camera)
        cam_agent.set_state(state)

    def render_frame(step: int, hud_lines: list[str], drift_x: float, drift_z: float) -> np.ndarray:
        obs = sim.get_sensor_observations()
        rgb = _normalize_rgb(obs["rgb"])
        depth = _normalize_depth(obs["depth"])
        rgb = _apply_underwater_postprocess(
            rgb,
            depth,
            fog_density=float(config.underwater_fog_density),
            water_rgb=tuple(config.underwater_water_rgb),
            attenuation_rgb=tuple(config.underwater_attenuation_rgb),
        )
        return _draw_hud(rgb, lines=hud_lines, drift_x_mps=drift_x, drift_z_mps=drift_z)

    def write_video_and_gif(
        frames: list[np.ndarray],
        mp4_path: Path,
        gif_path: Path,
    ) -> None:
        if not frames:
            return
        mp4_path.parent.mkdir(parents=True, exist_ok=True)
        gif_path.parent.mkdir(parents=True, exist_ok=True)

        if config.write_video:
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(
                str(mp4_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(1.0, float(config.video_fps)),
                (w, h),
            )
            try:
                for fr in frames:
                    writer.write(cv2.cvtColor(fr, cv2.COLOR_RGB2BGR))
            finally:
                writer.release()

        gif_frames = frames[:: max(1, int(config.gif_stride))]
        imageio.mimsave(gif_path, gif_frames, fps=max(1.0, float(config.video_fps) / max(1, int(config.gif_stride))))

    artifacts: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "invocation": str(config.invocation or ""),
        "output_dir": str(output_dir),
        "stage": {
            "scene_id": str(scene_id),
            "base_scene": str(Path(config.base_scene).expanduser().resolve()) if config.base_scene else "",
            "obj": str(stage_obj),
            "meta": str(stage_meta_path),
            "drift_cache": str(Path(config.drift_cache_path).resolve()) if config.drift_cache_path else "",
            "drift_origin_lat": drift_origin_lat,
            "drift_origin_lon": drift_origin_lon,
            "drift_source": "synthetic" if drift_field is None else f"cache:{drift_field.path}",
            "rock_model_gltf": str(Path(config.rock_model_gltf).expanduser().resolve())
            if config.rock_model_gltf
            else "",
            "underwater_postprocess": {
                "fog_density": float(config.underwater_fog_density),
                "water_rgb": list(config.underwater_water_rgb),
                "attenuation_rgb": list(config.underwater_attenuation_rgb),
            },
            "particles": {
                "silt_count": int(config.silt_particle_count),
                "plume_count": int(config.plume_particle_count),
            },
        },
        "tasks": [],
    }

    try:
        # -----------------------
        # Task 1: Plume localization (single agent)
        # -----------------------
        sim.get_rigid_object_manager().remove_all_objects()
        env1 = spawn_environment()

        # Choose a non-trivial but reachable start/target separation (avoid spanning the entire 1° scene).
        x_span = float(x_max - x_min)
        z_span = float(z_max - z_min)
        start_pos = np.array([x_min + 0.25 * x_span, -2.8, z_min + 0.25 * z_span], dtype=np.float32)
        start_pos, _ = apply_depth_clamp(start_pos)
        vehicle = _spawn_object(sim, tpl_red, translation=start_pos)

        # Initialize plume marker at an offset, then let currents advect it.
        plume_center = np.array([x_min + 0.35 * x_span, -3.3, z_min + 0.35 * z_span], dtype=np.float32)
        plume_center, _ = apply_depth_clamp(plume_center)
        plume_obj = _spawn_object(sim, tpl_marker, translation=plume_center)
        plume_rng1 = np.random.default_rng(int(config.seed) + 202)
        sigma0 = float(config.plume_sigma_m)
        plume_particles, plume_offsets = spawn_plume_cloud(plume_center, rng=plume_rng1, sigma_m=sigma0)

        frames: list[np.ndarray] = []
        traj_rows: list[dict[str, Any]] = []
        energy = 0.0
        path_len = 0.0
        collisions = 0.0
        success = 0.0
        t_success = None

        pos = start_pos.copy()
        update_camera(0, focus=plume_center)
        drift_x0, drift_z0, _ = _sample_drift_sim_xz(
            drift_field,
            pos,
            0,
            horizontal_scale,
            drift_origin_lat,
            drift_origin_lon,
            config.synthetic_drift_amplitude_mps,
            config.current_gain,
        )
        scene_png = task1_dir / "scene.png"
        Image.fromarray(render_frame(0, ["Task1: Plume localization", "step=0"], drift_x0, drift_z0)).save(scene_png)

        for step in range(int(config.max_steps_task1)):
            # Advect plume center by currents at its location.
            drift_px, drift_pz, _ = _sample_drift_sim_xz(
                drift_field,
                plume_center,
                step,
                horizontal_scale,
                drift_origin_lat,
                drift_origin_lon,
                config.synthetic_drift_amplitude_mps,
                config.current_gain,
            )
            plume_center = plume_center + np.array([drift_px, 0.0, drift_pz], dtype=np.float32) * float(config.dt_s) * float(
                config.plume_advect_gain
            )
            plume_center = _clamp_to_stage_bounds(plume_center, extents)
            plume_center, _ = apply_depth_clamp(plume_center)
            plume_obj.translation = plume_center

            t_s = float(step) * float(config.dt_s)
            sigma_t = math.sqrt(sigma0 * sigma0 + 2.0 * float(config.plume_diffusion_m2ps) * t_s)
            update_plume_cloud(plume_particles, plume_offsets, plume_center, sigma_m=sigma_t, step=step)
            step_silt_particles(step, env1["silt"])

            # Control toward plume based on concentration gradient.
            ctrl = _vehicle_controller_plume(
                pos=pos, plume_center=plume_center, speed=float(config.vehicle_speed_mps), rng=rng, sigma_m=float(sigma_t)
            )
            drift_x, drift_z, drift_tag = _sample_drift_sim_xz(
                drift_field,
                pos,
                step,
                horizontal_scale,
                drift_origin_lat,
                drift_origin_lon,
                config.synthetic_drift_amplitude_mps,
                config.current_gain,
            )
            vel = ctrl + np.array([drift_x, 0.0, drift_z], dtype=np.float32)
            next_pos = pos + vel * float(config.dt_s)
            next_pos = _clamp_to_stage_bounds(next_pos, extents)
            next_pos, collided = apply_depth_clamp(next_pos)
            collisions += collided
            path_len += float(np.linalg.norm((next_pos - pos)[[0, 2]]))
            energy += float(np.dot(ctrl, ctrl)) * float(config.dt_s)
            pos = next_pos

            # Orient vehicle along velocity direction.
            heading = np.asarray(vel[[0, 2]], dtype=np.float32)
            if float(np.linalg.norm(heading)) > 1e-5:
                yaw = float(math.atan2(float(heading[0]), float(heading[1])))
                vehicle.rotation = _yaw_quat_magnum(yaw)
            vehicle.translation = pos

            dist = float(np.linalg.norm((pos - plume_center)[[0, 2]]))
            conc = _plume_concentration(pos, plume_center, sigma_m=float(sigma_t))
            if success < 0.5 and dist <= float(config.success_radius_m):
                success = 1.0
                t_success = step

            focus = 0.7 * plume_center + 0.3 * pos
            update_camera(step, focus=focus)
            hud = [
                "Task1: Plume localization (single AUV)",
                f"step={step:03d}  dist_to_plume={dist:.2f} m  conc={conc:.3f}  sigma={sigma_t:.1f}",
                f"energy_proxy={energy:.1f}  collisions={int(collisions)}",
                f"drift={drift_tag}",
            ]
            frame = render_frame(step, hud, drift_x, drift_z)
            frames.append(frame)

            traj_rows.append(
                {
                    "step": int(step),
                    "x": float(pos[0]),
                    "y": float(pos[1]),
                    "z": float(pos[2]),
                    "plume_x": float(plume_center[0]),
                    "plume_y": float(plume_center[1]),
                    "plume_z": float(plume_center[2]),
                    "drift_x_mps": float(drift_x),
                    "drift_z_mps": float(drift_z),
                    "dist_to_plume_m": float(dist),
                    "concentration": float(conc),
                    "sigma_m": float(sigma_t),
                }
            )

            if success > 0.5:
                break

        final_png = task1_dir / "final.png"
        Image.fromarray(frames[-1]).save(final_png)
        mp4_path = task1_dir / "rollout.mp4"
        gif_path = task1_dir / "rollout.gif"
        write_video_and_gif(frames, mp4_path=mp4_path, gif_path=gif_path)

        metrics1 = {
            "task": "plume_localization",
            "success": float(success),
            "time_to_success_steps": int(t_success) if t_success is not None else -1,
            "steps": int(len(frames)),
            "path_length_m": float(path_len),
            "energy_proxy": float(energy),
            "seafloor_collision_count": float(collisions),
            "dataset_grounded": bool(drift_field is not None),
            "plume_diffusion_m2ps": float(config.plume_diffusion_m2ps),
            "environment": {
                "rock_count": int(len(rock_specs)),
                "silt_particle_count": int(config.silt_particle_count),
                "plume_particle_count": int(config.plume_particle_count),
            },
        }
        metrics_path = task1_dir / "metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(metrics1, f, indent=2)
        traj_path = task1_dir / "trajectory.jsonl"
        with traj_path.open("w", encoding="utf-8") as f:
            for row in traj_rows:
                f.write(json.dumps(row) + "\n")

        artifacts["tasks"].append(
            {
                "task_id": "plume_localization",
                "scene_png": str(scene_png),
                "final_png": str(final_png),
                "video_mp4": str(mp4_path) if config.write_video else "",
                "gif": str(gif_path),
                "metrics_json": str(metrics_path),
                "trajectory_jsonl": str(traj_path),
            }
        )

        # -----------------------
        # Task 2: Plume containment (multi-agent)
        # -----------------------
        sim.get_rigid_object_manager().remove_all_objects()
        env2 = spawn_environment()

        n_agents = 3
        plume_center = np.array([0.0, -3.2, 0.0], dtype=np.float32)
        plume_center, _ = apply_depth_clamp(plume_center)
        centers = []
        for i in range(n_agents):
            ang = (2.0 * math.pi) * (float(i) / float(n_agents))
            centers.append(
                plume_center
                + np.array(
                    [
                        float(config.formation_radius_m * math.cos(ang)),
                        0.0,
                        float(config.formation_radius_m * math.sin(ang)),
                    ],
                    dtype=np.float32,
                )
            )
        # Add a small deterministic perturbation so the controller must work under drift.
        for i in range(n_agents):
            jitter = rng.normal(loc=0.0, scale=1.8, size=(3,)).astype(np.float32)
            jitter[1] = 0.0
            centers[i] = centers[i] + jitter
        vehicles = [
            _spawn_object(sim, tpl_red, translation=centers[0]),
            _spawn_object(sim, tpl_green, translation=centers[1]),
            _spawn_object(sim, tpl_blue, translation=centers[2]),
        ]
        positions = [c.copy() for c in centers]
        for i in range(len(positions)):
            positions[i], _ = apply_depth_clamp(_clamp_to_stage_bounds(positions[i], extents))
            vehicles[i].translation = positions[i]

        plume_obj = _spawn_object(sim, tpl_marker, translation=plume_center)
        plume_rng2 = np.random.default_rng(int(config.seed) + 303)
        plume_particles2, plume_offsets2 = spawn_plume_cloud(plume_center, rng=plume_rng2, sigma_m=sigma0)

        frames = []
        traj_rows = []
        energy_by_agent = [0.0 for _ in range(n_agents)]
        collisions_by_agent = [0.0 for _ in range(n_agents)]
        formation_hold = 0
        success2 = 0.0
        t_success2 = None

        update_camera(0, focus=plume_center)
        drift_x0, drift_z0, _ = _sample_drift_sim_xz(
            drift_field,
            plume_center,
            0,
            horizontal_scale,
            drift_origin_lat,
            drift_origin_lon,
            config.synthetic_drift_amplitude_mps,
            config.current_gain,
        )
        scene_png2 = task2_dir / "scene.png"
        Image.fromarray(render_frame(0, ["Task2: Plume containment (multi-agent)", "step=0"], drift_x0, drift_z0)).save(
            scene_png2
        )

        for step in range(int(config.max_steps_task2)):
            drift_px, drift_pz, drift_tag = _sample_drift_sim_xz(
                drift_field,
                plume_center,
                step,
                horizontal_scale,
                drift_origin_lat,
                drift_origin_lon,
                config.synthetic_drift_amplitude_mps,
                config.current_gain,
            )
            plume_center = plume_center + np.array([drift_px, 0.0, drift_pz], dtype=np.float32) * float(config.dt_s) * float(
                config.plume_advect_gain
            )
            plume_center = _clamp_to_stage_bounds(plume_center, extents)
            plume_center, _ = apply_depth_clamp(plume_center)
            plume_obj.translation = plume_center

            t_s = float(step) * float(config.dt_s)
            sigma_t = math.sqrt(sigma0 * sigma0 + 2.0 * float(config.plume_diffusion_m2ps) * t_s)
            update_plume_cloud(plume_particles2, plume_offsets2, plume_center, sigma_m=sigma_t, step=step)
            step_silt_particles(step, env2["silt"])

            desired_positions: list[np.ndarray] = []
            for i in range(n_agents):
                ang = (2.0 * math.pi) * (float(i) / float(n_agents))
                desired_positions.append(
                    plume_center
                    + np.array(
                        [
                            float(config.formation_radius_m * math.cos(ang)),
                            0.0,
                            float(config.formation_radius_m * math.sin(ang)),
                        ],
                        dtype=np.float32,
                    )
                )

            formation_errors = []
            for i in range(n_agents):
                ctrl = _vehicle_controller_formation(
                    pos=positions[i],
                    desired=desired_positions[i],
                    kp=float(config.formation_kp),
                    max_speed=float(config.vehicle_speed_mps),
                )
                drift_x, drift_z, _ = _sample_drift_sim_xz(
                    drift_field,
                    positions[i],
                    step,
                    horizontal_scale,
                    drift_origin_lat,
                    drift_origin_lon,
                    config.synthetic_drift_amplitude_mps,
                    config.current_gain,
                )
                vel = ctrl + np.array([drift_x, 0.0, drift_z], dtype=np.float32)
                next_pos = positions[i] + vel * float(config.dt_s)
                next_pos = _clamp_to_stage_bounds(next_pos, extents)
                next_pos, collided = apply_depth_clamp(next_pos)
                collisions_by_agent[i] += collided
                energy_by_agent[i] += float(np.dot(ctrl, ctrl)) * float(config.dt_s)
                positions[i] = next_pos
                vehicles[i].translation = next_pos
                heading = np.asarray(vel[[0, 2]], dtype=np.float32)
                if float(np.linalg.norm(heading)) > 1e-5:
                    yaw = float(math.atan2(float(heading[0]), float(heading[1])))
                    vehicles[i].rotation = _yaw_quat_magnum(yaw)
                formation_errors.append(float(np.linalg.norm((positions[i] - desired_positions[i])[[0, 2]])))

            mean_err = float(np.mean(formation_errors)) if formation_errors else 0.0
            if mean_err <= float(config.formation_success_error_m):
                formation_hold += 1
            else:
                formation_hold = 0

            if success2 < 0.5 and formation_hold >= int(config.formation_success_hold_steps):
                success2 = 1.0
                t_success2 = step

            focus = plume_center.copy()
            update_camera(step, focus=focus)
            hud = [
                "Task2: Plume containment (3 AUVs)",
                f"step={step:03d}  mean_formation_error={mean_err:.2f} m  hold={formation_hold}/{config.formation_success_hold_steps}",
                f"energy_proxy={sum(energy_by_agent):.1f}  collisions={sum(int(c) for c in collisions_by_agent)}  sigma={sigma_t:.1f}",
                f"drift={drift_tag}",
            ]
            frame = render_frame(step, hud, drift_px, drift_pz)
            frames.append(frame)

            traj_rows.append(
                {
                    "step": int(step),
                    "plume_x": float(plume_center[0]),
                    "plume_y": float(plume_center[1]),
                    "plume_z": float(plume_center[2]),
                    "mean_formation_error_m": float(mean_err),
                    "drift_x_mps": float(drift_px),
                    "drift_z_mps": float(drift_pz),
                    "energy_proxy_total": float(sum(energy_by_agent)),
                    "sigma_m": float(sigma_t),
                }
            )
            for i in range(n_agents):
                traj_rows[-1].update(
                    {
                        f"a{i}_x": float(positions[i][0]),
                        f"a{i}_y": float(positions[i][1]),
                        f"a{i}_z": float(positions[i][2]),
                        f"a{i}_energy_proxy": float(energy_by_agent[i]),
                        f"a{i}_collided": float(collisions_by_agent[i]),
                    }
                )

            if success2 > 0.5:
                break

        final_png2 = task2_dir / "final.png"
        Image.fromarray(frames[-1]).save(final_png2)
        mp4_path2 = task2_dir / "rollout.mp4"
        gif_path2 = task2_dir / "rollout.gif"
        write_video_and_gif(frames, mp4_path=mp4_path2, gif_path=gif_path2)

        metrics2 = {
            "task": "plume_containment_multiagent",
            "agents": int(n_agents),
            "success": float(success2),
            "time_to_success_steps": int(t_success2) if t_success2 is not None else -1,
            "steps": int(len(frames)),
            "mean_energy_proxy": float(np.mean(energy_by_agent)),
            "energy_proxy_total": float(sum(energy_by_agent)),
            "seafloor_collision_count_total": float(sum(collisions_by_agent)),
            "dataset_grounded": bool(drift_field is not None),
            "plume_diffusion_m2ps": float(config.plume_diffusion_m2ps),
            "environment": {
                "rock_count": int(len(rock_specs)),
                "silt_particle_count": int(config.silt_particle_count),
                "plume_particle_count": int(config.plume_particle_count),
            },
            "formation": {
                "radius_m": float(config.formation_radius_m),
                "kp": float(config.formation_kp),
                "success_error_m": float(config.formation_success_error_m),
                "success_hold_steps": int(config.formation_success_hold_steps),
            },
        }
        metrics_path2 = task2_dir / "metrics.json"
        with metrics_path2.open("w", encoding="utf-8") as f:
            json.dump(metrics2, f, indent=2)
        traj_path2 = task2_dir / "trajectory.jsonl"
        with traj_path2.open("w", encoding="utf-8") as f:
            for row in traj_rows:
                f.write(json.dumps(row) + "\n")

        artifacts["tasks"].append(
            {
                "task_id": "plume_containment_multiagent",
                "scene_png": str(scene_png2),
                "final_png": str(final_png2),
                "video_mp4": str(mp4_path2) if config.write_video else "",
                "gif": str(gif_path2),
                "metrics_json": str(metrics_path2),
                "trajectory_jsonl": str(traj_path2),
            }
        )

        # Top-level config + manifest
        with (output_dir / "run_config.json").open("w", encoding="utf-8") as f:
            json.dump({"config": asdict(config)}, f, indent=2)
        with (output_dir / "media_manifest.json").open("w", encoding="utf-8") as f:
            json.dump(artifacts, f, indent=2)

        return artifacts

    finally:
        sim.close()
