from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Optional

import imageio
import numpy as np

import habitat_sim

from .runner import (
    CameraConfig,
    _apply_underwater_postprocess,
    _camera_state_from_orbit,
    _normalize_depth,
    _normalize_rgb,
    _register_mesh_template,
    _setup_sim,
    _spawn_object,
)


@dataclass
class ShowcaseConfig:
    stage_obj: str
    stage_meta: Optional[str] = None  # currently unused, kept for symmetry with underwater runner
    model_gltf: str = ""
    model_count: int = 1
    model_scale: float = 1.0
    output_dir: Optional[str] = None
    invocation: str = ""
    seed: int = 0

    steps: int = 240
    fps: float = 24.0
    gif_stride: int = 2
    camera: CameraConfig = field(default_factory=CameraConfig)

    fog_density: float = 0.070
    water_rgb: tuple[int, int, int] = (10, 55, 80)
    attenuation_rgb: tuple[float, float, float] = (0.14, 0.07, 0.03)

    enable_silt: bool = True
    silt_count: int = 120
    silt_scale: float = 0.028


def _default_output_dir(model_id: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = model_id.replace("/", "_").replace(" ", "_")
    return Path("runs") / f"polyhaven_showcase_{safe}_{stamp}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def render_showcase(cfg: ShowcaseConfig) -> dict[str, str]:
    stage_obj = Path(cfg.stage_obj).expanduser().resolve()
    if not stage_obj.exists():
        raise FileNotFoundError(f"Stage OBJ not found: {stage_obj}")

    model_path = Path(cfg.model_gltf).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model GLTF not found: {model_path}")

    out_dir = Path(cfg.output_dir) if cfg.output_dir else _default_output_dir(model_path.stem)
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    sim, cam_agent = _setup_sim(scene_id=stage_obj, camera=cfg.camera)
    try:
        rng = np.random.default_rng(int(cfg.seed))

        # Register model and place it near origin; also place a few copies around center if requested.
        model_tpl = _register_mesh_template(sim, "showcase_model", model_path, scale=float(cfg.model_scale))

        center = np.array([0.0, -1.2, 0.0], dtype=np.float32)
        spawned: list[habitat_sim.physics.ManagedRigidObject] = []
        for i in range(int(max(1, cfg.model_count))):
            angle = (2.0 * math.pi) * (float(i) / float(max(1, cfg.model_count)))
            radius = 4.0 if cfg.model_count > 1 else 0.0
            pos = np.array(
                [
                    float(center[0] + radius * math.cos(angle)),
                    float(center[1]),
                    float(center[2] + radius * math.sin(angle)),
                ],
                dtype=np.float32,
            )
            obj = _spawn_object(sim, model_tpl, translation=pos, motion_type=habitat_sim.physics.MotionType.STATIC)
            spawned.append(obj)

        # Optional: add a small field of silt particles for underwater cue/parallax.
        silt_objs: list[habitat_sim.physics.ManagedRigidObject] = []
        if bool(cfg.enable_silt) and int(cfg.silt_count) > 0:
            assets_root = Path(__file__).resolve().parents[1] / "assets" / "meshes"
            particle_silt = assets_root / "particle_silt.obj"
            if particle_silt.exists():
                silt_tpl = _register_mesh_template(sim, "particle_silt_showcase", particle_silt, scale=float(cfg.silt_scale))
                for _ in range(int(cfg.silt_count)):
                    x = float(rng.uniform(-0.55 * cfg.camera.orbit_radius_m, 0.55 * cfg.camera.orbit_radius_m))
                    z = float(rng.uniform(-0.55 * cfg.camera.orbit_radius_m, 0.55 * cfg.camera.orbit_radius_m))
                    y = float(rng.uniform(-2.6, -0.6))
                    silt_objs.append(
                        _spawn_object(
                            sim,
                            silt_tpl,
                            translation=np.array([x, y, z], dtype=np.float32),
                            motion_type=habitat_sim.physics.MotionType.KINEMATIC,
                        )
                    )

        frames: list[np.ndarray] = []
        for step in range(int(cfg.steps)):
            cam_state = _camera_state_from_orbit(step, center=center, camera=cfg.camera)
            cam_agent.set_state(cam_state)

            obs = sim.get_sensor_observations()
            rgb = _normalize_rgb(obs["rgb"])
            depth = _normalize_depth(obs["depth"])
            out = _apply_underwater_postprocess(
                rgb,
                depth,
                fog_density=float(cfg.fog_density),
                water_rgb=tuple(cfg.water_rgb),
                attenuation_rgb=tuple(cfg.attenuation_rgb),
            )
            frames.append(out)

        scene_png = out_dir / "scene.png"
        orbit_mp4 = out_dir / "orbit.mp4"
        orbit_gif = out_dir / "orbit.gif"

        import PIL.Image

        PIL.Image.fromarray(frames[0]).save(scene_png)

        with imageio.get_writer(orbit_mp4, fps=float(cfg.fps)) as writer:
            for frame in frames:
                writer.append_data(frame)

        stride = int(max(1, cfg.gif_stride))
        imageio.mimsave(orbit_gif, frames[::stride], fps=float(cfg.fps) / float(stride))

        manifest = {
            "type": "polyhaven_showcase",
            "stage_obj": str(stage_obj),
            "model_gltf": str(model_path),
            "model_count": int(cfg.model_count),
            "model_scale": float(cfg.model_scale),
            "steps": int(cfg.steps),
            "fps": float(cfg.fps),
            "gif_stride": int(cfg.gif_stride),
            "camera": asdict(cfg.camera),
            "underwater_postprocess": {
                "fog_density": float(cfg.fog_density),
                "water_rgb": list(cfg.water_rgb),
                "attenuation_rgb": list(cfg.attenuation_rgb),
            },
            "invocation": str(cfg.invocation),
        }
        _write_json(out_dir / "showcase_manifest.json", manifest)

        return {
            "output_dir": str(out_dir),
            "scene_png": str(scene_png),
            "orbit_mp4": str(orbit_mp4),
            "orbit_gif": str(orbit_gif),
            "manifest": str(out_dir / "showcase_manifest.json"),
        }
    finally:
        sim.close()

