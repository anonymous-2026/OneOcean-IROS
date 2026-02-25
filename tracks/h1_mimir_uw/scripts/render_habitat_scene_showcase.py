#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class UnderwaterPostprocess:
    beta_rgb: tuple[float, float, float] = (0.06, 0.035, 0.018)
    ambient_rgb: tuple[float, float, float] = (12.0, 40.0, 85.0)
    haze_strength: float = 0.78
    particle_count: int = 220
    particle_radius_px: int = 1
    particle_alpha: float = 0.18


def _apply_underwater(rgb_u8: np.ndarray, depth_m: np.ndarray, cfg: UnderwaterPostprocess, rng: np.random.Generator) -> np.ndarray:
    rgb = rgb_u8
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    rgb = rgb.astype(np.float32)
    depth = depth_m.astype(np.float32)
    depth = np.nan_to_num(depth, nan=50.0, posinf=50.0, neginf=50.0)
    depth = np.clip(depth, 0.0, 80.0)

    amb = np.array(cfg.ambient_rgb, dtype=np.float32).reshape(1, 1, 3)
    beta = np.array(cfg.beta_rgb, dtype=np.float32).reshape(1, 1, 3)
    trans = np.exp(-beta * depth[..., None])
    haze = (1.0 - trans) * cfg.haze_strength
    out = rgb * trans + amb * haze
    out = 0.92 * out + 0.08 * amb

    h, w = out.shape[:2]
    n = int(max(0, cfg.particle_count))
    if n > 0:
        xs = rng.integers(0, w, size=n)
        ys = rng.integers(0, h, size=n)
        for x, y in zip(xs.tolist(), ys.tolist()):
            r = int(cfg.particle_radius_px)
            x0 = max(0, x - r)
            x1 = min(w, x + r + 1)
            y0 = max(0, y - r)
            y1 = min(h, y + r + 1)
            out[y0:y1, x0:x1, :] = (1.0 - cfg.particle_alpha) * out[y0:y1, x0:x1, :] + cfg.particle_alpha * amb

    return np.clip(out, 0.0, 255.0).astype(np.uint8)


def _load_scene_meta(scene_path: Path) -> dict[str, Any]:
    meta = scene_path.parent / "scene_meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    return {}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Render a stable orbit showcase of the H1 stage + static proxies.")
    ap.add_argument("--scene", type=str, required=True, help="Path to stage file (e.g., stage.obj)")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--assets-dir", type=str, default=str(Path(__file__).resolve().parents[1] / "assets"))
    ap.add_argument("--n-agents", type=int, default=10)
    ap.add_argument("--n-rocks", type=int, default=24)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=544)
    ap.add_argument("--hfov", type=float, default=80.0)
    ap.add_argument("--zfar", type=float, default=120.0)
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--radius-mult", type=float, default=1.9)
    ap.add_argument("--elev-deg", type=float, default=18.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-underwater", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    scene_path = Path(args.scene).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    import habitat_sim
    import habitat_sim.utils.common as c
    import habitat_sim.utils.settings as hs
    import imageio.v3 as iio

    rng = np.random.default_rng(int(args.seed))
    meta = _load_scene_meta(scene_path) or {}
    bounds = meta.get("mesh", {}).get("bounds", None)
    if bounds is None:
        lo = np.array([-4.0, -2.0, -4.0], dtype=np.float32)
        hi = np.array([4.0, 2.0, 4.0], dtype=np.float32)
        center = np.zeros((3,), dtype=np.float32)
        radius = 4.0
    else:
        lo = np.array(bounds[0], dtype=np.float32)
        hi = np.array(bounds[1], dtype=np.float32)
        center = (lo + hi) * 0.5
        radius = float(meta.get("mesh", {}).get("radius", float(np.linalg.norm(hi - lo) * 0.5)))

    pad = np.array([0.25, 0.15, 0.25], dtype=np.float32)
    spawn_lo = lo + pad
    spawn_hi = hi - pad

    settings = dict(hs.default_sim_settings)
    settings.update(
        {
            "scene": str(scene_path),
            "color_sensor": True,
            "depth_sensor": True,
            "semantic_sensor": False,
            "width": int(args.width),
            "height": int(args.height),
            "hfov": float(args.hfov),
            "sensor_height": 0.0,
            "zfar": float(args.zfar),
            "clear_color": [0.0, 0.0, 0.0, 1.0],
            "seed": int(args.seed),
            "enable_physics": True,
        }
    )
    sim = habitat_sim.Simulator(hs.make_cfg(settings))
    agent = sim.initialize_agent(0)

    assets_dir = Path(args.assets_dir).resolve()
    uuv_cfg = assets_dir / "uuv_proxy.object_config.json"
    rock_cfg = assets_dir / "rock_proxy.object_config.json"

    rom = sim.get_rigid_object_manager()
    otm = sim.get_object_template_manager()
    uuv_tid = int(otm.load_configs(str(uuv_cfg))[0])
    rock_tid = int(otm.load_configs(str(rock_cfg))[0])

    rocks: list[Any] = []
    for _ in range(int(max(0, args.n_rocks))):
        o = rom.add_object_by_template_id(rock_tid)
        o.motion_type = habitat_sim.physics.MotionType.STATIC
        o.translation = np.array(
            [
                rng.uniform(float(spawn_lo[0]), float(spawn_hi[0])),
                float(spawn_lo[1] + 0.05 + 0.22 * rng.random()),
                rng.uniform(float(spawn_lo[2]), float(spawn_hi[2])),
            ],
            dtype=np.float32,
        )
        o.rotation = c.quat_to_magnum(c.quat_from_angle_axis(float(rng.uniform(0, 2 * math.pi)), np.array([0, 1, 0], dtype=np.float32)))
        rocks.append(o)

    n_agents = int(np.clip(args.n_agents, 2, 10))
    agents: list[Any] = []
    ring_r = 1.55
    for i in range(n_agents):
        o = rom.add_object_by_template_id(uuv_tid)
        o.motion_type = habitat_sim.physics.MotionType.KINEMATIC
        ang = 2.0 * math.pi * (i / n_agents)
        pos = center + np.array([ring_r * math.cos(ang), 0.85, ring_r * math.sin(ang)], dtype=np.float32)
        pos[0] = float(np.clip(pos[0], spawn_lo[0], spawn_hi[0]))
        pos[1] = float(np.clip(pos[1], spawn_lo[1] + 0.4, spawn_hi[1] - 0.2))
        pos[2] = float(np.clip(pos[2], spawn_lo[2], spawn_hi[2]))
        o.translation = pos
        o.rotation = c.quat_to_magnum(c.quat_from_angle_axis(float(ang + math.pi), np.array([0, 1, 0], dtype=np.float32)))
        agents.append(o)

    orbit_r = max(0.8, float(args.radius_mult) * max(0.8, float(radius)))
    elev = math.radians(float(args.elev_deg))
    forward_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    uw_cfg = UnderwaterPostprocess()
    mp4_path = out_dir / "scene_orbit.mp4"
    keyframe_path = out_dir / "scene_keyframe.png"
    frames: list[np.ndarray] = []

    for i in range(int(args.frames)):
        theta = 2.0 * math.pi * (i / max(1, int(args.frames)))
        cam_pos = center + np.array(
            [orbit_r * math.cos(theta), orbit_r * math.sin(elev) + 0.75, orbit_r * math.sin(theta)],
            dtype=np.float32,
        )
        to_center = (center - cam_pos).astype(np.float32)
        rot = c.quat_from_two_vectors(forward_axis, to_center / max(1e-6, float(np.linalg.norm(to_center))))

        state = habitat_sim.AgentState()
        state.position = cam_pos
        state.rotation = rot
        agent.set_state(state)

        obs = sim.get_sensor_observations()
        rgb = np.asarray(obs["color_sensor"])
        depth = np.asarray(obs["depth_sensor"]).astype(np.float32)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]

        if bool(args.no_underwater):
            out = rgb[:, :, :3] if (rgb.ndim == 3 and rgb.shape[2] == 4) else rgb
            out = out.astype(np.uint8)
        else:
            out = _apply_underwater(rgb, depth, uw_cfg, rng)

        frames.append(out)
        if i == 0:
            iio.imwrite(keyframe_path, out)

    iio.imwrite(mp4_path, frames, fps=float(args.fps))

    manifest = {
        "scene": str(scene_path),
        "scene_meta": str(scene_path.parent / "scene_meta.json") if (scene_path.parent / "scene_meta.json").exists() else None,
        "output_dir": str(out_dir),
        "outputs": {"mp4": str(mp4_path), "keyframe_png": str(keyframe_path)},
        "render": {"frames": int(args.frames), "fps": float(args.fps), "width": int(args.width), "height": int(args.height)},
        "orbit": {"orbit_radius_m": float(orbit_r), "elev_deg": float(args.elev_deg)},
        "proxies": {"n_agents": int(n_agents), "n_rocks": int(max(0, int(args.n_rocks)))},
        "underwater_postprocess": None if bool(args.no_underwater) else asdict(uw_cfg),
        "seed": int(args.seed),
    }
    (out_dir / "media_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    sim.close()


if __name__ == "__main__":
    main()

