#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class UnderwaterPostprocess:
    beta_rgb: tuple[float, float, float] = (0.06, 0.035, 0.018)  # attenuation per meter (R>G>B)
    ambient_rgb: tuple[float, float, float] = (12.0, 40.0, 85.0)  # haze color
    haze_strength: float = 0.75
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

    # low-contrast wash
    out = 0.92 * out + 0.08 * amb

    # particles (screen-space)
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


def _load_scene_meta(scene_path: Path) -> dict[str, Any] | None:
    meta = scene_path.parent / "scene_meta.json"
    if meta.exists():
        return json.loads(meta.read_text(encoding="utf-8"))
    return None


def _look_at_quat(forward: np.ndarray, target_dir: np.ndarray) -> Any:
    import habitat_sim.utils.common as c

    f = np.asarray(forward, dtype=np.float32)
    t = np.asarray(target_dir, dtype=np.float32)
    n = np.linalg.norm(t)
    if n < 1e-6:
        return c.quat_from_coeffs(np.array([0, 0, 0, 1], dtype=np.float32))
    t = t / n
    return c.quat_from_two_vectors(f, t)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", type=str, required=True, help="Path to stage file (e.g., stage.obj)")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--hfov", type=float, default=80.0)
    ap.add_argument("--zfar", type=float, default=120.0)
    ap.add_argument("--frames", type=int, default=180)
    ap.add_argument("--fps", type=float, default=24.0)
    ap.add_argument("--radius-mult", type=float, default=1.6, help="Orbit radius multiplier on scene radius")
    ap.add_argument("--elev-deg", type=float, default=18.0)
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    scene_path = Path(args.scene).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = _load_scene_meta(scene_path) or {}
    radius = float(meta.get("mesh", {}).get("radius", 3.0))
    center = np.array(meta.get("mesh", {}).get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
    orbit_r = max(0.8, float(args.radius_mult) * max(0.8, radius))
    elev = math.radians(float(args.elev_deg))

    import habitat_sim
    import habitat_sim.utils.settings as hs
    import imageio.v3 as iio

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
            "enable_physics": False,
        }
    )
    sim = habitat_sim.Simulator(hs.make_cfg(settings))
    agent = sim.initialize_agent(0)

    rng = np.random.default_rng(int(args.seed))
    uw_cfg = UnderwaterPostprocess()

    mp4_path = out_dir / "orbit_underwater.mp4"
    keyframe_path = out_dir / "keyframe_underwater.png"

    frames: list[np.ndarray] = []
    forward_axis = np.array([0.0, 0.0, -1.0], dtype=np.float32)

    for i in range(int(args.frames)):
        theta = 2.0 * math.pi * (i / max(1, int(args.frames)))
        cam_pos = center + np.array(
            [orbit_r * math.cos(theta), orbit_r * math.sin(elev), orbit_r * math.sin(theta)],
            dtype=np.float32,
        )
        to_center = (center - cam_pos).astype(np.float32)
        rot = _look_at_quat(forward_axis, to_center)

        state = habitat_sim.AgentState()
        state.position = cam_pos
        state.rotation = rot
        agent.set_state(state)

        obs = sim.get_sensor_observations()
        rgb = np.asarray(obs["color_sensor"])
        depth = np.asarray(obs["depth_sensor"]).astype(np.float32)

        # Habitat depth sensor returns in meters (float), shape (H,W,1) sometimes.
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]

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
        "render": {
            "frames": int(args.frames),
            "fps": float(args.fps),
            "width": int(args.width),
            "height": int(args.height),
            "orbit_radius_m": orbit_r,
            "elev_deg": float(args.elev_deg),
        },
        "underwater_postprocess": asdict(uw_cfg),
    }
    (out_dir / "render_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    sim.close()


if __name__ == "__main__":
    main()
