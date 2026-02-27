from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass(frozen=True)
class GateCfg:
    scenario_name: str = "Dam-HoveringCamera"
    package_name: str = "Ocean"
    ticks_per_sec: int = 20
    fps: int = 20
    orbit_seconds: float = 6.0
    move_seconds: float = 6.0
    orbit_radius_m: float = 18.0
    orbit_height_m: float = 6.0
    render_quality: int = 3  # 0..3
    show_viewport: bool = False
    window_width: int = 1280
    window_height: int = 720


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tag_now_local() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_uint8_rgb(frame) -> "np.ndarray":
    import numpy as np

    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = arr.astype(np.uint8, copy=False)
    if arr.ndim != 3:
        raise ValueError(f"Expected HxWxC frame, got shape={arr.shape}")
    if arr.shape[-1] == 4:
        return arr[:, :, :3]
    if arr.shape[-1] == 3:
        return arr
    raise ValueError(f"Expected 3 or 4 channels, got shape={arr.shape}")


def _write_png(path: Path, frame_rgb_u8) -> None:
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.imwrite(path, frame_rgb_u8)


class _Mp4Writer:
    def __init__(self, path: Path, fps: int):
        import imageio.v2 as imageio

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._writer = imageio.get_writer(self.path, fps=fps)

    def append(self, frame_rgb_u8) -> None:
        self._writer.append_data(frame_rgb_u8)

    def close(self) -> None:
        self._writer.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


def _downscale_for_gif(frame_rgb_u8, *, target_width: int = 480):
    from PIL import Image
    import numpy as np

    arr = np.asarray(frame_rgb_u8, dtype=np.uint8)
    h, w = int(arr.shape[0]), int(arr.shape[1])
    tw = int(max(64, target_width))
    if w <= tw:
        return arr
    th = int(round(h * (tw / float(w))))
    img = Image.fromarray(arr, mode="RGB").resize((tw, max(1, th)), resample=Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def _write_gif(path: Path, frames_rgb_u8: list, *, fps: int = 8) -> None:
    import imageio.v2 as imageio

    if not frames_rgb_u8:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    duration = 1.0 / float(max(1, int(fps)))
    imageio.mimsave(path, frames_rgb_u8, duration=duration)


def _patch_scenario_for_gate(scenario: dict, cfg: GateCfg) -> dict:
    scenario = json.loads(json.dumps(scenario))
    scenario["package_name"] = cfg.package_name
    scenario["window_width"] = cfg.window_width
    scenario["window_height"] = cfg.window_height

    scenario["ticks_per_sec"] = int(cfg.ticks_per_sec)
    scenario["frames_per_sec"] = int(cfg.fps)

    if "agents" not in scenario or not scenario["agents"]:
        raise ValueError("Scenario has no agents.")

    main_agent = scenario["agents"][0]
    sensors = list(main_agent.get("sensors", []))
    if not sensors:
        raise ValueError("Scenario agent has no sensors.")

    for sensor in sensors:
        hz = int(sensor.get("Hz", cfg.ticks_per_sec))
        sensor["Hz"] = min(hz, cfg.ticks_per_sec)

        if sensor.get("sensor_type") in {"RGBCamera"}:
            sensor["Hz"] = cfg.fps

    sensors.append(
        {
            "sensor_type": "ViewportCapture",
            "sensor_name": "ViewportCapture",
            "Hz": cfg.fps,
            "configuration": {"CaptureWidth": cfg.window_width, "CaptureHeight": cfg.window_height},
        }
    )
    main_agent["sensors"] = sensors
    scenario["agents"][0] = main_agent
    return scenario


def _pose_to_position(pose) -> list[float] | None:
    import numpy as np

    if pose is None:
        return None
    arr = np.asarray(pose, dtype=np.float32)
    if arr.shape != (4, 4):
        return None
    p = arr[:3, 3]
    return [float(p[0]), float(p[1]), float(p[2])]


def _look_at_rpy(camera_xyz: list[float], target_xyz: list[float]) -> list[float]:
    dx = target_xyz[0] - camera_xyz[0]
    dy = target_xyz[1] - camera_xyz[1]
    dz = target_xyz[2] - camera_xyz[2]
    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = math.hypot(dx, dy)
    pitch = math.degrees(math.atan2(dz, max(1e-6, dist_xy)))
    # HoloOcean uses [roll, pitch, yaw] in degrees for TeleportCamera.
    return [0.0, pitch, yaw]


def _warmup_until(state_fn, required_keys: list[str], max_ticks: int = 240):
    for _ in range(max_ticks):
        state = state_fn()
        if all((k in state and state[k] is not None) for k in required_keys):
            return state
    raise RuntimeError(f"Sensors not available after warmup ticks: {required_keys}")


def main() -> int:
    cfg = GateCfg(
        scenario_name=os.environ.get("SCENARIO_NAME", GateCfg.scenario_name),
        show_viewport=os.environ.get("SHOW_VIEWPORT", "0").strip() in {"1", "true", "True"},
    )

    out_dir = Path(os.environ.get("OUT_DIR", f"runs/h2_holoocean/gate_media_{_tag_now_local()}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ticks_per_frame = int(cfg.ticks_per_sec // cfg.fps)
    if ticks_per_frame * cfg.fps != cfg.ticks_per_sec:
        raise ValueError(f"ticks_per_sec must be divisible by fps; got {cfg.ticks_per_sec=} {cfg.fps=}")
    if ticks_per_frame != 1:
        raise ValueError("This script assumes ticks_per_sec == fps to avoid HoloOcean tick_every gating.")

    orbit_frames = max(1, int(round(cfg.orbit_seconds * cfg.fps)))
    move_frames = max(1, int(round(cfg.move_seconds * cfg.fps)))

    # Scenario baseline (installed package json), then patch for viewport capture + reduced tick rates.
    import holoocean
    from holoocean.packagemanager import get_scenario

    base_scenario = get_scenario(cfg.scenario_name)
    scenario = _patch_scenario_for_gate(base_scenario, cfg)

    manifest = {
        "created_at_utc": _utc_now(),
        "track": "h2_holoocean",
        "script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "out_dir": str(out_dir),
        "cfg": asdict(cfg),
        "scenario_name": cfg.scenario_name,
        "scenario_cfg_note": "scenario loaded from installed package then patched in-memory (package_name + ViewportCapture + tick/fps settings).",
        "worlds_root": "/home/shuaijun/.local/share/holoocean/0.5.0/worlds/Ocean/",
        "world_zip_cache": "/data/private/user2/workspace/ocean/project_mgmt/sync/_external_scene_cache/holocean_worlds/Ocean_v0.5.0_Linux.zip",
        "world_zip_sha256": "1f7ad5f829ae6c9a82daa43b6d21fdfaca07711eb09c1d578f5249b13f040618",
        "outputs": {},
        "command_hint": f"cd oneocean(iros-2026-code) && {sys.executable} tracks/h2_holoocean/render_gate_media.py",
    }

    orbit_png = out_dir / "orbit_keyframe.png"
    orbit_mp4 = out_dir / "orbit_viewport.mp4"
    orbit_gif = out_dir / "orbit_viewport.gif"
    move_png = out_dir / "move_keyframe.png"
    move_mp4 = out_dir / "move_viewport.mp4"
    move_fp_mp4 = out_dir / "move_leftcamera.mp4"
    move_fp_png = out_dir / "move_leftcamera_keyframe.png"
    move_gif = out_dir / "move_viewport.gif"
    move_fp_gif = out_dir / "move_leftcamera.gif"

    manifest["outputs"]["orbit_keyframe_png"] = str(orbit_png)
    manifest["outputs"]["orbit_viewport_mp4"] = str(orbit_mp4)
    manifest["outputs"]["orbit_viewport_gif"] = str(orbit_gif)
    manifest["outputs"]["move_keyframe_png"] = str(move_png)
    manifest["outputs"]["move_viewport_mp4"] = str(move_mp4)
    manifest["outputs"]["move_viewport_gif"] = str(move_gif)
    manifest["outputs"]["move_leftcamera_keyframe_png"] = str(move_fp_png)
    manifest["outputs"]["move_leftcamera_mp4"] = str(move_fp_mp4)
    manifest["outputs"]["move_leftcamera_gif"] = str(move_fp_gif)

    with holoocean.make(
        scenario_cfg=scenario,
        show_viewport=cfg.show_viewport,
        ticks_per_sec=cfg.ticks_per_sec,
        frames_per_sec=cfg.fps,
        verbose=False,
    ) as env:
        env.set_render_quality(int(cfg.render_quality))
        env.should_render_viewport(True)

        # Wait for cameras to start producing frames.
        def _tick_once():
            return env.tick(num_ticks=ticks_per_frame, publish=False)

        _tick_once()
        state0 = _warmup_until(lambda: _tick_once(), ["ViewportCapture", "LeftCamera"], max_ticks=120)

        import numpy as np

        viewport0 = _ensure_uint8_rgb(state0["ViewportCapture"])
        _write_png(orbit_png, viewport0)

        # Orbit / third-person clip.
        center = _pose_to_position(state0.get("PoseSensor", None))
        if center is None:
            # Scenario default start for Dam/PierHarbor hovering camera.
            agent0 = scenario["agents"][0]
            center = [float(x) for x in agent0.get("location", [0.0, 0.0, -5.0])]

        orbit_gif_frames = []
        orbit_stride = max(1, int(round(float(cfg.fps) / 8.0)))
        with _Mp4Writer(orbit_mp4, fps=cfg.fps) as vw:
            for i in range(orbit_frames):
                a = 2.0 * math.pi * (i / float(orbit_frames))
                x = center[0] + cfg.orbit_radius_m * math.cos(a)
                y = center[1] + cfg.orbit_radius_m * math.sin(a)
                z = center[2] + cfg.orbit_height_m
                rpy = _look_at_rpy([x, y, z], center)
                env.move_viewport([x, y, z], rpy)
                st = _tick_once()
                if "ViewportCapture" not in st or st["ViewportCapture"] is None:
                    continue
                rgb = _ensure_uint8_rgb(st["ViewportCapture"])
                vw.append(rgb)
                if (i % orbit_stride) == 0:
                    orbit_gif_frames.append(_downscale_for_gif(rgb, target_width=480))
        _write_gif(orbit_gif, orbit_gif_frames, fps=8)

        # Vehicle motion clip (apply a simple constant thrust command).
        action = np.zeros((8,), dtype=np.float32)
        action[4:8] = 18.0

        move_gif_frames = []
        fpv_gif_frames = []
        move_stride = max(1, int(round(float(cfg.fps) / 8.0)))
        with _Mp4Writer(move_mp4, fps=cfg.fps) as vw, _Mp4Writer(move_fp_mp4, fps=cfg.fps) as fw:
            last = None
            for i in range(move_frames):
                # Keep viewport slightly above/behind the vehicle if PoseSensor is available.
                pos = _pose_to_position(last.get("PoseSensor", None)) if last is not None else None
                if pos is not None:
                    px, py, pz = pos
                    r = 10.0
                    cx = px - r
                    cy = py - r
                    cz = pz + 3.5
                    env.move_viewport([cx, cy, cz], _look_at_rpy([cx, cy, cz], [px, py, pz]))

                last = env.step(action, ticks=ticks_per_frame, publish=False)
                if i == 0 and "ViewportCapture" in last and last["ViewportCapture"] is not None:
                    _write_png(move_png, _ensure_uint8_rgb(last["ViewportCapture"]))
                if i == 0 and "LeftCamera" in last and last["LeftCamera"] is not None:
                    _write_png(move_fp_png, _ensure_uint8_rgb(last["LeftCamera"]))

                if "ViewportCapture" in last and last["ViewportCapture"] is not None:
                    rgb = _ensure_uint8_rgb(last["ViewportCapture"])
                    vw.append(rgb)
                    if (i % move_stride) == 0:
                        move_gif_frames.append(_downscale_for_gif(rgb, target_width=480))
                if "LeftCamera" in last and last["LeftCamera"] is not None:
                    rgb = _ensure_uint8_rgb(last["LeftCamera"])
                    fw.append(rgb)
                    if (i % move_stride) == 0:
                        fpv_gif_frames.append(_downscale_for_gif(rgb, target_width=480))

        _write_gif(move_gif, move_gif_frames, fps=8)
        _write_gif(move_fp_gif, fpv_gif_frames, fps=8)

    (out_dir / "media_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
