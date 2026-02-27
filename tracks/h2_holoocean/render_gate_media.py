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
    max_sensor_hz: int = 20
    strip_nonvis_sensors: bool = False
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
    if cfg.strip_nonvis_sensors:
        keep = {"PoseSensor", "VelocitySensor", "DepthSensor", "CollisionSensor"}
        sensors = [s for s in sensors if s.get("sensor_type") in keep]

    for sensor in sensors:
        hz = int(sensor.get("Hz", cfg.ticks_per_sec))
        sensor["Hz"] = min(hz, cfg.ticks_per_sec, int(cfg.max_sensor_hz))

        if sensor.get("sensor_type") in {"RGBCamera"}:
            sensor["Hz"] = cfg.fps

    # Ensure an FPV camera exists even for scenarios without RGBCamera by default.
    if not any(s.get("sensor_type") == "RGBCamera" for s in sensors):
        sensors.append(
            {
                "sensor_type": "RGBCamera",
                "sensor_name": "LeftCamera",
                "socket": "CameraLeftSocket",
                "Hz": cfg.fps,
                "configuration": {"CaptureWidth": 512, "CaptureHeight": 512},
            }
        )

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

def _state_agent(state: dict, agent_name: str) -> dict:
    if agent_name in state and isinstance(state[agent_name], dict):
        return state[agent_name]
    return state


def _get_sensor(state: dict, *, agent_name: str, sensor_key: str):
    agent_state = _state_agent(state, agent_name)
    if sensor_key in agent_state and agent_state[sensor_key] is not None:
        return agent_state[sensor_key]
    if sensor_key in state and state[sensor_key] is not None:
        return state[sensor_key]
    return None


def _warmup_until_agent(state_fn, *, agent_name: str, required_keys: list[str], max_ticks: int = 240):
    for _ in range(max_ticks):
        state = state_fn()
        if all((_get_sensor(state, agent_name=agent_name, sensor_key=k) is not None) for k in required_keys):
            return state
    raise RuntimeError(f"Sensors not available after warmup ticks for agent={agent_name!r}: {required_keys}")

def _safe_call(fn, *, retries: int = 6, base_sleep_s: float = 0.15):
    import time

    last_exc: Exception | None = None
    for i in range(int(retries)):
        try:
            return fn()
        except Exception as e:
            last_exc = e
            if e.__class__.__name__ != "BusyError":
                raise
            time.sleep(float(base_sleep_s) * float(i + 1))
    raise RuntimeError(f"HoloOcean call failed after {retries} retries (BusyError).") from last_exc


def _make_env_with_retries(*, make_fn, retries: int = 4, base_sleep_s: float = 0.8):
    import time

    last_exc: Exception | None = None
    for i in range(int(retries)):
        try:
            return make_fn()
        except Exception as e:
            last_exc = e
            if e.__class__.__name__ != "BusyError":
                raise
            time.sleep(float(base_sleep_s) * float(i + 1))
    raise RuntimeError(f"HoloOcean make() failed after {retries} retries (BusyError).") from last_exc

def _agent_type(scenario: dict) -> str:
    try:
        agents = scenario.get("agents") or []
        if agents:
            return str(agents[0].get("agent_type") or "")
    except Exception:
        pass
    return ""


def _action_candidates(agent_type: str):
    import numpy as np

    agent_type = str(agent_type or "")
    if agent_type == "HoveringAUV":
        a = np.zeros((8,), dtype=np.float32)
        a[4:8] = 18.0
        yield a
        # alternative: slightly more forward + down for visible motion
        b = np.zeros((8,), dtype=np.float32)
        b[:4] = -2.0
        b[4:8] = 18.0
        yield b
        return

    if agent_type == "TorpedoAUV":
        # Heuristic: many torpedo-style controllers accept a small action vector (e.g., throttle + control surfaces).
        # We don't assume an exact schema; the caller will probe with env.step().
        for throttle in (10.0, 18.0, 1.0):
            yield np.array([throttle, 0.0, 0.0, 0.0], dtype=np.float32)
            yield np.array([throttle, 0.2, 0.0, 0.0], dtype=np.float32)
            yield np.array([throttle, -0.2, 0.0, 0.0], dtype=np.float32)
        return

    # Generic fallback: try a few common action sizes.
    for n in (8, 6, 4, 3, 2, 1):
        a = np.zeros((n,), dtype=np.float32)
        if n >= 1:
            a[0] = 10.0
        if n >= 4:
            a[-1] = 0.2
        yield a


def _probe_action(env, *, agent_type: str, ticks_per_frame: int, base_state: dict):
    import numpy as np

    pose0 = _pose_to_position(base_state.get("PoseSensor", None))
    for act in _action_candidates(agent_type):
        try:
            st1 = env.step(np.asarray(act, dtype=np.float32), ticks=ticks_per_frame, publish=False)
        except Exception:
            continue
        pose1 = _pose_to_position(st1.get("PoseSensor", None))
        if pose0 is None or pose1 is None:
            return act
        dx = float(pose1[0] - pose0[0])
        dy = float(pose1[1] - pose0[1])
        dz = float(pose1[2] - pose0[2])
        if (dx * dx + dy * dy + dz * dz) ** 0.5 > 0.05:
            return act
    return None


def main() -> int:
    cfg = GateCfg(
        scenario_name=os.environ.get("SCENARIO_NAME", GateCfg.scenario_name),
        show_viewport=os.environ.get("SHOW_VIEWPORT", "0").strip() in {"1", "true", "True"},
        ticks_per_sec=int(os.environ.get("TICKS_PER_SEC", str(GateCfg.ticks_per_sec))),
        fps=int(os.environ.get("FPS", str(GateCfg.fps))),
        orbit_seconds=float(os.environ.get("ORBIT_SECONDS", str(GateCfg.orbit_seconds))),
        move_seconds=float(os.environ.get("MOVE_SECONDS", str(GateCfg.move_seconds))),
        render_quality=int(os.environ.get("RENDER_QUALITY", str(GateCfg.render_quality))),
        max_sensor_hz=int(os.environ.get("MAX_SENSOR_HZ", str(GateCfg.max_sensor_hz))),
        strip_nonvis_sensors=os.environ.get("STRIP_NONVIS_SENSORS", "0").strip() in {"1", "true", "True"},
    )

    out_dir = Path(os.environ.get("OUT_DIR", f"runs/h2_holoocean/gate_media_{_tag_now_local()}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    ticks_per_frame = int(cfg.ticks_per_sec // cfg.fps)
    if ticks_per_frame * cfg.fps != cfg.ticks_per_sec:
        raise ValueError(f"ticks_per_sec must be divisible by fps; got {cfg.ticks_per_sec=} {cfg.fps=}")

    orbit_frames = max(1, int(round(cfg.orbit_seconds * cfg.fps)))
    move_frames = max(1, int(round(cfg.move_seconds * cfg.fps)))

    # Scenario baseline (installed package json), then patch for viewport capture + reduced tick rates.
    import holoocean
    from holoocean.packagemanager import get_scenario

    base_scenario = get_scenario(cfg.scenario_name)
    scenario = _patch_scenario_for_gate(base_scenario, cfg)
    main_agent_name = str(scenario.get("main_agent") or (scenario.get("agents") or [{}])[0].get("agent_name") or "auv0")

    manifest = {
        "created_at_utc": _utc_now(),
        "track": "h2_holoocean",
        "script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "out_dir": str(out_dir),
        "cfg": asdict(cfg),
        "scenario_name": cfg.scenario_name,
        "scenario_cfg_note": "scenario loaded from installed package then patched in-memory (package_name + ViewportCapture + tick/fps settings).",
        "main_agent_name": main_agent_name,
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

    env_cm = _make_env_with_retries(
        make_fn=lambda: holoocean.make(
            scenario_cfg=scenario,
            show_viewport=cfg.show_viewport,
            ticks_per_sec=cfg.ticks_per_sec,
            frames_per_sec=cfg.fps,
            verbose=False,
        )
    )

    with env_cm as env:
        env.set_render_quality(int(cfg.render_quality))
        env.should_render_viewport(True)

        # Wait for cameras to start producing frames.
        def _tick_once():
            return _safe_call(lambda: env.tick(num_ticks=ticks_per_frame, publish=False))

        _tick_once()
        # Some scenarios return state nested under agent_name; always warm up on the main agent.
        state0 = _warmup_until_agent(lambda: _tick_once(), agent_name=main_agent_name, required_keys=["ViewportCapture"], max_ticks=200)
        agent0 = _state_agent(state0, main_agent_name)

        import numpy as np

        viewport0 = _ensure_uint8_rgb(_get_sensor(state0, agent_name=main_agent_name, sensor_key="ViewportCapture"))
        _write_png(orbit_png, viewport0)
        fpv_source = "LeftCamera" if (_get_sensor(state0, agent_name=main_agent_name, sensor_key="LeftCamera") is not None) else "ViewportCapture(fallback)"
        manifest["outputs"]["fpv_source_note"] = fpv_source

        # Orbit / third-person clip.
        center = _pose_to_position(_get_sensor(state0, agent_name=main_agent_name, sensor_key="PoseSensor"))
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
                vp = _get_sensor(st, agent_name=main_agent_name, sensor_key="ViewportCapture")
                if vp is None:
                    continue
                rgb = _ensure_uint8_rgb(vp)
                vw.append(rgb)
                if (i % orbit_stride) == 0:
                    orbit_gif_frames.append(_downscale_for_gif(rgb, target_width=480))
        _write_gif(orbit_gif, orbit_gif_frames, fps=8)

        # Vehicle motion clip (probe an action vector that actually moves for this agent type).
        agent_type = _agent_type(scenario)
        action = _probe_action(env, agent_type=agent_type, ticks_per_frame=ticks_per_frame, base_state=_state_agent(state0, main_agent_name))
        if action is None:
            manifest["outputs"]["move_note"] = f"No compatible action found for agent_type={agent_type!r}; wrote orbit clip only."
            action = np.zeros((1,), dtype=np.float32)

        move_gif_frames = []
        fpv_gif_frames = []
        move_stride = max(1, int(round(float(cfg.fps) / 8.0)))
        with _Mp4Writer(move_mp4, fps=cfg.fps) as vw, _Mp4Writer(move_fp_mp4, fps=cfg.fps) as fw:
            last = None
            for i in range(move_frames):
                # Keep viewport slightly above/behind the vehicle if PoseSensor is available.
                last_a = _state_agent(last, main_agent_name) if last is not None else None
                pos = _pose_to_position(last_a.get("PoseSensor", None)) if last_a is not None else None
                if pos is not None:
                    px, py, pz = pos
                    r = 10.0
                    cx = px - r
                    cy = py - r
                    cz = pz + 3.5
                    env.move_viewport([cx, cy, cz], _look_at_rpy([cx, cy, cz], [px, py, pz]))

                last = _safe_call(lambda: env.step(action, ticks=ticks_per_frame, publish=False))
                if i == 0:
                    vp = _get_sensor(last, agent_name=main_agent_name, sensor_key="ViewportCapture")
                    if vp is not None:
                        _write_png(move_png, _ensure_uint8_rgb(vp))
                if i == 0:
                    fp = _get_sensor(last, agent_name=main_agent_name, sensor_key="LeftCamera")
                    if fp is None:
                        fp = _get_sensor(last, agent_name=main_agent_name, sensor_key="ViewportCapture")
                    if fp is not None:
                        _write_png(move_fp_png, _ensure_uint8_rgb(fp))

                vp = _get_sensor(last, agent_name=main_agent_name, sensor_key="ViewportCapture")
                if vp is not None:
                    rgb = _ensure_uint8_rgb(vp)
                    vw.append(rgb)
                    if (i % move_stride) == 0:
                        move_gif_frames.append(_downscale_for_gif(rgb, target_width=480))
                fp = _get_sensor(last, agent_name=main_agent_name, sensor_key="LeftCamera")
                if fp is None:
                    fp = _get_sensor(last, agent_name=main_agent_name, sensor_key="ViewportCapture")
                if fp is not None:
                    rgb = _ensure_uint8_rgb(fp)
                    fw.append(rgb)
                    if (i % move_stride) == 0:
                        fpv_gif_frames.append(_downscale_for_gif(rgb, target_width=480))

        _write_gif(move_gif, move_gif_frames, fps=8)
        _write_gif(move_fp_gif, fpv_gif_frames, fps=8)

    (out_dir / "media_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
