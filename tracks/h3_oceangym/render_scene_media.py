from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

# Allow running as a script: `python tracks/h3_oceangym/render_scene_media.py`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _ensure_ssl_cert_file() -> None:
    try:
        import certifi  # type: ignore

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass


def _tag_now_local() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_uint8_rgb(frame):
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


def _write_png(path: Path, rgb_u8) -> None:
    import cv2  # type: ignore

    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))


class _Mp4Writer:
    def __init__(self, path: Path, fps: int, size_hw: tuple[int, int]):
        import cv2  # type: ignore

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        w, h = int(size_hw[1]), int(size_hw[0])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self.path), fourcc, float(fps), (w, h))
        if not self._writer.isOpened():
            raise RuntimeError(f"Failed to open mp4 writer for {self.path}")

    def append_rgb(self, rgb_u8) -> None:
        import cv2  # type: ignore

        self._writer.write(cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR))

    def close(self) -> None:
        self._writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


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
    return [0.0, pitch, yaw]


@dataclass(frozen=True)
class RenderCfg:
    preset: str = "ocean_worlds_camera"
    fps: int = 20
    ticks_per_sec: int = 20
    orbit_seconds: float = 12.0
    move_seconds: float = 12.0
    orbit_radius_m: float = 18.0
    orbit_height_m: float = 6.0
    show_viewport: bool = False
    window_width: int = 1280
    window_height: int = 720
    render_quality: int = 3


def _warmup_until(env, required_keys: list[str], max_ticks: int = 240):
    for _ in range(max_ticks):
        st = env.tick(num_ticks=1, publish=False)
        if all((k in st and st[k] is not None) for k in required_keys):
            return st
    raise RuntimeError(f"Sensors not available after warmup ticks: {required_keys}")


def _scan_outputs(out_dir: Path) -> dict[str, dict]:
    scenarios: dict[str, dict] = {}
    for child in sorted(out_dir.iterdir()):
        if not child.is_dir():
            continue
        name = child.name
        scenarios[name] = {
            "orbit_keyframe_png": str(child / "orbit_keyframe.png"),
            "orbit_viewport_mp4": str(child / "orbit_viewport.mp4"),
            "move_keyframe_png": str(child / "move_keyframe.png"),
            "move_viewport_mp4": str(child / "move_viewport.mp4"),
            "move_leftcamera_mp4": str(child / "move_leftcamera.mp4"),
        }
    return scenarios


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default=RenderCfg.preset)
    ap.add_argument(
        "--scenarios",
        nargs="*",
        default=None,
        help="Optional explicit scenario names (overrides --preset).",
    )
    ap.add_argument("--out_dir", default=None)
    ap.add_argument(
        "--write_manifest_only",
        action="store_true",
        help="Do not render; only (re)write media_manifest.json by scanning the output directory.",
    )
    ap.add_argument("--show_viewport", action="store_true")
    args = ap.parse_args()

    _ensure_ssl_cert_file()

    import holoocean  # type: ignore
    from holoocean import packagemanager as pm  # type: ignore

    from tracks.h3_oceangym.holoocean_patch import HoloCfg, patch_scenario_for_recording
    from tracks.h3_oceangym.scenarios import scenario_preset

    cfg = RenderCfg(show_viewport=bool(args.show_viewport))
    scenarios = list(args.scenarios) if args.scenarios else scenario_preset(args.preset)

    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / "oceangym_h3" / f"scene_media_{_tag_now_local()}"
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    holo_cfg = HoloCfg(
        ticks_per_sec=cfg.ticks_per_sec,
        fps=cfg.fps,
        window_width=cfg.window_width,
        window_height=cfg.window_height,
        render_quality=cfg.render_quality,
        show_viewport=cfg.show_viewport,
    )

    manifest = {
        "track": "h3_oceangym",
        "script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "out_dir": str(out_dir),
        "cfg": asdict(cfg),
        "scenarios": _scan_outputs(out_dir),
    }

    if args.write_manifest_only:
        (out_dir / "media_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("[h3] wrote:", out_dir / "media_manifest.json")
        return 0

    for scenario_name in scenarios:
        base = pm.get_scenario(scenario_name)
        scenario = patch_scenario_for_recording(base, holo_cfg, add_viewport_capture=True)

        scenario_dir = out_dir / scenario_name.replace("/", "_")
        scenario_dir.mkdir(parents=True, exist_ok=True)

        orbit_png = scenario_dir / "orbit_keyframe.png"
        orbit_mp4 = scenario_dir / "orbit_viewport.mp4"
        move_png = scenario_dir / "move_keyframe.png"
        move_mp4 = scenario_dir / "move_viewport.mp4"
        move_fp_mp4 = scenario_dir / "move_leftcamera.mp4"

        manifest["scenarios"][scenario_name] = {
            "orbit_keyframe_png": str(orbit_png),
            "orbit_viewport_mp4": str(orbit_mp4),
            "move_keyframe_png": str(move_png),
            "move_viewport_mp4": str(move_mp4),
            "move_leftcamera_mp4": str(move_fp_mp4),
        }

        orbit_frames = max(1, int(round(cfg.orbit_seconds * cfg.fps)))
        move_frames = max(1, int(round(cfg.move_seconds * cfg.fps)))

        with holoocean.make(
            scenario_cfg=scenario,
            show_viewport=cfg.show_viewport,
            ticks_per_sec=cfg.ticks_per_sec,
            frames_per_sec=cfg.fps,
            verbose=False,
        ) as env:
            env.set_render_quality(int(cfg.render_quality))
            env.should_render_viewport(True)

            st0 = _warmup_until(env, ["ViewportCapture", "LeftCamera", "PoseSensor"], max_ticks=240)
            center = _pose_to_position(st0.get("PoseSensor", None)) or [0.0, 0.0, -5.0]

            viewport0 = _ensure_uint8_rgb(st0["ViewportCapture"])
            _write_png(orbit_png, viewport0)

            with _Mp4Writer(orbit_mp4, fps=cfg.fps, size_hw=viewport0.shape[:2]) as vw:
                for i in range(orbit_frames):
                    a = 2.0 * math.pi * (i / float(orbit_frames))
                    x = center[0] + cfg.orbit_radius_m * math.cos(a)
                    y = center[1] + cfg.orbit_radius_m * math.sin(a)
                    z = center[2] + cfg.orbit_height_m
                    env.move_viewport([x, y, z], _look_at_rpy([x, y, z], center))
                    st = env.tick(num_ticks=1, publish=False)
                    if st.get("ViewportCapture") is None:
                        continue
                    vw.append_rgb(_ensure_uint8_rgb(st["ViewportCapture"]))

            import numpy as np

            action = np.zeros((8,), dtype=np.float32)
            action[4:8] = 18.0

            fp0 = _ensure_uint8_rgb(st0["LeftCamera"])
            with _Mp4Writer(move_mp4, fps=cfg.fps, size_hw=viewport0.shape[:2]) as vw, _Mp4Writer(
                move_fp_mp4, fps=cfg.fps, size_hw=fp0.shape[:2]
            ) as fw:
                last = st0
                for i in range(move_frames):
                    pos = _pose_to_position(last.get("PoseSensor", None))
                    if pos is not None:
                        px, py, pz = pos
                        cx, cy, cz = px - 10.0, py - 10.0, pz + 3.5
                        env.move_viewport([cx, cy, cz], _look_at_rpy([cx, cy, cz], [px, py, pz]))

                    last = env.step(action, ticks=1, publish=False)
                    if i == 0 and last.get("ViewportCapture") is not None:
                        _write_png(move_png, _ensure_uint8_rgb(last["ViewportCapture"]))
                    if last.get("ViewportCapture") is not None:
                        vw.append_rgb(_ensure_uint8_rgb(last["ViewportCapture"]))
                    if last.get("LeftCamera") is not None:
                        fw.append_rgb(_ensure_uint8_rgb(last["LeftCamera"]))

    (out_dir / "media_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[h3] wrote:", out_dir / "media_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
