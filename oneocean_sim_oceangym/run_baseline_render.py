import argparse
import json
import os
import time
from pathlib import Path


def _ensure_ssl_cert_file() -> None:
    try:
        import certifi  # type: ignore

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass


def _json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="Dam-Hovering", help="HoloOcean scenario name")
    ap.add_argument("--steps", type=int, default=300, help="Number of ticks")
    ap.add_argument("--fps", type=int, default=20, help="Video FPS")
    ap.add_argument(
        "--out_dir",
        default=None,
        help="Output directory (default: runs/oceangym_h3/baseline_<timestamp>/)",
    )
    args = ap.parse_args()

    _ensure_ssl_cert_file()

    import numpy as np
    import cv2  # type: ignore
    import holoocean  # type: ignore
    from holoocean import packagemanager as pm  # type: ignore

    installed = pm.installed_packages()
    if "Ocean" not in installed:
        holoocean.install("Ocean")

    scenario_name = args.scenario
    scenario_cfg = pm.get_scenario(scenario_name)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir) if args.out_dir else Path("runs") / "oceangym_h3" / f"baseline_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "track": "H3 (OceanGym)",
        "mode": "baseline_render_only",
        "scenario": scenario_name,
        "steps": args.steps,
        "fps": args.fps,
        "outputs": {},
    }

    # Pick a camera to record
    camera_candidates = [
        "FrontCamera",
        "RGBCamera",
        "LeftCamera",
        "RightCamera",
        "DownCamera",
        "UpCamera",
        "BackCamera",
    ]

    with holoocean.make(scenario_name, scenario_cfg=scenario_cfg, start_world=False) as env:
        state0 = env.tick()
        cam_key = next((k for k in camera_candidates if k in state0), None)
        if cam_key is None:
            raise RuntimeError(f"No RGB camera found in state keys: {sorted(state0.keys())[:40]}")

        frame0 = state0[cam_key]
        if frame0.shape[-1] == 4:
            frame0 = frame0[:, :, :3]
        h, w = frame0.shape[:2]

        mp4_path = out_dir / "rollout.mp4"
        png_path = out_dir / "scene.png"
        manifest["outputs"]["video_mp4"] = str(mp4_path)
        manifest["outputs"]["screenshot_png"] = str(png_path)
        manifest["camera_key"] = cam_key
        manifest["resolution"] = [int(w), int(h)]

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(mp4_path), fourcc, float(args.fps), (w, h))
        if not writer.isOpened():
            raise RuntimeError("Failed to open cv2.VideoWriter (mp4v).")

        # Save first screenshot
        cv2.imwrite(str(png_path), cv2.cvtColor(frame0, cv2.COLOR_RGB2BGR))

        # Simple action: small random thrusts to ensure visible motion
        rng = np.random.default_rng(0)
        for _ in range(args.steps):
            cmd = rng.normal(0.0, 10.0, size=(8,)).astype(np.float32)
            env.act("auv0", cmd)
            state = env.tick()
            frame = state.get(cam_key)
            if frame is None:
                continue
            if frame.shape[-1] == 4:
                frame = frame[:, :, :3]
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        writer.release()

    _json_dump(out_dir / "media_manifest.json", manifest)
    print("[oceangym] wrote:", mp4_path)
    print("[oceangym] wrote:", png_path)
    print("[oceangym] wrote:", out_dir / "media_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

