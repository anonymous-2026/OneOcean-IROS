from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path


def _ensure_ssl_cert_file() -> None:
    try:
        import certifi  # type: ignore

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _stable_seed(*parts: str | int) -> int:
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"|")
    return int(h.hexdigest()[:8], 16)


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

    def append_bgr(self, bgr) -> None:
        self._writer.write(bgr)

    def close(self) -> None:
        self._writer.release()


def _state_get(state: dict, agent: str, key: str, n_agents: int):
    if n_agents == 1:
        return state.get(key)
    return state.get(agent, {}).get(key)


def _look_at_rpy(camera_xyz: tuple[float, float, float], target_xyz: tuple[float, float, float]) -> list[float]:
    import math

    dx = target_xyz[0] - camera_xyz[0]
    dy = target_xyz[1] - camera_xyz[1]
    dz = target_xyz[2] - camera_xyz[2]
    yaw = math.degrees(math.atan2(dy, dx))
    dist_xy = math.hypot(dx, dy)
    pitch = math.degrees(math.atan2(dz, max(1e-6, dist_xy)))
    return [0.0, pitch, yaw]


def _run_containment(env, *, seed: int, ticks_per_sec: int, fps: int, n_agents: int, out_dir: Path) -> dict:
    import math
    import numpy as np

    rng = np.random.default_rng(seed)
    dt = 1.0 / float(ticks_per_sec)
    agent_names = [f"auv{i}" for i in range(n_agents)]

    state = env.tick(num_ticks=1, publish=False)
    pose0 = _state_get(state, "auv0", "PoseSensor", n_agents)
    if pose0 is None:
        raise RuntimeError("PoseSensor missing for auv0")
    p0 = np.asarray(pose0, dtype=np.float32)[:3, 3]
    src = (float(p0[0] + rng.uniform(-30.0, 30.0)), float(p0[1] + rng.uniform(-30.0, 30.0)), float(p0[2]))

    # Ring baseline.
    ring_r = 25.0
    ring = []
    for i in range(n_agents):
        a = 2.0 * math.pi * (i / float(n_agents))
        ring.append((src[0] + ring_r * math.cos(a), src[1] + ring_r * math.sin(a), src[2]))

    contain_leak_radius_m = 60.0
    contain_capture_radius_m = 6.0
    contain_spawn_per_step = 8
    current_u_mps = 0.25
    current_v_mps = 0.10

    particles = np.zeros((0, 3), dtype=np.float32)
    removed = 0
    leaked = 0
    diff_sigma = 0.5

    out_dir.mkdir(parents=True, exist_ok=True)
    vp_mp4_path = out_dir / "contain_viewport.mp4"
    fp_mp4_path = out_dir / "contain_leftcamera.mp4"

    # Viewport policy: orbit around the source so multi-agent structure stays visible.
    cam_r = max(55.0, ring_r * 2.2)
    cam_h = 18.0

    # Warm up viewport and start recording only when non-black.
    writer_vp = None
    writer_fp = None
    for _ in range(300):
        env.move_viewport(
            [src[0] + cam_r, src[1], src[2] + cam_h],
            _look_at_rpy((src[0] + cam_r, src[1], src[2] + cam_h), src),
        )
        state = env.tick(num_ticks=1, publish=False)
        vp = _state_get(state, "auv0", "ViewportCapture", n_agents)
        fp = _state_get(state, "auv0", "LeftCamera", n_agents)
        if vp is None or fp is None:
            continue
        vp0 = _ensure_uint8_rgb(vp)
        if float(vp0.mean()) < 2.0:
            continue
        import cv2  # type: ignore

        writer_vp = _Mp4Writer(vp_mp4_path, fps=fps, size_hw=vp0.shape[:2])
        writer_vp.append_bgr(cv2.cvtColor(vp0, cv2.COLOR_RGB2BGR))
        fp0 = _ensure_uint8_rgb(fp)
        writer_fp = _Mp4Writer(fp_mp4_path, fps=fps, size_hw=fp0.shape[:2])
        writer_fp.append_bgr(cv2.cvtColor(fp0, cv2.COLOR_RGB2BGR))
        break

    steps = 300
    for step in range(steps):
        theta = 2.0 * math.pi * (step / float(max(1, steps)))
        cx = src[0] + cam_r * math.cos(theta)
        cy = src[1] + cam_r * math.sin(theta)
        cz = src[2] + cam_h
        env.move_viewport([cx, cy, cz], _look_at_rpy((cx, cy, cz), src))

        spawn = rng.normal(0.0, 2.0, size=(contain_spawn_per_step, 3)).astype(np.float32)
        spawn[:, 2] = 0.0
        src_xyz = np.array([src[0], src[1], src[2]], dtype=np.float32)
        particles = np.concatenate([particles, src_xyz + spawn], axis=0)

        if particles.size:
            particles[:, 0] += current_u_mps * dt
            particles[:, 1] += current_v_mps * dt
            particles += rng.normal(0.0, diff_sigma * math.sqrt(dt), size=particles.shape).astype(np.float32)

        for i, name in enumerate(agent_names):
            tx, ty, tz = ring[i]
            env.act(name, np.array([tx, ty, tz, 0.0, 0.0, 0.0], dtype=np.float32))

        state = env.tick(num_ticks=1, publish=False)

        if writer_vp is not None:
            vp = _state_get(state, "auv0", "ViewportCapture", n_agents)
            if vp is not None:
                import cv2  # type: ignore

                writer_vp.append_bgr(cv2.cvtColor(_ensure_uint8_rgb(vp), cv2.COLOR_RGB2BGR))
        if writer_fp is not None:
            fp = _state_get(state, "auv0", "LeftCamera", n_agents)
            if fp is not None:
                import cv2  # type: ignore

                writer_fp.append_bgr(cv2.cvtColor(_ensure_uint8_rgb(fp), cv2.COLOR_RGB2BGR))

        if particles.size:
            keep = np.ones((particles.shape[0],), dtype=bool)
            for name in agent_names:
                pose = _state_get(state, name, "PoseSensor", n_agents)
                if pose is None:
                    continue
                p = np.asarray(pose, dtype=np.float32)[:3, 3]
                d2 = (particles[:, 0] - p[0]) ** 2 + (particles[:, 1] - p[1]) ** 2 + (particles[:, 2] - p[2]) ** 2
                hit = d2 <= (contain_capture_radius_m**2)
                if hit.any():
                    keep &= ~hit
            n_before = particles.shape[0]
            particles = particles[keep]
            removed += int(n_before - particles.shape[0])

        if particles.size:
            dx = particles[:, 0] - src[0]
            dy = particles[:, 1] - src[1]
            far = (dx * dx + dy * dy) >= (contain_leak_radius_m**2)
            if far.any():
                leaked += int(far.sum())
                particles = particles[~far]

    if writer_vp is not None:
        writer_vp.close()
    if writer_fp is not None:
        writer_fp.close()

    success = (leaked <= 0.25 * removed) if removed > 0 else (leaked == 0)
    return {
        "task": "plume_containment_multiagent",
        "seed": int(seed),
        "n_agents": int(n_agents),
        "source_xyz": [float(src[0]), float(src[1]), float(src[2])],
        "steps": int(steps),
        "time_s": float(steps * dt),
        "removed_particles": int(removed),
        "leaked_particles": int(leaked),
        "success": bool(success),
        "media": {"viewport_mp4": str(vp_mp4_path), "leftcamera_mp4": str(fp_mp4_path)},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Existing task suite output root to update.")
    ap.add_argument("--preset", default="ocean_worlds_camera")
    ap.add_argument("--scenario", default=None, help="Optional single scenario name to rerun.")
    ap.add_argument("--n_multiagent", type=int, default=8)
    ap.add_argument("--gl_version", type=int, default=3, choices=(3, 4))
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    _ensure_ssl_cert_file()

    import holoocean  # type: ignore
    from holoocean import packagemanager as pm  # type: ignore
    from holoocean.holoocean import GL_VERSION  # type: ignore

    from tracks.h3_oceangym.holoocean_patch import HoloCfg, add_hovering_auv_agents, patch_scenario_for_recording
    from tracks.h3_oceangym.scenarios import scenario_preset

    out_root = Path(args.out_dir).resolve()
    root_manifest_path = out_root / "results_manifest.json"
    if not root_manifest_path.exists():
        raise FileNotFoundError(root_manifest_path)

    root_manifest = json.loads(root_manifest_path.read_text(encoding="utf-8"))
    suite_cfg = root_manifest.get("cfg", {})
    ticks_per_sec = int(suite_cfg.get("ticks_per_sec", 20))
    fps = int(suite_cfg.get("fps", 20))

    holo_cfg = HoloCfg(
        ticks_per_sec=ticks_per_sec,
        fps=fps,
        window_width=1280,
        window_height=720,
        render_quality=int(suite_cfg.get("render_quality", 3)),
        show_viewport=bool(suite_cfg.get("show_viewport", False)),
    )

    scenarios = [args.scenario] if args.scenario else scenario_preset(args.preset)
    n_agents = int(args.n_multiagent)

    for scenario_name in scenarios:
        print("[h3] rerun containment:", scenario_name)
        base = pm.get_scenario(scenario_name)
        scenario = patch_scenario_for_recording(base, holo_cfg, add_viewport_capture=True)
        scenario = add_hovering_auv_agents(scenario, n_agents=n_agents)
        for a in scenario.get("agents", []):
            a["control_scheme"] = 1

        seed = _stable_seed(scenario_name, "plume_containment_multiagent", 0)
        ep_dir = out_root / scenario_name.replace("/", "_") / "plume_containment_multiagent" / "ep000"

        with holoocean.make(
            scenario_cfg=scenario,
            show_viewport=bool(suite_cfg.get("show_viewport", False)),
            ticks_per_sec=ticks_per_sec,
            frames_per_sec=fps,
            gl_version=(GL_VERSION.OPENGL3 if int(args.gl_version) == 3 else GL_VERSION.OPENGL4),
            verbose=bool(args.verbose),
        ) as env:
            env.set_render_quality(int(suite_cfg.get("render_quality", 3)))
            env.should_render_viewport(True)
            res = _run_containment(env, seed=seed, ticks_per_sec=ticks_per_sec, fps=fps, n_agents=n_agents, out_dir=ep_dir)

        # Update per-task results+media manifests.
        task_dir = ep_dir.parent
        per_task = {
            "task": "plume_containment_multiagent",
            "n_agents": int(n_agents),
            "episodes": [res],
        }
        (task_dir / "results_manifest.json").write_text(json.dumps(per_task, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        (task_dir / "media_manifest.json").write_text(
            json.dumps(
                {
                    "ep000_viewport_mp4": res["media"]["viewport_mp4"],
                    "ep000_leftcamera_mp4": res["media"]["leftcamera_mp4"],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )

        # Update root results manifest in-place (replace the per-task entry).
        scen_block = root_manifest.get("scenarios", {}).get(scenario_name)
        if isinstance(scen_block, dict):
            eps = scen_block.get("episodes", [])
            if isinstance(eps, list):
                for i, t in enumerate(eps):
                    if isinstance(t, dict) and t.get("task") == "plume_containment_multiagent":
                        eps[i] = per_task
                        break

    root_manifest_path.write_text(json.dumps(root_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[h3] updated containment task under:", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
