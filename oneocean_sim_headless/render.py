from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np


def _read_pose_csv(path: Path) -> np.ndarray:
    rows = []
    with path.open("r", encoding="utf-8", newline="") as fp:
        r = csv.reader(fp)
        header = next(r)
        if header[:4] != ["t", "x", "y", "z"]:
            raise ValueError(f"Unexpected pose header in {path}: {header[:8]}")
        for row in r:
            if not row:
                continue
            t = float(row[0])
            x = float(row[1])
            y = float(row[2])
            z = float(row[3])
            rows.append((t, x, y, z))
    return np.asarray(rows, dtype=np.float64)  # (T,4)


def _draw_disk(img: np.ndarray, *, cx: int, cy: int, r: int, color: tuple[int, int, int]) -> None:
    h, w = img.shape[:2]
    r = int(max(1, r))
    x0 = max(0, cx - r)
    x1 = min(w - 1, cx + r)
    y0 = max(0, cy - r)
    y1 = min(h - 1, cy + r)
    if x1 < x0 or y1 < y0:
        return
    yy, xx = np.ogrid[y0 : y1 + 1, x0 : x1 + 1]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
    # Important: avoid chained advanced indexing (it writes into a temporary copy).
    sub = img[y0 : y1 + 1, x0 : x1 + 1]
    sub[mask] = np.array(color, dtype=np.uint8)


def _map_xz_to_px(x: float, z: float, *, lo: np.ndarray, hi: np.ndarray, width: int, height: int) -> tuple[int, int]:
    # x=east, z=north; map to image with origin at top-left; z increases downward in pixels.
    u = (float(x) - float(lo[0])) / max(1e-9, float(hi[0] - lo[0]))
    v = (float(z) - float(lo[2])) / max(1e-9, float(hi[2] - lo[2]))
    px = int(np.clip(round(u * (width - 1)), 0, width - 1))
    py = int(np.clip(round(v * (height - 1)), 0, height - 1))
    return px, py


def _load_semantics_index(run_dir: Path) -> dict[int, dict[str, Any]]:
    path = run_dir / "environment_samples" / "semantics.jsonl"
    if not path.exists():
        return {}
    out: dict[int, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        t = float(obj.get("t", 0.0))
        # We'll quantize to a step index later using dt_s.
        out.setdefault(int(round(t * 1000.0)), obj)
    return out


def render_topdown_rollout(*, run_dir: str | Path, out_mp4: str | Path, out_keyframe: str | Path, stride: int = 2) -> None:
    """Render a compact top-down MP4 from the recorded pose streams.

    This is a headless visualization aid (paper/demo proof), not a photorealistic renderer.
    """
    import imageio.v2 as imageio  # optional dependency

    run_dir = Path(run_dir).expanduser().resolve()
    meta = json.loads((run_dir / "run_meta.json").read_text(encoding="utf-8"))
    lo = np.asarray(meta["bounds_xyz"]["lo"], dtype=np.float64)
    hi = np.asarray(meta["bounds_xyz"]["hi"], dtype=np.float64)
    n_agents = int(meta.get("n_agents", 1))

    poses = []
    for i in range(n_agents):
        p = run_dir / "agents" / f"agent_{i:03d}" / "pose_groundtruth" / "data.csv"
        poses.append(_read_pose_csv(p))
    t_len = int(min(p.shape[0] for p in poses))
    poses = [p[:t_len] for p in poses]

    # Semantics (optional, written sparsely).
    sem = _load_semantics_index(run_dir)

    width = 720
    height = 720
    stride = int(max(1, stride))
    frames = []

    colors = [
        (255, 80, 80),
        (80, 200, 255),
        (255, 220, 80),
        (140, 255, 120),
        (220, 120, 255),
        (255, 160, 40),
        (120, 180, 255),
        (255, 120, 180),
        (160, 255, 220),
        (220, 220, 220),
    ]

    for ti in range(0, t_len, stride):
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img[:, :, :] = np.array([10, 14, 22], dtype=np.uint8)  # dark ocean bg

        # Draw semantic objects if present near this time.
        key = int(round(float(poses[0][ti, 0]) * 1000.0))
        s = sem.get(key)
        if s is not None:
            if "cleanup_sources_xyz" in s:
                for p in s["cleanup_sources_xyz"]:
                    px, py = _map_xz_to_px(float(p[0]), float(p[2]), lo=lo, hi=hi, width=width, height=height)
                    _draw_disk(img, cx=px, cy=py, r=6, color=(255, 220, 80))
            if "barrel_xyz" in s:
                p = s["barrel_xyz"]
                px, py = _map_xz_to_px(float(p[0]), float(p[2]), lo=lo, hi=hi, width=width, height=height)
                _draw_disk(img, cx=px, cy=py, r=7, color=(255, 160, 40))
            if "fish_centroid_xyz" in s:
                p = s["fish_centroid_xyz"]
                px, py = _map_xz_to_px(float(p[0]), float(p[2]), lo=lo, hi=hi, width=width, height=height)
                _draw_disk(img, cx=px, cy=py, r=6, color=(120, 220, 255))
            if "leak_xyz" in s:
                for p in s["leak_xyz"]:
                    px, py = _map_xz_to_px(float(p[0]), float(p[2]), lo=lo, hi=hi, width=width, height=height)
                    _draw_disk(img, cx=px, cy=py, r=5, color=(255, 60, 60))

        # Draw agents + short trails.
        for ai, p in enumerate(poses):
            c = colors[ai % len(colors)]
            # trail
            t0 = max(0, ti - 40 * stride)
            for tj in range(t0, ti, max(1, 6 * stride)):
                x, z = float(p[tj, 1]), float(p[tj, 3])
                px, py = _map_xz_to_px(x, z, lo=lo, hi=hi, width=width, height=height)
                _draw_disk(img, cx=px, cy=py, r=2, color=(40, 60, 90))
            x, z = float(p[ti, 1]), float(p[ti, 3])
            px, py = _map_xz_to_px(x, z, lo=lo, hi=hi, width=width, height=height)
            _draw_disk(img, cx=px, cy=py, r=5, color=c)

        frames.append(img)

    out_mp4 = Path(out_mp4).expanduser().resolve()
    out_mp4.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(out_mp4, frames, fps=20)

    out_keyframe = Path(out_keyframe).expanduser().resolve()
    out_keyframe.parent.mkdir(parents=True, exist_ok=True)
    mid = frames[len(frames) // 2] if frames else np.zeros((height, width, 3), dtype=np.uint8)
    imageio.imwrite(out_keyframe, mid)
