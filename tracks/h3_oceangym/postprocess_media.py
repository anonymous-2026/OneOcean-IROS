from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _have_imageio() -> bool:
    try:
        import imageio.v2  # noqa: F401

        return True
    except Exception:
        return False


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(delete=False, dir=str(path.parent), prefix=path.name + ".", suffix=".tmp") as f:
        f.write(data)
        tmp = Path(f.name)
    tmp.replace(path)


def _atomic_copyfile(src: Path, dst: Path) -> None:
    _atomic_write_bytes(dst, src.read_bytes())


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _cv2() -> object:
    import cv2  # type: ignore

    return cv2


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _keyframe_indices(frame_count: int) -> list[int]:
    if frame_count <= 1:
        return [0]
    return sorted({0, frame_count // 2, frame_count - 1})


def _write_keyframes(mp4: Path) -> list[Path]:
    cv2 = _cv2()
    cap = cv2.VideoCapture(str(mp4))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    idxs = _keyframe_indices(frame_count)
    out_paths: list[Path] = []
    for idx in idxs:
        out = mp4.parent / f"{mp4.stem}_keyframe_{idx:03d}.png"
        if out.exists():
            out_paths.append(out)
            continue
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        _ensure_dir(out.parent)
        cv2.imwrite(str(out), frame)
        out_paths.append(out)
    cap.release()
    return out_paths


@dataclass(frozen=True)
class GifCfg:
    fps: int = 10
    max_frames: int = 90
    max_width: int = 640


def _write_gif_from_mp4(mp4: Path, *, cfg: GifCfg) -> Path | None:
    if not _have_imageio():
        return None

    import imageio.v2 as imageio  # type: ignore
    import numpy as np

    cv2 = _cv2()
    cap = cv2.VideoCapture(str(mp4))
    fps_in = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if frame_count <= 0:
        cap.release()
        return None

    stride = 1
    if fps_in > 1e-3:
        stride = max(1, int(round(fps_in / float(cfg.fps))))
    max_take = int(cfg.max_frames * stride)

    gif_path = mp4.with_suffix(".gif")
    if gif_path.exists():
        cap.release()
        return gif_path

    writer = imageio.get_writer(
        str(gif_path),
        mode="I",
        fps=int(cfg.fps),
        loop=0,
        palettesize=256,
        subrectangles=True,
    )
    try:
        taken = 0
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % stride != 0:
                i += 1
                continue
            if taken >= max_take:
                break

            h, w = frame.shape[:2]
            if w > int(cfg.max_width):
                scale = float(cfg.max_width) / float(w)
                frame = cv2.resize(frame, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            writer.append_data(np.asarray(rgb, dtype=np.uint8))
            taken += 1
            i += 1
    finally:
        cap.release()
        writer.close()

    return gif_path


def _postprocess_root(root: Path, *, gif_cfg: GifCfg) -> dict:
    root = root.resolve()
    mp4s = sorted(p for p in root.rglob("*.mp4") if p.is_file())
    outputs: dict[str, dict] = {}

    for mp4 in mp4s:
        kfs = _write_keyframes(mp4)
        gif = _write_gif_from_mp4(mp4, cfg=gif_cfg)
        outputs[str(mp4)] = {
            "keyframes_png": [str(p) for p in kfs],
            "gif": str(gif) if gif is not None else None,
        }

    manifest = {
        "tool": "tracks/h3_oceangym/postprocess_media.py",
        "python": sys.executable,
        "cwd": os.getcwd(),
        "root": str(root),
        "gif_cfg": {"fps": gif_cfg.fps, "max_frames": gif_cfg.max_frames, "max_width": gif_cfg.max_width},
        "count_mp4": len(mp4s),
        "outputs": outputs,
        "notes": (
            "If gif is null, run this script with a Python that has imageio+Pillow installed "
            "(e.g., /home/shuaijun/miniconda3/bin/python)."
        ),
    }
    _write_json(root / "postprocess_media_manifest.json", manifest)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="+", required=True, help="One or more run output directories to postprocess.")
    ap.add_argument("--gif_fps", type=int, default=GifCfg.fps)
    ap.add_argument("--gif_max_frames", type=int, default=GifCfg.max_frames)
    ap.add_argument("--gif_max_width", type=int, default=GifCfg.max_width)
    args = ap.parse_args()

    gif_cfg = GifCfg(fps=int(args.gif_fps), max_frames=int(args.gif_max_frames), max_width=int(args.gif_max_width))
    for r in args.roots:
        root = Path(r)
        if not root.exists():
            raise FileNotFoundError(root)
        m = _postprocess_root(root, gif_cfg=gif_cfg)
        print("[h3] wrote:", Path(m["root"]) / "postprocess_media_manifest.json")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

