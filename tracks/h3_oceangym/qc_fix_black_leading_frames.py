from __future__ import annotations

import argparse
import os
from pathlib import Path


def _fix_one(path: Path, *, mean_threshold: float, max_drop: int) -> dict:
    import cv2  # type: ignore
    import numpy as np

    path = path.resolve()
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return {"path": str(path), "ok": False, "reason": "cannot_open"}

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 20.0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if w <= 0 or h <= 0:
        cap.release()
        return {"path": str(path), "ok": False, "reason": "bad_shape"}

    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        frames.append(frame)
    cap.release()

    if not frames:
        return {"path": str(path), "ok": False, "reason": "no_frames"}

    drop = 0
    for f in frames[: max_drop + 1]:
        if float(f.mean()) >= mean_threshold:
            break
        drop += 1

    if drop == 0:
        return {"path": str(path), "ok": True, "changed": False, "dropped": 0, "frames": len(frames)}

    kept = frames[drop:]
    if not kept:
        return {"path": str(path), "ok": False, "reason": "all_frames_black", "dropped": drop, "frames": len(frames)}

    # Keep `.mp4` extension so OpenCV picks a valid container/codec.
    tmp = path.with_name(path.stem + ".tmp" + path.suffix)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(tmp), fourcc, fps, (w, h))
    if not writer.isOpened():
        return {"path": str(path), "ok": False, "reason": "writer_open_failed"}

    for f in kept:
        writer.write(f)
    writer.release()

    os.replace(tmp, path)
    return {"path": str(path), "ok": True, "changed": True, "dropped": drop, "frames": len(frames), "kept": len(kept)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("mp4", nargs="+")
    ap.add_argument("--mean-threshold", type=float, default=2.0)
    ap.add_argument("--max-drop", type=int, default=10)
    args = ap.parse_args()

    results = []
    for p in args.mp4:
        results.append(_fix_one(Path(p), mean_threshold=float(args.mean_threshold), max_drop=int(args.max_drop)))

    changed = [r for r in results if r.get("ok") and r.get("changed")]
    print(f"[qc] processed {len(results)} files; changed={len(changed)}")
    for r in results:
        if not r.get("ok"):
            print("[qc] FAIL", r)
        elif r.get("changed"):
            print("[qc] FIXED", r)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
