from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Iterable

import imageio.v2 as imageio
import numpy as np
from PIL import Image


def _sample_evenly(items: list[Path], max_count: int) -> list[Path]:
    if max_count <= 0 or not items:
        return []
    if len(items) <= max_count:
        return items
    if max_count == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (max_count - 1)) for i in range(max_count)]
    sampled: list[Path] = []
    used = set()
    for idx in indices:
        idx = int(max(0, min(len(items) - 1, idx)))
        if idx in used:
            continue
        used.add(idx)
        sampled.append(items[idx])
    return sampled


def _copy_files(files: Iterable[Path], target_dir: Path) -> list[str]:
    target_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source in files:
        if source.exists():
            destination = target_dir / source.name
            shutil.copy2(source, destination)
            copied.append(str(destination))
    return copied


def _build_contact_sheet(image_paths: list[Path], output_path: Path) -> str | None:
    if not image_paths:
        return None
    images: list[Image.Image] = []
    for path in image_paths:
        if path.exists():
            with Image.open(path) as image:
                images.append(image.convert("RGB"))
    if not images:
        return None

    target_w, target_h = 640, 360
    resample = getattr(getattr(Image, "Resampling", Image), "BILINEAR")
    resized = [img.resize((target_w, target_h), resample) for img in images]
    cols = 2
    rows = int(np.ceil(len(resized) / cols))
    sheet = Image.new("RGB", (cols * target_w, rows * target_h), color=(8, 20, 35))
    for index, frame in enumerate(resized):
        row = index // cols
        col = index % cols
        sheet.paste(frame, (col * target_w, row * target_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path)
    return str(output_path)


def _video_to_gif(
    video_path: Path,
    output_path: Path,
    fps: float,
    max_frames: int,
) -> str | None:
    if not video_path.exists():
        return None
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    source_fps = float(meta.get("fps", 12.0))
    source_fps = max(1.0, source_fps)
    stride = max(1, int(round(source_fps / max(1.0, fps))))
    frames = []
    for index, frame in enumerate(reader):
        if index % stride != 0:
            continue
        frames.append(frame)
        if len(frames) >= max_frames:
            break
    reader.close()
    if not frames:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=max(1.0, fps))
    return str(output_path)


def _images_to_gif(
    image_paths: list[Path],
    output_path: Path,
    fps: float,
    max_frames: int,
) -> str | None:
    selected = _sample_evenly([path for path in image_paths if path.exists()], max_frames)
    if not selected:
        return None
    frames = [imageio.imread(path) for path in selected]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(output_path, frames, fps=max(1.0, fps))
    return str(output_path)


def build_media_package(
    run_dir: Path,
    output_dir: Path | None = None,
    scene_count: int = 3,
    progress_count: int = 4,
    topdown_count: int = 3,
    gif_fps: float = 8.0,
    gif_max_frames: int = 80,
) -> dict[str, str]:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    out_dir = (output_dir or (run_dir / "media_package")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    screenshots = sorted((run_dir / "screenshots").glob("*.png"))
    topdowns = sorted((run_dir / "topdown").glob("*.png"))
    videos = sorted((run_dir / "videos").glob("*.mp4"))

    scene_images = _sample_evenly(screenshots, scene_count)
    progress_images = _sample_evenly(screenshots, progress_count)
    topdown_images = _sample_evenly(topdowns, topdown_count)

    copied_scene = _copy_files(scene_images, out_dir / "scene")
    copied_progress = _copy_files(progress_images, out_dir / "progress")
    copied_topdown = _copy_files(topdown_images, out_dir / "topdown")

    copied_video: list[str] = []
    if videos:
        copied_video = _copy_files([videos[0]], out_dir / "video")

    gif_path = out_dir / "gifs" / "hero_rollout.gif"
    if copied_video:
        gif_output = _video_to_gif(
            video_path=Path(copied_video[0]),
            output_path=gif_path,
            fps=gif_fps,
            max_frames=gif_max_frames,
        )
    else:
        gif_output = _images_to_gif(
            image_paths=progress_images,
            output_path=gif_path,
            fps=gif_fps,
            max_frames=gif_max_frames,
        )

    contact_sheet_output = _build_contact_sheet(
        image_paths=(scene_images + progress_images)[:6],
        output_path=out_dir / "scene_overview.png",
    )

    metrics_json = run_dir / "metrics.json"
    run_config_json = run_dir / "run_config.json"
    copied_metadata = _copy_files([metrics_json, run_config_json], out_dir / "metadata")

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "source_run_dir": str(run_dir),
        "media_counts": {
            "scene_images": len(copied_scene),
            "progress_images": len(copied_progress),
            "topdown_images": len(copied_topdown),
            "videos": len(copied_video),
            "gif": 1 if gif_output else 0,
            "metadata": len(copied_metadata),
        },
        "paths": {
            "scene_images": copied_scene,
            "progress_images": copied_progress,
            "topdown_images": copied_topdown,
            "videos": copied_video,
            "gif": gif_output or "",
            "scene_overview_png": contact_sheet_output or "",
            "metadata": copied_metadata,
        },
        "notes": [
            "Packaged for website/demo/paper qualitative evidence.",
            "GIF is derived from run MP4 when available, otherwise from progress screenshots.",
        ],
    }
    manifest_path = out_dir / "media_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return {
        "media_dir": str(out_dir),
        "media_manifest_json": str(manifest_path),
        "hero_gif": gif_output or "",
        "hero_video": copied_video[0] if copied_video else "",
    }
