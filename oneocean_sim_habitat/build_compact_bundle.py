from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Iterable


def _sample_evenly(items: list[Path], max_count: int) -> list[Path]:
    if max_count <= 0 or not items:
        return []
    if len(items) <= max_count:
        return items
    if max_count == 1:
        return [items[len(items) // 2]]
    indices = [round(i * (len(items) - 1) / (max_count - 1)) for i in range(max_count)]
    used = set()
    sampled: list[Path] = []
    for idx in indices:
        idx = int(max(0, min(len(items) - 1, idx)))
        if idx in used:
            continue
        sampled.append(items[idx])
        used.add(idx)
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


def build_compact_bundle(
    run_dir: Path,
    output_dir: Path | None = None,
    screenshot_count: int = 4,
    topdown_count: int = 6,
    include_video: bool = True,
) -> dict[str, str]:
    run_dir = run_dir.resolve()
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")
    out_dir = (output_dir or (run_dir / "compact_bundle")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    copied_metrics = _copy_files(
        [run_dir / "metrics.json", run_dir / "run_config.json"],
        out_dir / "metadata",
    )

    screenshots = sorted((run_dir / "screenshots").glob("*.png"))
    topdowns = sorted((run_dir / "topdown").glob("*.png"))
    sampled_screens = _sample_evenly(screenshots, screenshot_count)
    sampled_topdown = _sample_evenly(topdowns, topdown_count)
    copied_screens = _copy_files(sampled_screens, out_dir / "screenshots")
    copied_topdown = _copy_files(sampled_topdown, out_dir / "topdown")

    copied_video: list[str] = []
    if include_video:
        videos = sorted((run_dir / "videos").glob("*.mp4"))
        if videos:
            copied_video = _copy_files([videos[0]], out_dir / "videos")

    copied_demo_export: list[str] = []
    demo_export = run_dir / "demo_export"
    if demo_export.exists():
        copied_demo_export = _copy_files(
            [
                demo_export / "drone_map_data.json",
                demo_export / "drone_path_data.json",
                demo_export / "assets_manifest.json",
            ],
            out_dir / "demo_export",
        )

    copied_compat: list[str] = []
    compat = run_dir / "compat"
    if compat.exists():
        copied_compat = _copy_files(
            [compat / "metrics_s1_compat.json", compat / "metrics_s1_compat.csv"],
            out_dir / "compat",
        )

    manifest = {
        "source_run_dir": str(run_dir),
        "bundle_dir": str(out_dir),
        "counts": {
            "metadata": len(copied_metrics),
            "screenshots": len(copied_screens),
            "topdown": len(copied_topdown),
            "videos": len(copied_video),
            "demo_export": len(copied_demo_export),
            "compat": len(copied_compat),
        },
        "paths": {
            "metadata": copied_metrics,
            "screenshots": copied_screens,
            "topdown": copied_topdown,
            "videos": copied_video,
            "demo_export": copied_demo_export,
            "compat": copied_compat,
        },
    }
    manifest_path = out_dir / "compact_bundle_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return {
        "bundle_dir": str(out_dir),
        "manifest_json": str(manifest_path),
    }
