from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path
from typing import Any

import numpy as np

from .build_compact_bundle import build_compact_bundle
from .build_media_package import build_media_package
from .drift import DriftConfig
from .export_demo_assets import ExportConfig, export_demo_assets
from .export_s1_compatible_metrics import CompatConfig, export_s1_compatible_metrics
from .publish_e2_demo_assets import publish_e2_demo_assets
from .runner import RunConfig, run_habitat_ocean_proxy


@dataclass(frozen=True)
class BatchCase:
    name: str
    use_drift_cache: bool = False
    obstacle_proxy_mode: str = "off"
    preset: str = "compact"
    drift: DriftConfig = field(default_factory=DriftConfig)


@dataclass
class BatchConfig:
    cases: list[BatchCase]
    episodes: int = 1
    max_steps: int = 40
    seed: int = 42
    output_root: str | None = None
    drift_cache_path: str | None = None
    drift_origin_lat: float | None = None
    drift_origin_lon: float | None = None
    write_video: bool = False
    video_fps: float = 12.0
    bundle_screenshot_count: int = 2
    bundle_topdown_count: int = 2
    bundle_include_video: bool = False
    build_best_media_package: bool = False
    best_media_output_dir: str | None = None
    publish_best_e2: bool = False
    e2_target_dir: str | None = None


def default_case_library() -> dict[str, BatchCase]:
    return {
        "synthetic_compact": BatchCase(
            name="synthetic_compact",
            use_drift_cache=False,
            obstacle_proxy_mode="off",
            preset="compact",
            drift=DriftConfig(mode="synthetic_wave", amplitude_mps=0.35),
        ),
        "cache_compact": BatchCase(
            name="cache_compact",
            use_drift_cache=True,
            obstacle_proxy_mode="off",
            preset="compact",
            drift=DriftConfig(mode="synthetic_wave", amplitude_mps=0.35),
        ),
        "cache_obstacle": BatchCase(
            name="cache_obstacle",
            use_drift_cache=True,
            obstacle_proxy_mode="terminate",
            preset="compact",
            drift=DriftConfig(mode="synthetic_wave", amplitude_mps=0.35),
        ),
    }


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"oneocean_habitat_s2_batch_{stamp}"


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}")
    return payload


def _score_case(
    success_rate: float,
    final_distance_avg: float,
    avg_steps: float,
    obstacle_terminated_rate: float,
) -> float:
    return (
        100.0 * success_rate
        - final_distance_avg
        - 0.2 * avg_steps
        - 10.0 * obstacle_terminated_rate
    )


def _write_summary_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8") as file:
            file.write("case_name,status\n")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_batch_regression(config: BatchConfig) -> dict[str, Any]:
    output_root = Path(config.output_root) if config.output_root else _default_output_root()
    output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    complete_records: list[dict[str, Any]] = []

    for index, case in enumerate(config.cases):
        case_name = case.name.strip()
        case_dir = output_root / f"{index:02d}_{case_name}"
        if case.use_drift_cache and not config.drift_cache_path:
            records.append(
                {
                    "case_name": case_name,
                    "status": "skipped",
                    "reason": "drift_cache_path_missing",
                    "run_dir": str(case_dir),
                }
            )
            continue

        run_cfg = RunConfig(
            episodes=config.episodes,
            max_steps=config.max_steps,
            seed=config.seed + index,
            output_dir=str(case_dir),
            preset=case.preset,
            write_video=config.write_video,
            video_fps=config.video_fps,
            drift=case.drift,
            drift_cache_path=config.drift_cache_path if case.use_drift_cache else None,
            drift_origin_lat=config.drift_origin_lat,
            drift_origin_lon=config.drift_origin_lon,
            obstacle_proxy_mode=case.obstacle_proxy_mode,
        )
        run_outputs = run_habitat_ocean_proxy(run_cfg)
        run_dir = Path(run_outputs["output_dir"]).resolve()

        demo_outputs = export_demo_assets(ExportConfig(run_dir=run_dir, episode=0))
        compat_outputs = export_s1_compatible_metrics(CompatConfig(run_dir=run_dir, episode=0))
        bundle_outputs = build_compact_bundle(
            run_dir=run_dir,
            screenshot_count=config.bundle_screenshot_count,
            topdown_count=config.bundle_topdown_count,
            include_video=config.bundle_include_video,
        )

        metrics = _load_json(Path(run_outputs["metrics_json"]).resolve())
        summary = metrics.get("summary", {})
        episodes = metrics.get("episodes", [])
        final_distances = [
            float(row.get("final_distance_to_goal", 0.0))
            for row in episodes
            if isinstance(row, dict)
        ]
        final_distance_avg = float(np.mean(final_distances)) if final_distances else 0.0

        success_rate = float(summary.get("success_rate", 0.0))
        avg_steps = float(summary.get("avg_steps", 0.0))
        obstacle_terminated_rate = float(summary.get("obstacle_terminated_rate", 0.0))

        compat_payload = _load_json(Path(compat_outputs["metrics_json"]).resolve())
        compat_episodes = compat_payload.get("episodes", [])
        compat_episode = compat_episodes[0] if compat_episodes else {}
        path_efficiency = float(compat_episode.get("path_efficiency", 0.0))
        energy_proxy = float(compat_episode.get("energy_proxy", 0.0))

        score = _score_case(
            success_rate=success_rate,
            final_distance_avg=final_distance_avg,
            avg_steps=avg_steps,
            obstacle_terminated_rate=obstacle_terminated_rate,
        )

        record = {
            "case_name": case_name,
            "status": "ok",
            "run_dir": str(run_dir),
            "metrics_json": str(Path(run_outputs["metrics_json"]).resolve()),
            "demo_map_json": str(Path(demo_outputs["map_json"]).resolve()),
            "demo_path_json": str(Path(demo_outputs["path_json"]).resolve()),
            "compat_json": str(Path(compat_outputs["metrics_json"]).resolve()),
            "bundle_manifest_json": str(Path(bundle_outputs["manifest_json"]).resolve()),
            "success_rate": success_rate,
            "avg_steps": avg_steps,
            "final_distance_avg": final_distance_avg,
            "obstacle_terminated_rate": obstacle_terminated_rate,
            "path_efficiency": path_efficiency,
            "energy_proxy": energy_proxy,
            "score": score,
        }
        records.append(record)
        complete_records.append(record)

    best_case_name = ""
    best_case_run_dir = ""
    best_case_score = float("-inf")
    if complete_records:
        best_case = max(complete_records, key=lambda row: float(row.get("score", float("-inf"))))
        best_case_name = str(best_case["case_name"])
        best_case_run_dir = str(best_case["run_dir"])
        best_case_score = float(best_case["score"])

    media_package_outputs: dict[str, str] | None = None
    if config.build_best_media_package and best_case_run_dir:
        media_package_outputs = build_media_package(
            run_dir=Path(best_case_run_dir),
            output_dir=Path(config.best_media_output_dir) if config.best_media_output_dir else None,
        )

    e2_publish_outputs: dict[str, str] | None = None
    if config.publish_best_e2 and best_case_run_dir:
        e2_publish_outputs = publish_e2_demo_assets(
            run_dir=best_case_run_dir,
            target_dir=config.e2_target_dir,
        )

    summary_csv = output_root / "batch_summary.csv"
    _write_summary_csv(records, summary_csv)

    manifest = {
        "output_root": str(output_root),
        "config": {
            "episodes": int(config.episodes),
            "max_steps": int(config.max_steps),
            "seed": int(config.seed),
            "drift_cache_path": str(Path(config.drift_cache_path).resolve()) if config.drift_cache_path else "",
            "build_best_media_package": bool(config.build_best_media_package),
            "best_media_output_dir": (
                str(Path(config.best_media_output_dir).resolve()) if config.best_media_output_dir else ""
            ),
            "publish_best_e2": bool(config.publish_best_e2),
            "e2_target_dir": (
                str(Path(config.e2_target_dir).resolve()) if config.e2_target_dir else ""
            ),
        },
        "cases": records,
        "best_case": {
            "name": best_case_name,
            "run_dir": best_case_run_dir,
            "score": best_case_score,
        },
        "media_package": media_package_outputs or {},
        "e2_publish": e2_publish_outputs or {},
        "summary_csv": str(summary_csv),
    }

    manifest_path = output_root / "batch_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    return {
        "output_root": str(output_root),
        "manifest_json": str(manifest_path),
        "summary_csv": str(summary_csv),
        "best_case_name": best_case_name,
        "best_case_run_dir": best_case_run_dir,
    }
