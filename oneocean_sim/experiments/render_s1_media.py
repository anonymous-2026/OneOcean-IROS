from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from io import BytesIO
import json
from pathlib import Path
import re
import shutil
from typing import Any

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render S1 screenshots/videos/gifs from a matrix-style run")
    parser.add_argument("--matrix-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--case-limit", type=int, default=8)
    parser.add_argument("--all-cases", action="store_true")
    parser.add_argument("--max-frames", type=int, default=90)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--screenshot-count", type=int, default=3)
    return parser.parse_args()


def _slug(text: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", text).strip("-").lower()
    return normalized or "case"


def _load_trajectory_csv(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            parsed: dict[str, float] = {}
            for key, value in row.items():
                if value is None or value == "":
                    continue
                parsed[key] = float(value)
            rows.append(parsed)
    if not rows:
        raise ValueError(f"Empty trajectory file: {path}")
    return rows


def _select_cases(cases: list[dict[str, Any]], case_limit: int, all_cases: bool) -> list[dict[str, Any]]:
    if all_cases:
        return cases

    priorities = [
        ("scene", "navigation", "compensated", "on"),
        ("scene", "station_keeping", "compensated", "on"),
        ("scene", "navigation", "naive", "on"),
        ("scene", "navigation", "compensated", "off"),
    ]

    selected: list[dict[str, Any]] = []
    selected_ids = set()
    by_key = {}
    for case in cases:
        key = (
            str(case.get("variant", "")),
            str(case.get("task", "")),
            str(case.get("controller_mode", "")),
            str(case.get("tide_mode", "on" if case.get("include_tides", True) else "off")),
        )
        by_key[key] = case
    for key in priorities:
        case = by_key.get(key)
        if case is None:
            continue
        case_id = int(case.get("case_id", -1))
        if case_id in selected_ids:
            continue
        selected.append(case)
        selected_ids.add(case_id)
        if len(selected) >= case_limit:
            return selected
    for case in cases:
        case_id = int(case.get("case_id", -1))
        if case_id in selected_ids:
            continue
        selected.append(case)
        selected_ids.add(case_id)
        if len(selected) >= case_limit:
            break
    return selected


def _sample_indices(length: int, count: int) -> list[int]:
    if length <= 0:
        return []
    count = max(1, min(length, count))
    if count == 1:
        return [length - 1]
    values = np.linspace(0, length - 1, num=count, dtype=int)
    deduped: list[int] = []
    used = set()
    for value in values:
        index = int(value)
        if index in used:
            continue
        used.add(index)
        deduped.append(index)
    if (length - 1) not in used:
        deduped.append(length - 1)
    deduped.sort()
    return deduped


def _bounds(rows: list[dict[str, float]]) -> tuple[tuple[float, float], tuple[float, float]]:
    x = np.asarray([row["x_m"] for row in rows], dtype=float)
    y = np.asarray([row["y_m"] for row in rows], dtype=float)
    goal_x = float(rows[0]["goal_x_m"])
    goal_y = float(rows[0]["goal_y_m"])
    x_min = min(float(np.min(x)), goal_x)
    x_max = max(float(np.max(x)), goal_x)
    y_min = min(float(np.min(y)), goal_y)
    y_max = max(float(np.max(y)), goal_y)
    pad_x = max(20.0, 0.15 * max(1.0, x_max - x_min))
    pad_y = max(20.0, 0.15 * max(1.0, y_max - y_min))
    return (x_min - pad_x, x_max + pad_x), (y_min - pad_y, y_max + pad_y)


def _figure_to_rgb_array(fig: plt.Figure) -> np.ndarray:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=128)
    buf.seek(0)
    arr = imageio.imread(buf)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[:, :, :3]
    return arr


def _draw_frame(
    rows: list[dict[str, float]],
    frame_index: int,
    x_lim: tuple[float, float],
    y_lim: tuple[float, float],
    title: str,
) -> plt.Figure:
    x = np.asarray([row["x_m"] for row in rows], dtype=float)
    y = np.asarray([row["y_m"] for row in rows], dtype=float)
    u = np.asarray([row["current_u_mps"] for row in rows], dtype=float)
    v = np.asarray([row["current_v_mps"] for row in rows], dtype=float)
    goal_x = float(rows[0]["goal_x_m"])
    goal_y = float(rows[0]["goal_y_m"])
    distance = float(rows[frame_index]["distance_to_goal_m"])

    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    ax.plot(x, y, linewidth=1.0, color="#f2c14e", alpha=0.35)
    ax.plot(x[: frame_index + 1], y[: frame_index + 1], linewidth=2.2, color="#c1121f")
    ax.scatter([x[0]], [y[0]], color="#2a9d8f", s=52, label="start")
    ax.scatter([x[frame_index]], [y[frame_index]], color="#1d3557", s=52, label="agent")
    ax.scatter([goal_x], [goal_y], color="#ffb703", marker="*", s=120, label="goal")

    stride = max(1, len(rows) // 20)
    ax.quiver(
        x[::stride],
        y[::stride],
        u[::stride] * 24.0,
        v[::stride] * 24.0,
        color="#3a86ff",
        alpha=0.4,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.003,
    )
    ax.quiver(
        [x[frame_index]],
        [y[frame_index]],
        [u[frame_index] * 30.0],
        [v[frame_index] * 30.0],
        color="#4361ee",
        alpha=0.9,
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.005,
    )

    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.24)
    ax.set_title(f"{title} | step {frame_index} | dist {distance:.1f} m")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def _case_title(case: dict[str, Any]) -> str:
    task = str(case.get("task", "task"))
    variant = str(case.get("variant", "variant"))
    mode = str(case.get("controller_mode", case.get("controller", "controller")))
    tide_mode = str(case.get("tide_mode", "on" if case.get("include_tides", True) else "off"))
    return f"S1 {task} | {variant} | {mode} | tide-{tide_mode}"


def _copy_if_exists(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not source.exists():
        return ""
    shutil.copy2(source, destination)
    return str(destination)


def main() -> None:
    args = parse_args()
    matrix_root = Path(args.matrix_root).expanduser().resolve()
    manifest_path = matrix_root / "matrix_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing matrix manifest: {manifest_path}")

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (matrix_root / "media").resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest_path.open("r", encoding="utf-8") as file:
        matrix_manifest = json.load(file)
    cases = list(matrix_manifest.get("cases", []))
    selected_cases = _select_cases(cases=cases, case_limit=args.case_limit, all_cases=args.all_cases)

    media_cases: list[dict[str, Any]] = []
    for case in selected_cases:
        case_id = int(case.get("case_id", -1))
        case_title = _case_title(case)
        case_slug = _slug(f"{case_id}_{case_title}")
        case_dir = output_dir / "cases" / case_slug
        case_dir.mkdir(parents=True, exist_ok=True)

        case_output_dir = Path(str(case["output_dir"]))
        trajectory_path = case_output_dir / "trajectories" / f"episode_{args.episode_index:03d}.csv"
        if not trajectory_path.exists():
            continue
        rows = _load_trajectory_csv(trajectory_path)
        x_lim, y_lim = _bounds(rows)
        frame_indices = _sample_indices(len(rows), args.max_frames)
        screenshot_indices = _sample_indices(len(rows), args.screenshot_count)

        static_index = frame_indices[-1]
        static_fig = _draw_frame(
            rows=rows,
            frame_index=static_index,
            x_lim=x_lim,
            y_lim=y_lim,
            title=case_title,
        )
        static_png_path = case_dir / "scene_overview.png"
        static_fig.savefig(static_png_path, dpi=160)
        plt.close(static_fig)

        frame_arrays: list[np.ndarray] = []
        screenshots: list[str] = []
        screenshot_set = set(screenshot_indices)
        for idx in frame_indices:
            fig = _draw_frame(rows=rows, frame_index=idx, x_lim=x_lim, y_lim=y_lim, title=case_title)
            frame_arrays.append(_figure_to_rgb_array(fig))
            if idx in screenshot_set:
                screenshot_path = case_dir / "screenshots" / f"step_{idx:04d}.png"
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(screenshot_path, dpi=150)
                screenshots.append(str(screenshot_path))
            plt.close(fig)

        gif_path = case_dir / "rollout.gif"
        mp4_path = case_dir / "rollout.mp4"
        imageio.mimsave(gif_path, frame_arrays, fps=max(1.0, args.fps))
        imageio.mimsave(mp4_path, frame_arrays, fps=max(1.0, args.fps))

        metrics_json = case_output_dir / "metrics.json"
        case_metrics = {}
        if metrics_json.exists():
            with metrics_json.open("r", encoding="utf-8") as file:
                case_metrics = json.load(file).get("summary", {})

        media_cases.append(
            {
                "case_id": case_id,
                "task": case.get("task"),
                "variant": case.get("variant"),
                "controller_mode": case.get("controller_mode"),
                "tide_mode": case.get("tide_mode", "on" if case.get("include_tides", True) else "off"),
                "trajectory_csv": str(trajectory_path),
                "scene_overview_png": str(static_png_path),
                "screenshots": screenshots,
                "rollout_gif": str(gif_path),
                "rollout_mp4": str(mp4_path),
                "summary_metrics": {
                    "success_rate": case_metrics.get("success_rate"),
                    "final_distance_to_goal_m_mean": case_metrics.get("final_distance_to_goal_m_mean"),
                    "energy_proxy_mean": case_metrics.get("energy_proxy_mean"),
                },
            }
        )

    if not media_cases:
        raise RuntimeError("No media cases rendered")

    hero_case = media_cases[0]
    hero_dir = output_dir / "hero"
    hero_scene = _copy_if_exists(Path(hero_case["scene_overview_png"]), hero_dir / "scene_overview.png")
    hero_gif = _copy_if_exists(Path(hero_case["rollout_gif"]), hero_dir / "rollout.gif")
    hero_mp4 = _copy_if_exists(Path(hero_case["rollout_mp4"]), hero_dir / "rollout.mp4")

    media_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "generated_by": "oneocean_sim.experiments.render_s1_media",
        "source_matrix_root": str(matrix_root),
        "source_matrix_manifest": str(manifest_path),
        "episode_index": args.episode_index,
        "fps": args.fps,
        "max_frames": args.max_frames,
        "case_count": len(media_cases),
        "media_dir": str(output_dir),
        "hero_assets": {
            "scene_overview_png": hero_scene,
            "rollout_gif": hero_gif,
            "rollout_mp4": hero_mp4,
        },
        "cases": media_cases,
    }

    media_manifest_path = output_dir / "media_manifest.json"
    with media_manifest_path.open("w", encoding="utf-8") as file:
        json.dump(media_manifest, file, indent=2)

    print(
        json.dumps(
            {
                "media_manifest": str(media_manifest_path),
                "media_dir": str(output_dir),
                "cases": len(media_cases),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
