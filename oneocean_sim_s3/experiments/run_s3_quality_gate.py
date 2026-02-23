from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ..runner import RunConfigS3, run_task_s3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S3 3D quality-gate suite (A2)")
    parser.add_argument("--output-root", type=str, default="runs/s3_3d_underwater_hero_v1")
    parser.add_argument("--variants", type=str, default="scene")
    parser.add_argument("--tasks", type=str, default="reef_navigation,formation_navigation")
    parser.add_argument("--tide-modes", type=str, default="on,off")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=360)
    parser.add_argument("--seed-base", type=int, default=800)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--depth-index", type=int, default=0)

    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--render-fps", type=int, default=12)
    parser.add_argument("--render-frame-stride", type=int, default=2)
    parser.add_argument("--camera-mode", type=str, default="follow", choices=["follow", "orbit"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    tide_modes = [m.strip() for m in args.tide_modes.split(",") if m.strip()]

    cases: list[dict[str, object]] = []
    case_id = 0

    for variant in variants:
        for tide_mode in tide_modes:
            include_tides = tide_mode == "on"
            for task in tasks:
                out_dir = output_root / f"{case_id:02d}_{task}_{variant}_tides_{tide_mode}"
                config = RunConfigS3(
                    task=task,
                    variant=variant,
                    episodes=int(args.episodes),
                    seed=int(args.seed_base + case_id * 10),
                    time_index=int(args.time_index),
                    depth_index=int(args.depth_index),
                    include_tides=bool(include_tides),
                    max_steps=int(args.max_steps),
                    render=not bool(args.no_render),
                    render_width=int(args.render_width),
                    render_height=int(args.render_height),
                    render_fps=int(args.render_fps),
                    render_frame_stride=int(args.render_frame_stride),
                    camera_mode=str(args.camera_mode),
                    render_episode_index=0,
                )

                outputs = run_task_s3(config=config, output_dir=str(out_dir))
                metrics_json = Path(outputs["metrics_json"])
                with metrics_json.open("r", encoding="utf-8") as file:
                    metrics = json.load(file)
                summary = metrics.get("summary", {})

                media_manifest = out_dir / "media" / "media_manifest.json"
                cases.append(
                    {
                        "case_id": int(case_id),
                        "task": str(task),
                        "variant": str(variant),
                        "tide_mode": str(tide_mode),
                        "episodes": int(args.episodes),
                        "seed": int(config.seed),
                        "success_rate": summary.get("success_rate"),
                        "energy_proxy_mean": summary.get("energy_proxy_mean"),
                        "output_dir": str(out_dir),
                        "metrics_json": str(metrics_json),
                        "media_manifest": str(media_manifest) if media_manifest.exists() else None,
                    }
                )
                case_id += 1

    manifest_path = output_root / "suite_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump({"cases": cases}, file, indent=2)

    summary_path = output_root / "suite_summary.md"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write("# S3 3D Quality-Gate Suite Summary (A2)\n\n")
        file.write("| case_id | task | variant | tide_mode | episodes | seed | success_rate | energy_proxy_mean | output_dir |\n")
        file.write("|---:|---|---|---|---:|---:|---:|---:|---|\n")
        for row in cases:
            sr = row.get("success_rate")
            ep = row.get("energy_proxy_mean")
            sr_cell = "n/a" if sr is None else f"{float(sr):.3f}"
            ep_cell = "n/a" if ep is None else f"{float(ep):.3f}"
            file.write(
                f"| {row['case_id']} | {row['task']} | {row['variant']} | {row['tide_mode']} | "
                f"{row['episodes']} | {row['seed']} | {sr_cell} | {ep_cell} | {row['output_dir']} |\n"
            )

    print(
        json.dumps(
            {
                "suite_manifest": str(manifest_path),
                "suite_summary_md": str(summary_path),
                "cases": len(cases),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
