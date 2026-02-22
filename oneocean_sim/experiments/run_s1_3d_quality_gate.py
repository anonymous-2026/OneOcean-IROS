from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from ..runner3d import Run3DConfig, run_task_3d


def _default_output_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("runs") / f"s1_3d_quality_v1_{stamp}"


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S1 3D quality-gate suite (MuJoCo primary)")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--variant", type=str, default="scene", choices=["tiny", "scene", "public"])
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--depth-index", type=int, default=0)
    parser.add_argument("--dt-sec", type=float, default=0.05)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--target-domain-size-m", type=float, default=900.0)
    parser.add_argument("--current-scale", type=float, default=80.0)
    parser.add_argument("--render-width", type=int, default=960)
    parser.add_argument("--render-height", type=int, default=544)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--camera", type=str, default="cam_main")
    parser.add_argument("--record-all-episodes", action="store_true")
    parser.add_argument("--disable-tides", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.output_root) if args.output_root else _default_output_root()
    root.mkdir(parents=True, exist_ok=True)

    base = Run3DConfig(
        variant=args.variant,
        episodes=int(args.episodes),
        seed=int(args.seed),
        time_index=int(args.time_index),
        depth_index=int(args.depth_index),
        include_tides=not bool(args.disable_tides),
        dt_sec=float(args.dt_sec),
        max_steps=int(args.max_steps),
        target_domain_size_m=float(args.target_domain_size_m),
        current_scale=float(args.current_scale),
        record_media=True,
        record_all_episodes=bool(args.record_all_episodes),
        render_width=int(args.render_width),
        render_height=int(args.render_height),
        fps=int(args.fps),
        camera=str(args.camera),
    )

    suite: list[dict[str, Any]] = []
    runs = [
        ("nav_obstacles_3d", "compensated"),
        ("nav_obstacles_3d", "naive"),
        ("plume_source_localization_3d", "compensated"),
        ("plume_source_localization_3d", "naive"),
    ]

    for idx, (task, controller) in enumerate(runs):
        cfg = Run3DConfig(**asdict(base))
        cfg.task = task
        cfg.controller = controller
        cfg.seed = int(base.seed + 1000 * idx)

        out_dir = root / f"{task}__{controller}"
        cmd = (
            f"MUJOCO_GL=egl /data/private/user2/workspace/robosuite_learning/.venv/bin/python "
            f"-m oneocean_sim.cli.run_3d_task --task {task} --controller {controller} "
            f"--variant {cfg.variant} --episodes {cfg.episodes} --seed {cfg.seed} "
            f"--time-index {cfg.time_index} --depth-index {cfg.depth_index} "
            f"--max-steps {cfg.max_steps} --dt-sec {cfg.dt_sec} "
            f"--target-domain-size-m {cfg.target_domain_size_m} "
            f"--current-scale {cfg.current_scale} "
            f"--render-width {cfg.render_width} --render-height {cfg.render_height} "
            f"--fps {cfg.fps} --camera {cfg.camera} --output-dir {out_dir}"
        )
        outputs = run_task_3d(config=cfg, output_dir=str(out_dir), command=cmd)
        suite.append(
            {
                "task": task,
                "controller": controller,
                "outputs": outputs,
                "command": cmd,
                "config": asdict(cfg),
            }
        )

    manifest_path = root / "suite_manifest.json"
    manifest_path.write_text(json.dumps({"suite": suite}, indent=2), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    for item in suite:
        metrics = _read_json(Path(item["outputs"]["metrics_json"]))
        s = metrics.get("summary", {})
        summary_rows.append(
            {
                "task": item["task"],
                "controller": item["controller"],
                "success_rate": s.get("success_rate", "n/a"),
                "time_sec_mean": s.get("time_sec_mean", "n/a"),
                "energy_proxy_mean": s.get("energy_proxy_mean", "n/a"),
                "output_dir": Path(item["outputs"]["output_dir"]).as_posix(),
            }
        )

    md = [
        "# S1 3D Quality-Gate Suite (v1)",
        "",
        f"- Output root: `{root}`",
        f"- Variant: `{args.variant}`",
        f"- Episodes per run: `{args.episodes}`",
        "",
        _md_table(summary_rows, ["task", "controller", "success_rate", "time_sec_mean", "energy_proxy_mean", "output_dir"]),
        "",
        "See `suite_manifest.json` for the exact commands/configs and per-run `media_manifest.json`.",
    ]
    _write_text(root / "suite_summary.md", "\n".join(md) + "\n")


if __name__ == "__main__":
    main()
