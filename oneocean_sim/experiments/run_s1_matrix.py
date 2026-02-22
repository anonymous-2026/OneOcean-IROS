from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..runner import RunConfig, run_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a compact S1 experiment matrix")
    parser.add_argument("--output-root", type=str, default="runs/s1_matrix")
    parser.add_argument("--variants", type=str, default="tiny,scene")
    parser.add_argument("--tasks", type=str, default="navigation,station_keeping")
    parser.add_argument(
        "--controller-modes",
        type=str,
        default="compensated,naive",
        help="Comma-separated: compensated,naive",
    )
    parser.add_argument(
        "--tide-modes",
        type=str,
        default="on,off",
        help="Comma-separated: on,off",
    )
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--seed-base", type=int, default=100)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--depth-index", type=int, default=0)
    parser.add_argument("--disable-tides", action="store_true")
    return parser.parse_args()


def resolve_controller(task: str, mode: str) -> str:
    if mode == "compensated":
        return "goal_seek" if task == "navigation" else "station_keep"
    if mode == "naive":
        return "goal_seek_naive" if task == "navigation" else "station_keep_naive"
    raise ValueError(f"Unsupported controller mode: {mode}")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    tasks = [item.strip() for item in args.tasks.split(",") if item.strip()]
    controller_modes = [item.strip() for item in args.controller_modes.split(",") if item.strip()]
    tide_modes = [item.strip() for item in args.tide_modes.split(",") if item.strip()]
    if args.disable_tides:
        tide_modes = ["off"]

    case_rows: list[dict[str, object]] = []
    case_id = 0

    for variant in variants:
        for task in tasks:
            for controller_mode in controller_modes:
                controller = resolve_controller(task=task, mode=controller_mode)
                for tide_mode in tide_modes:
                    include_tides = tide_mode == "on"
                    seed = args.seed_base + case_id
                    out_dir = output_root / f"{case_id:02d}_{task}_{variant}_{controller_mode}_{tide_mode}"
                    config = RunConfig(
                        task=task,
                        controller=controller,
                        variant=variant,
                        episodes=args.episodes,
                        seed=seed,
                        time_index=args.time_index,
                        depth_index=args.depth_index,
                        max_steps=args.max_steps,
                        include_tides=include_tides,
                    )
                    outputs = run_task(config=config, output_dir=str(out_dir))
                    with Path(outputs["metrics_json"]).open("r", encoding="utf-8") as file:
                        metrics = json.load(file)
                    summary = metrics.get("summary", {})
                    case_rows.append(
                        {
                            "case_id": case_id,
                            "task": task,
                            "variant": variant,
                            "controller": controller,
                            "controller_mode": controller_mode,
                            "tide_mode": tide_mode,
                            "include_tides": include_tides,
                            "seed": seed,
                            "episodes": args.episodes,
                            "success_rate": summary.get("success_rate", None),
                            "final_distance_mean": summary.get("final_distance_to_goal_m_mean", None),
                            "energy_proxy_mean": summary.get("energy_proxy_mean", None),
                            "sim_steps_per_sec_mean": summary.get("sim_steps_per_sec_mean", None),
                            "output_dir": outputs["output_dir"],
                            "metrics_json": outputs["metrics_json"],
                        }
                    )
                    case_id += 1

    manifest_path = output_root / "matrix_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump({"cases": case_rows}, file, indent=2)

    markdown_path = output_root / "matrix_summary.md"
    with markdown_path.open("w", encoding="utf-8") as file:
        file.write("# S1 Matrix Summary\n\n")
        file.write(
            "| case_id | task | variant | controller_mode | tide_mode | episodes | success_rate | final_distance_mean | energy_proxy_mean | sim_steps_per_sec_mean | output_dir |\n"
        )
        file.write("|---:|---|---|---|---|---:|---:|---:|---:|---:|---|\n")
        for row in case_rows:
            success_rate = row["success_rate"]
            final_distance_mean = row["final_distance_mean"]
            energy_proxy_mean = row["energy_proxy_mean"]
            sim_steps_per_sec_mean = row["sim_steps_per_sec_mean"]
            success_cell = "n/a" if success_rate is None else f"{float(success_rate):.4f}"
            final_distance_cell = (
                "n/a" if final_distance_mean is None else f"{float(final_distance_mean):.4f}"
            )
            energy_cell = "n/a" if energy_proxy_mean is None else f"{float(energy_proxy_mean):.4f}"
            sim_rate_cell = (
                "n/a" if sim_steps_per_sec_mean is None else f"{float(sim_steps_per_sec_mean):.2f}"
            )
            file.write(
                f"| {row['case_id']} | {row['task']} | {row['variant']} | {row['controller_mode']} | "
                f"{row['tide_mode']} | {row['episodes']} | {success_cell} | "
                f"{final_distance_cell} | {energy_cell} | {sim_rate_cell} | {row['output_dir']} |\n"
            )

    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "summary_md": str(markdown_path),
                "cases": len(case_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
