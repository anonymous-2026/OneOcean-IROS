from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from ..data import open_dataset, resolve_dataset_path
from ..runner import RunConfig, run_task


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run S1 robustness suite (official dataset only)")
    parser.add_argument("--output-root", type=str, default="runs/s1_robustness_v1")
    parser.add_argument("--variant", type=str, default="scene", choices=["tiny", "scene", "public"])
    parser.add_argument("--dataset-path", type=str, default=None)
    parser.add_argument("--tasks", type=str, default="navigation,station_keeping")
    parser.add_argument("--controller-modes", type=str, default="compensated,naive")
    parser.add_argument("--tide-modes", type=str, default="on,off")
    parser.add_argument("--time-indices", type=str, default="0,-1")
    parser.add_argument("--depth-indices", type=str, default="0")
    parser.add_argument("--goal-distance-values", type=str, default="250,450")
    parser.add_argument("--max-speed-values", type=str, default="1.2,1.8")
    parser.add_argument("--episodes", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=700)
    parser.add_argument("--seed-base", type=int, default=900)
    parser.add_argument("--goal-tolerance-m", type=float, default=25.0)
    parser.add_argument("--station-success-radius-m", type=float, default=30.0)
    parser.add_argument("--station-mean-radius-m", type=float, default=40.0)
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_csv_tokens(csv_text: str) -> list[str]:
    return [token.strip() for token in csv_text.split(",") if token.strip()]


def _resolve_indices(spec: str, size: int) -> list[int]:
    resolved: list[int] = []
    for token in _parse_csv_tokens(spec):
        lower = token.lower()
        if lower in ("last",):
            index = size - 1
        else:
            index = int(token)
            if index < 0:
                index = size + index
        if not (0 <= index < size):
            raise IndexError(f"Index {index} out of range [0, {size - 1}] for spec={spec}")
        resolved.append(index)
    unique = []
    seen = set()
    for index in resolved:
        if index in seen:
            continue
        seen.add(index)
        unique.append(index)
    if not unique:
        raise ValueError(f"No valid indices from spec: {spec}")
    return unique


def _parse_float_values(spec: str) -> list[float]:
    values = [float(token) for token in _parse_csv_tokens(spec)]
    if not values:
        raise ValueError(f"Empty float list from spec: {spec}")
    return values


def _resolve_controller(task: str, mode: str) -> str:
    if mode == "compensated":
        return "goal_seek" if task == "navigation" else "station_keep"
    if mode == "naive":
        return "goal_seek_naive" if task == "navigation" else "station_keep_naive"
    raise ValueError(f"Unsupported controller mode: {mode}")


def _official_dataset(dataset_path: Path) -> bool:
    normalized = str(dataset_path)
    return ("OceanEnv/Data_pipeline/Data/Combined" in normalized) and normalized.endswith(".nc")


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_path = resolve_dataset_path(dataset_path=args.dataset_path, variant=args.variant)
    dataset = open_dataset(dataset_path)
    try:
        time_indices = _resolve_indices(args.time_indices, dataset.sizes["time"])
        depth_indices = _resolve_indices(args.depth_indices, dataset.sizes["depth"])
    finally:
        dataset.close()

    tasks = _parse_csv_tokens(args.tasks)
    controller_modes = _parse_csv_tokens(args.controller_modes)
    tide_modes = _parse_csv_tokens(args.tide_modes)
    goal_distance_values = _parse_float_values(args.goal_distance_values)
    max_speed_values = _parse_float_values(args.max_speed_values)

    case_rows: list[dict[str, object]] = []
    case_id = 0

    for task in tasks:
        for controller_mode in controller_modes:
            controller = _resolve_controller(task=task, mode=controller_mode)
            for tide_mode in tide_modes:
                include_tides = tide_mode == "on"
                for time_index in time_indices:
                    for depth_index in depth_indices:
                        for max_speed in max_speed_values:
                            goal_values = goal_distance_values if task == "navigation" else [0.0]
                            for goal_distance in goal_values:
                                seed = args.seed_base + case_id
                                speed_tag = f"{max_speed:.2f}".replace(".", "p")
                                goal_tag = f"{goal_distance:.0f}"
                                out_dir = output_root / (
                                    f"{case_id:03d}_{task}_{controller_mode}_{tide_mode}_"
                                    f"t{time_index}_d{depth_index}_g{goal_tag}_s{speed_tag}"
                                )
                                config = RunConfig(
                                    task=task,
                                    controller=controller,
                                    variant=args.variant,
                                    dataset_path=str(dataset_path),
                                    episodes=args.episodes,
                                    seed=seed,
                                    time_index=time_index,
                                    depth_index=depth_index,
                                    max_steps=args.max_steps,
                                    max_speed_mps=max_speed,
                                    goal_distance_m=goal_distance,
                                    goal_tolerance_m=args.goal_tolerance_m,
                                    station_success_radius_m=args.station_success_radius_m,
                                    station_mean_radius_m=args.station_mean_radius_m,
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
                                        "variant": args.variant,
                                        "dataset_path": str(dataset_path),
                                        "official_dataset": _official_dataset(dataset_path),
                                        "controller": controller,
                                        "controller_mode": controller_mode,
                                        "tide_mode": tide_mode,
                                        "include_tides": include_tides,
                                        "time_index": time_index,
                                        "depth_index": depth_index,
                                        "goal_distance_m": goal_distance,
                                        "max_speed_mps": max_speed,
                                        "episodes": args.episodes,
                                        "seed": seed,
                                        "success_rate": _as_float(summary.get("success_rate")),
                                        "final_distance_mean": _as_float(
                                            summary.get("final_distance_to_goal_m_mean")
                                        ),
                                        "energy_proxy_mean": _as_float(summary.get("energy_proxy_mean")),
                                        "sim_steps_per_sec_mean": _as_float(
                                            summary.get("sim_steps_per_sec_mean")
                                        ),
                                        "metrics_json": outputs["metrics_json"],
                                        "output_dir": outputs["output_dir"],
                                    }
                                )
                                case_id += 1

    manifest_path = output_root / "robustness_manifest.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(
            {
                "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
                "generated_by": "oneocean_sim.experiments.run_s1_robustness_suite",
                "dataset_path": str(dataset_path),
                "official_dataset": _official_dataset(dataset_path),
                "cases": case_rows,
            },
            file,
            indent=2,
        )

    aggregates: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {"count": 0.0, "success_sum": 0.0, "distance_sum": 0.0, "energy_sum": 0.0}
    )
    for row in case_rows:
        key = (str(row["task"]), str(row["controller_mode"]), str(row["tide_mode"]))
        aggregates[key]["count"] += 1.0
        aggregates[key]["success_sum"] += float(row["success_rate"] or 0.0)
        aggregates[key]["distance_sum"] += float(row["final_distance_mean"] or 0.0)
        aggregates[key]["energy_sum"] += float(row["energy_proxy_mean"] or 0.0)

    summary_path = output_root / "robustness_summary.md"
    with summary_path.open("w", encoding="utf-8") as file:
        file.write("# S1 Robustness Suite Summary\n\n")
        file.write(
            f"- Cases: `{len(case_rows)}`  \n- Dataset: `{dataset_path}`  \n"
            f"- Official generated dataset: `{_official_dataset(dataset_path)}`\n\n"
        )
        file.write("## Aggregated by Task/Controller/Tide\n\n")
        file.write(
            "| task | controller_mode | tide_mode | cases | success_rate_mean | final_distance_mean | energy_proxy_mean |\n"
        )
        file.write("|---|---|---|---:|---:|---:|---:|\n")
        for (task, controller_mode, tide_mode), agg in sorted(aggregates.items()):
            count = max(1.0, agg["count"])
            file.write(
                f"| {task} | {controller_mode} | {tide_mode} | {int(agg['count'])} | "
                f"{(agg['success_sum'] / count):.4f} | {(agg['distance_sum'] / count):.4f} | "
                f"{(agg['energy_sum'] / count):.4f} |\n"
            )
        file.write("\n## Case List\n\n")
        file.write(
            "| case_id | task | controller_mode | tide_mode | time_index | depth_index | goal_distance_m | max_speed_mps | episodes | success_rate | final_distance_mean | output_dir |\n"
        )
        file.write("|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for row in case_rows:
            success_cell = "n/a"
            if row["success_rate"] is not None:
                success_cell = f"{float(row['success_rate']):.4f}"
            final_distance_cell = "n/a"
            if row["final_distance_mean"] is not None:
                final_distance_cell = f"{float(row['final_distance_mean']):.4f}"
            file.write(
                f"| {row['case_id']} | {row['task']} | {row['controller_mode']} | {row['tide_mode']} | "
                f"{row['time_index']} | {row['depth_index']} | {float(row['goal_distance_m']):.1f} | "
                f"{float(row['max_speed_mps']):.2f} | {row['episodes']} | "
                f"{success_cell} | {final_distance_cell} | "
                f"{row['output_dir']} |\n"
            )

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "manifest": str(manifest_path),
                "summary_md": str(summary_path),
                "cases": len(case_rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
