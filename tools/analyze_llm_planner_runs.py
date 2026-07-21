#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _mean(values: list[Any]) -> float | None:
    finite = [number for value in values if (number := _finite_number(value)) is not None]
    return mean(finite) if finite else None


def _sum(values: list[Any]) -> float:
    return float(sum(number for value in values if (number := _finite_number(value)) is not None))


def _argument_value(arguments: list[Any], flag: str) -> str | None:
    string_arguments = [str(value) for value in arguments]
    try:
        return string_arguments[string_arguments.index(flag) + 1]
    except (ValueError, IndexError):
        return None


def _method_category(method: str) -> str:
    if method.startswith("llm_"):
        return "high_level_llm_planner"
    if method == "mlp_bc":
        return "low_level_behavior_cloning"
    return "low_level_heuristic"


def _read_records(root: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    records: list[dict[str, Any]] = []
    method_meta: dict[str, dict[str, Any]] = {}
    for method_directory in sorted(path for path in root.iterdir() if path.is_dir()):
        meta_path = method_directory / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            arguments = list(meta.get("argv", []))
            method_meta[method_directory.name] = {
                "planner_stride": _argument_value(arguments, "--llm-call-stride-steps"),
                "current_gain": _argument_value(arguments, "--current-gain"),
                "dynamics_model": _argument_value(arguments, "--dynamics-model"),
                "max_new_tokens": _argument_value(arguments, "--llm-max-new-tokens"),
            }
        for metrics_path in sorted(method_directory.rglob("metrics.json")):
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            final = dict(metrics.get("final", {}))
            cleanup_calls = int(final.get("llm_cleanup_calls", 0) or 0)
            waypoint_calls = int(final.get("llm_wp_calls", 0) or 0)
            cleanup_valid = int(final.get("llm_cleanup_valid", 0) or 0)
            waypoint_valid = int(final.get("llm_wp_valid", 0) or 0)
            completion_ratio = None
            task = str(metrics.get("task", ""))
            if task == "surface_pollution_cleanup_multiagent":
                done = _finite_number(final.get("sources_done"))
                total = _finite_number(final.get("sources_total"))
                completion_ratio = None if done is None or total is None or total <= 0.0 else done / total
            elif task == "area_scan_terrain_recon":
                completion_ratio = _finite_number(final.get("coverage"))
            elif task == "pipeline_inspection_leak_detection":
                done = _finite_number(final.get("leaks_detected"))
                total = _finite_number(final.get("leaks_total"))
                completion_ratio = None if done is None or total is None or total <= 0.0 else done / total
            records.append(
                {
                    "method": method_directory.name,
                    "method_category": _method_category(method_directory.name),
                    "task": task,
                    "difficulty": str(metrics.get("difficulty", "")),
                    "n_agents": int(metrics.get("n_agents", 0) or 0),
                    "success": bool(metrics.get("success", False)),
                    "time_to_success_s": metrics.get("time_to_success_s"),
                    "energy_proxy": metrics.get("energy_proxy"),
                    "constraint_violations": metrics.get("constraint_violations"),
                    "collision_rate": final.get("collision_rate"),
                    "planner_calls": cleanup_calls + waypoint_calls,
                    "valid_plans": cleanup_valid + waypoint_valid,
                    "uncached_calls": int(final.get("llm_uncached_calls", 0) or 0),
                    "latency_ms_total": final.get("llm_latency_ms_total"),
                    "prompt_tokens_total": int(final.get("llm_prompt_tokens_total", 0) or 0),
                    "output_tokens_total": int(final.get("llm_output_tokens_total", 0) or 0),
                    "task_completion_ratio": completion_ratio,
                }
            )
    return records, method_meta


def _aggregate(records: list[dict[str, Any]], method_meta: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["method"], record["task"], record["difficulty"], record["n_agents"])].append(record)
    rows: list[dict[str, Any]] = []
    for (method, task, difficulty, n_agents), group in sorted(groups.items()):
        calls = int(_sum([record["planner_calls"] for record in group]))
        valid = int(_sum([record["valid_plans"] for record in group]))
        uncached = int(_sum([record["uncached_calls"] for record in group]))
        prompt_tokens = int(_sum([record["prompt_tokens_total"] for record in group]))
        output_tokens = int(_sum([record["output_tokens_total"] for record in group]))
        llm_method = method.startswith("llm_")
        rows.append(
            {
                "method": method,
                "method_category": _method_category(method),
                "task": task,
                "difficulty": difficulty,
                "n_agents": n_agents,
                "episodes": len(group),
                "success_rate": mean([float(record["success"]) for record in group]),
                "time_to_success_mean_s": _mean([record["time_to_success_s"] for record in group]),
                "energy_proxy_mean": _mean([record["energy_proxy"] for record in group]),
                "constraint_violations_mean": _mean([record["constraint_violations"] for record in group]),
                "collision_rate_mean": _mean([record["collision_rate"] for record in group]),
                "task_completion_ratio_mean": _mean([record["task_completion_ratio"] for record in group]),
                "planner_stride_steps": method_meta.get(method, {}).get("planner_stride") if llm_method else None,
                "current_gain": method_meta.get(method, {}).get("current_gain"),
                "dynamics_model": method_meta.get(method, {}).get("dynamics_model"),
                "max_new_tokens": method_meta.get(method, {}).get("max_new_tokens") if llm_method else None,
                "planner_calls": calls if llm_method else None,
                "valid_plan_ratio": (valid / calls) if llm_method and calls else None,
                "fallback_ratio": (1.0 - valid / calls) if llm_method and calls else None,
                "uncached_calls": uncached if llm_method else None,
                "latency_ms_per_uncached_call": (_sum([record["latency_ms_total"] for record in group]) / uncached) if llm_method and uncached else None,
                "prompt_tokens_per_uncached_call": (prompt_tokens / uncached) if llm_method and uncached else None,
                "output_tokens_per_uncached_call": (output_tokens / uncached) if llm_method and uncached else None,
            }
        )
    return rows


def _format(value: Any, digits: int = 3) -> str:
    number = _finite_number(value)
    return "--" if number is None else f"{number:.{digits}f}"


def _write_csv(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(rows: list[dict[str, Any]], output: Path, source_run: str) -> None:
    llm_rows = [row for row in rows if str(row["method"]).startswith("llm_")]
    task_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in llm_rows:
        task_groups[str(row["task"])].append(row)
    strides = sorted({str(row["planner_stride_steps"]) for row in llm_rows if row["planner_stride_steps"] is not None})
    current_gains = sorted({str(row["current_gain"]) for row in rows if row["current_gain"] is not None})
    dynamics_models = sorted({str(row["dynamics_model"]) for row in rows if row["dynamics_model"] is not None})
    token_caps = sorted({str(row["max_new_tokens"]) for row in llm_rows if row["max_new_tokens"] is not None})
    lines = [
        "# LLM Planner Diagnostics",
        "",
        f"Source run: `{source_run}`. This is an aggregation of existing artifacts; no new model run was performed.",
        "",
        f"Recorded protocol: dynamics `{', '.join(dynamics_models)}`, current gain `{', '.join(current_gains)}`, planner stride `{', '.join(strides)}` steps, and output cap `{', '.join(token_caps)}` tokens. Each method/task group contains three medium-difficulty seeds.",
        "",
        "Heuristic and behavior-cloning methods generate low-level actions. The LLM methods only propose discrete source or waypoint assignments and share deterministic low-level goal following, clipping, dynamics, current, and constraints. Their rows are therefore planning diagnostics, not an architecture-matched end-to-end ranking.",
        "",
        "## Task-level results",
        "",
        "| Method | Task | Episodes | SR | Time (s) | Energy | Collision | Valid plan | Fallback | Latency/call (ms) | Tokens/call (in/out) | Completion |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        tokens = f"{_format(row['prompt_tokens_per_uncached_call'], 1)}/{_format(row['output_tokens_per_uncached_call'], 1)}"
        lines.append(
            f"| `{row['method']}` | `{row['task']}` | {row['episodes']} | {_format(row['success_rate'])} | {_format(row['time_to_success_mean_s'], 1)} | {_format(row['energy_proxy_mean'], 1)} | {_format(row['collision_rate_mean'])} | {_format(row['valid_plan_ratio'])} | {_format(row['fallback_ratio'])} | {_format(row['latency_ms_per_uncached_call'], 1)} | {tokens} | {_format(row['task_completion_ratio_mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Cross-model task diagnostics",
            "",
            "| Task | LLM rows | Mean SR | Mean valid ratio | Mean fallback ratio | Mean latency/call (ms) | Mean completion |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for task, group in sorted(task_groups.items()):
        lines.append(
            f"| `{task}` | {len(group)} | {_format(_mean([row['success_rate'] for row in group]))} | {_format(_mean([row['valid_plan_ratio'] for row in group]))} | {_format(_mean([row['fallback_ratio'] for row in group]))} | {_format(_mean([row['latency_ms_per_uncached_call'] for row in group]), 1)} | {_format(_mean([row['task_completion_ratio_mean'] for row in group]))} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `valid_plan_ratio` is the fraction of planner calls that produced a schema-valid assignment; `fallback_ratio` is its complement. A fallback uses the deterministic task allocator, so an LLM-labeled episode is not necessarily controlled by valid LLM plans at every planning event.",
            "- Cleanup tests discrete source assignment and dwell completion. Duplicate or invalid assignments can concentrate agents, which is visible in collision rate; this is not a continuous pollutant-mass experiment.",
            "- Area scan converts discrete waypoint assignments into a continuous coverage objective. A valid assignment can still repeatedly target already covered regions, so schema validity alone does not guarantee coverage efficiency.",
            "- Pipeline inspection requires spatial leak encounters, not only waypoint progress. A planner can produce valid low-frequency assignments while missing leaks between assigned waypoints; completion and waypoint error must be read together.",
            "- Energy is a trajectory-execution proxy, not model inference energy. Uncached latency and token counts are the relevant inference-cost measurements.",
            "- Existing records contain calls, valid/fallback outcomes, latency, and token counts for all three planning-sensitive tasks. The predeclared trigger for a new Qwen3-8B diagnostic is therefore not met, so expanding the model search would add cost without resolving a missing measurement.",
            "",
        ]
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate OneOcean LLM planner validity, fallback, cost, and task metrics.")
    parser.add_argument("run_root", help="Run directory containing one subdirectory per method.")
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("--md-out", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(args.run_root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Run root does not exist: {root}")
    records, method_meta = _read_records(root)
    if not records:
        raise SystemExit(f"No metrics.json files found under: {root}")
    rows = _aggregate(records, method_meta)
    _write_csv(rows, Path(args.csv_out).expanduser().resolve())
    _write_markdown(rows, Path(args.md_out).expanduser().resolve(), root.name)
    print(json.dumps({"source_run": root.name, "episodes": len(records), "groups": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
