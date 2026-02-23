from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consolidated S3 (A2) results manifest from a suite run")
    parser.add_argument("--suite-root", type=str, required=True, help="Path to runs/s3_3d_* suite folder")
    parser.add_argument("--output-json", type=str, default="project/results_manifest.json")
    parser.add_argument("--output-md", type=str, default="project/results_summary.md")
    parser.add_argument("--output-csv", type=str, default="project/results_summary.csv")
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root).expanduser().resolve()
    suite_manifest_path = suite_root / "suite_manifest.json"
    if not suite_manifest_path.exists():
        raise FileNotFoundError(f"Missing suite manifest: {suite_manifest_path}")

    with suite_manifest_path.open("r", encoding="utf-8") as file:
        suite = json.load(file)
    cases = suite.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite_manifest.json has no cases")

    rows: list[dict[str, object]] = []
    for case in cases:
        metrics_json_path = Path(str(case["metrics_json"])).expanduser().resolve()
        with metrics_json_path.open("r", encoding="utf-8") as file:
            metrics = json.load(file)
        summary = metrics.get("summary", {}) or {}

        run_config_path = Path(str(metrics.get("run_config_path", ""))).expanduser().resolve()
        dataset_path = None
        dataset_variant = None
        include_tides = None
        if run_config_path.exists():
            with run_config_path.open("r", encoding="utf-8") as file:
                run_cfg = json.load(file)
            dataset_ctx = run_cfg.get("dataset_context", {}) or {}
            dataset_path = dataset_ctx.get("dataset_path")
            dataset_variant = dataset_ctx.get("variant")
            include_tides = (run_cfg.get("run_config", {}) or {}).get("include_tides")

        rows.append(
            {
                "case_id": int(case["case_id"]),
                "task": str(case["task"]),
                "variant": str(case["variant"]),
                "tide_mode": str(case.get("tide_mode", "unknown")),
                "episodes": int(case.get("episodes", 0)),
                "seed": int(case.get("seed", 0)),
                "success_rate": _as_float(summary.get("success_rate")),
                "energy_proxy_mean": _as_float(summary.get("energy_proxy_mean")),
                "output_dir": str(case.get("output_dir", "")),
                "metrics_json": str(metrics_json_path),
                "media_manifest": case.get("media_manifest"),
                "run_config_json": str(run_config_path) if run_config_path.exists() else None,
                "dataset_path": dataset_path,
                "dataset_variant": dataset_variant,
                "include_tides": include_tides,
            }
        )

    by_key: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in rows:
        key = (str(row["task"]), str(row["variant"]), str(row["tide_mode"]))
        if key not in by_key:
            by_key[key] = {"count": 0.0, "success_sum": 0.0, "energy_sum": 0.0}
        agg = by_key[key]
        agg["count"] += 1.0
        agg["success_sum"] += float(row["success_rate"] or 0.0)
        agg["energy_sum"] += float(row["energy_proxy_mean"] or 0.0)

    aggregates = []
    for (task, variant, tide_mode), agg in sorted(by_key.items()):
        count = max(1.0, agg["count"])
        aggregates.append(
            {
                "task": task,
                "variant": variant,
                "tide_mode": tide_mode,
                "cases": int(agg["count"]),
                "success_rate_mean": agg["success_sum"] / count,
                "energy_proxy_mean": agg["energy_sum"] / count,
            }
        )

    official_dataset_cases = 0
    for row in rows:
        dataset_path = row.get("dataset_path")
        if isinstance(dataset_path, str) and dataset_path.endswith(".nc"):
            official_dataset_cases += 1

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md = Path(args.output_md).expanduser().resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_csv = Path(args.output_csv).expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "oneocean_sim_s3.experiments.build_s3_results_manifest",
        "suite_root": str(suite_root),
        "suite_manifest": str(suite_manifest_path),
        "case_count": int(len(rows)),
        "official_dataset_case_count": int(official_dataset_cases),
        "artifacts": {
            "results_manifest_json": str(output_json),
            "results_summary_md": str(output_md),
            "results_summary_csv": str(output_csv),
        },
        "cases": rows,
        "aggregates": aggregates,
        "key_numbers": {f"{a['task']}:{a['variant']}:{a['tide_mode']}": a for a in aggregates},
    }

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    with output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["task", "variant", "tide_mode", "cases", "success_rate_mean", "energy_proxy_mean"],
        )
        writer.writeheader()
        for row in aggregates:
            writer.writerow(row)

    with output_md.open("w", encoding="utf-8") as file:
        file.write("# S3 (A2) Results Summary\n\n")
        file.write(f"- Suite root: `{suite_root}`  \n")
        file.write(f"- Cases: `{len(rows)}`  \n")
        file.write(f"- Dataset-grounded cases: `{official_dataset_cases}`\n\n")
        file.write("## Aggregates\n\n")
        file.write("| task | variant | tide_mode | cases | success_rate_mean | energy_proxy_mean |\n")
        file.write("|---|---|---|---:|---:|---:|\n")
        for row in aggregates:
            file.write(
                f"| {row['task']} | {row['variant']} | {row['tide_mode']} | {row['cases']} | "
                f"{float(row['success_rate_mean']):.3f} | {float(row['energy_proxy_mean']):.3f} |\n"
            )

        file.write("\n## Case List\n\n")
        file.write("| case_id | task | variant | tide_mode | seed | success_rate | output_dir |\n")
        file.write("|---:|---|---|---|---:|---:|---|\n")
        for row in rows:
            sr = row.get("success_rate")
            sr_cell = "n/a" if sr is None else f"{float(sr):.3f}"
            file.write(
                f"| {row['case_id']} | {row['task']} | {row['variant']} | {row['tide_mode']} | "
                f"{row['seed']} | {sr_cell} | {row['output_dir']} |\n"
            )

    print(
        json.dumps(
            {
                "results_manifest_json": str(output_json),
                "results_summary_md": str(output_md),
                "results_summary_csv": str(output_csv),
                "case_count": len(rows),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
