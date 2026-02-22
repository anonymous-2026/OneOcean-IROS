from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build paper/web-ready artifacts from an S1 matrix run")
    parser.add_argument("--matrix-root", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt4(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4f}"


def _write_markdown_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("# S1 Matrix Aggregated Summary\n\n")
        file.write(
            "| case_id | variant | task | controller_mode | tide_mode | success_rate | final_distance_mean | energy_proxy_mean | sim_steps_per_sec_mean |\n"
        )
        file.write("|---:|---|---|---|---|---:|---:|---:|---:|\n")
        for row in rows:
            file.write(
                f"| {row['case_id']} | {row['variant']} | {row['task']} | "
                f"{row['controller_mode']} | {row['tide_mode']} | "
                f"{_fmt4(row['success_rate'])} | {_fmt4(row['final_distance_mean'])} | "
                f"{_fmt4(row['energy_proxy_mean'])} | {_fmt4(row['sim_steps_per_sec_mean'])} |\n"
            )


def _write_ablation_table(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        file.write("# S1 Controller Ablation (Compensated vs Naive)\n\n")
        file.write(
            "| variant | task | tide_mode | success_comp | success_naive | delta_success | final_dist_comp | final_dist_naive | delta_final_distance_reduction |\n"
        )
        file.write("|---|---|---|---:|---:|---:|---:|---:|---:|\n")
        for row in rows:
            file.write(
                f"| {row['variant']} | {row['task']} | {row['tide_mode']} | "
                f"{_fmt4(row['success_comp'])} | {_fmt4(row['success_naive'])} | "
                f"{_fmt4(row['delta_success'])} | {_fmt4(row['final_dist_comp'])} | "
                f"{_fmt4(row['final_dist_naive'])} | {_fmt4(row['delta_final_distance_reduction'])} |\n"
            )


def _plot_ablation(
    rows: list[dict[str, Any]],
    output_png: Path,
    output_pdf: Path,
    metric_comp_key: str,
    metric_naive_key: str,
    ylabel: str,
    title: str,
) -> None:
    if not rows:
        return

    labels = [f"{row['variant']}/{row['task']}/{row['tide_mode']}" for row in rows]
    comp_values = [_as_float(row[metric_comp_key]) or 0.0 for row in rows]
    naive_values = [_as_float(row[metric_naive_key]) or 0.0 for row in rows]

    x = np.arange(len(labels), dtype=np.float64)
    width = 0.38

    plt.figure(figsize=(max(10, len(labels) * 0.9), 4.8))
    plt.bar(x - width / 2.0, comp_values, width=width, label="compensated")
    plt.bar(x + width / 2.0, naive_values, width=width, label="naive")
    plt.xticks(x, labels, rotation=25, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_png, dpi=160)
    plt.savefig(output_pdf)
    plt.close()


def build_report(matrix_root: Path, output_dir: Path) -> dict[str, Any]:
    manifest_path = matrix_root / "matrix_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing matrix manifest: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as file:
        matrix_manifest = json.load(file)
    cases = matrix_manifest.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("matrix_manifest.json has no cases")

    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)

    for case in cases:
        metrics_json_path = Path(case["metrics_json"])
        with metrics_json_path.open("r", encoding="utf-8") as file:
            metrics_data = json.load(file)
        summary = metrics_data.get("summary", {})

        row = {
            "case_id": int(case["case_id"]),
            "variant": str(case["variant"]),
            "task": str(case["task"]),
            "controller": str(case.get("controller", "")),
            "controller_mode": str(case.get("controller_mode", "")),
            "tide_mode": str(case.get("tide_mode", "on" if case.get("include_tides", True) else "off")),
            "episodes": int(case.get("episodes", 0)),
            "success_rate": _as_float(summary.get("success_rate")),
            "final_distance_mean": _as_float(summary.get("final_distance_to_goal_m_mean")),
            "final_distance_std": _as_float(summary.get("final_distance_to_goal_m_std")),
            "energy_proxy_mean": _as_float(summary.get("energy_proxy_mean")),
            "sim_steps_per_sec_mean": _as_float(summary.get("sim_steps_per_sec_mean")),
            "output_dir": str(case.get("output_dir", "")),
            "metrics_json": str(metrics_json_path),
        }
        summary_rows.append(row)
        grouped[(row["variant"], row["task"], row["tide_mode"])][row["controller_mode"]] = row

    summary_rows.sort(
        key=lambda row: (
            row["variant"],
            row["task"],
            row["tide_mode"],
            row["controller_mode"],
            row["case_id"],
        )
    )

    summary_json = output_dir / "summary_rows.json"
    with summary_json.open("w", encoding="utf-8") as file:
        json.dump({"rows": summary_rows}, file, indent=2)

    summary_md = output_dir / "summary_table.md"
    _write_markdown_summary(summary_md, summary_rows)

    ablation_rows: list[dict[str, Any]] = []
    for (variant, task, tide_mode), mode_rows in grouped.items():
        compensated = mode_rows.get("compensated")
        naive = mode_rows.get("naive")
        if not compensated or not naive:
            continue
        success_comp = compensated["success_rate"]
        success_naive = naive["success_rate"]
        final_dist_comp = compensated["final_distance_mean"]
        final_dist_naive = naive["final_distance_mean"]
        ablation_rows.append(
            {
                "variant": variant,
                "task": task,
                "tide_mode": tide_mode,
                "success_comp": success_comp,
                "success_naive": success_naive,
                "delta_success": (
                    None
                    if success_comp is None or success_naive is None
                    else float(success_comp - success_naive)
                ),
                "final_dist_comp": final_dist_comp,
                "final_dist_naive": final_dist_naive,
                "delta_final_distance_reduction": (
                    None
                    if final_dist_comp is None or final_dist_naive is None
                    else float(final_dist_naive - final_dist_comp)
                ),
            }
        )
    ablation_rows.sort(key=lambda row: (row["variant"], row["task"], row["tide_mode"]))

    ablation_json = output_dir / "ablation_rows.json"
    with ablation_json.open("w", encoding="utf-8") as file:
        json.dump({"rows": ablation_rows}, file, indent=2)

    ablation_md = output_dir / "ablation_summary.md"
    _write_ablation_table(ablation_md, ablation_rows)

    success_png = output_dir / "fig_success_rate_ablation.png"
    success_pdf = output_dir / "fig_success_rate_ablation.pdf"
    _plot_ablation(
        rows=ablation_rows,
        output_png=success_png,
        output_pdf=success_pdf,
        metric_comp_key="success_comp",
        metric_naive_key="success_naive",
        ylabel="Success rate",
        title="S1 controller ablation: success rate",
    )

    distance_png = output_dir / "fig_final_distance_ablation.png"
    distance_pdf = output_dir / "fig_final_distance_ablation.pdf"
    _plot_ablation(
        rows=ablation_rows,
        output_png=distance_png,
        output_pdf=distance_pdf,
        metric_comp_key="final_dist_comp",
        metric_naive_key="final_dist_naive",
        ylabel="Final distance to goal (m)",
        title="S1 controller ablation: final distance",
    )

    results_manifest = {
        "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "source_matrix_root": str(matrix_root.resolve()),
        "source_manifest": str(manifest_path.resolve()),
        "artifact_dir": str(output_dir.resolve()),
        "artifacts": {
            "summary_rows_json": str(summary_json),
            "summary_table_md": str(summary_md),
            "ablation_rows_json": str(ablation_json),
            "ablation_summary_md": str(ablation_md),
            "fig_success_rate_png": str(success_png),
            "fig_success_rate_pdf": str(success_pdf),
            "fig_final_distance_png": str(distance_png),
            "fig_final_distance_pdf": str(distance_pdf),
        },
        "handoff_mapping": {
            "lane_b_paper": {
                "recommended_table": str(summary_md),
                "recommended_ablation_table": str(ablation_md),
                "recommended_figures": [str(success_pdf), str(distance_pdf)],
                "note": "Use compensated-vs-naive and tide on/off rows to support robotics-control claims.",
            },
            "lane_e_websites": {
                "cards_source": str(summary_json),
                "plot_pngs": [str(success_png), str(distance_png)],
                "note": "PNG files are optimized for website embedding.",
            },
        },
    }
    manifest_out = output_dir / "results_manifest.json"
    with manifest_out.open("w", encoding="utf-8") as file:
        json.dump(results_manifest, file, indent=2)

    return {
        "manifest": str(manifest_out),
        "summary_table": str(summary_md),
        "ablation_table": str(ablation_md),
        "success_plot": str(success_png),
        "distance_plot": str(distance_png),
        "cases": len(summary_rows),
    }


def main() -> None:
    args = parse_args()
    matrix_root = Path(args.matrix_root).expanduser().resolve()
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else (matrix_root / "report").resolve()
    )
    outputs = build_report(matrix_root=matrix_root, output_dir=output_dir)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
