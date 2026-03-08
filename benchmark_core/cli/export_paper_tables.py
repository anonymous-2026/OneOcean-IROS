#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _to_bool(s: object) -> bool:
    return str(s).strip().lower() in {"1", "true", "yes", "y", "t"}


def _to_float(s: object) -> float | None:
    if s is None:
        return None
    st = str(s).strip()
    if not st:
        return None
    try:
        v = float(st)
    except Exception:
        return None
    if v != v:  # NaN
        return None
    return float(v)


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(sum(xs) / float(len(xs)))


@dataclass(frozen=True)
class RunSpec:
    label: str
    root: Path


def _parse_runs(specs: list[str]) -> list[RunSpec]:
    out: list[RunSpec] = []
    for s in specs:
        if "=" in s:
            label, p = s.split("=", 1)
        elif ":" in s:
            label, p = s.split(":", 1)
        else:
            raise SystemExit(f"invalid --run {s!r}: expected label=path")
        out.append(RunSpec(label=str(label).strip(), root=Path(p).expanduser().resolve()))
    return out


def _load_rows(summary_csv: Path) -> list[dict[str, str]]:
    with summary_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _find_summary_csv(run_root: Path) -> Path:
    p = run_root / "summary.csv"
    if p.exists():
        return p
    raise FileNotFoundError(f"summary.csv not found under {run_root}")


def _group_rows(rows: list[dict[str, str]], *, difficulty: str | None = None) -> dict[tuple[str, str], list[dict[str, str]]]:
    g: dict[tuple[str, str], list[dict[str, str]]] = {}
    for r in rows:
        diff = str(r.get("difficulty", "")).strip()
        if difficulty and diff != str(difficulty):
            continue
        task = str(r.get("task", "")).strip()
        key = (task, diff)
        g.setdefault(key, []).append(r)
    return g


def _agg_task(rows: list[dict[str, str]]) -> dict[str, Any]:
    eps = int(len(rows))
    succ = [1.0 if _to_bool(r.get("success", "")) else 0.0 for r in rows]
    sr = _mean(succ) or 0.0

    tsucc = [_to_float(r.get("time_to_success_s", "")) for r in rows if _to_bool(r.get("success", ""))]
    tsucc_f = [x for x in tsucc if x is not None]

    return {
        "eps": eps,
        "SR": float(sr),
        "Tsucc_s": _mean(tsucc_f),
        "E_mean": _mean([x for x in (_to_float(r.get("energy_proxy", "")) for r in rows) if x is not None]),
        "Viol_mean": _mean([x for x in (_to_float(r.get("constraint_violations", "")) for r in rows) if x is not None]),
        "station_rms_error_m": _mean([x for x in (_to_float(r.get("station_rms_error_m", "")) for r in rows) if x is not None]),
        "waypoint_track_mean_error_m": _mean([x for x in (_to_float(r.get("waypoint_track_mean_error_m", "")) for r in rows) if x is not None]),
        "coverage_mean": _mean([x for x in (_to_float(r.get("coverage", "")) for r in rows) if x is not None]),
        "pipeline_score_mean": _mean([x for x in (_to_float(r.get("pipeline_score", "")) for r in rows) if x is not None]),
        "cleanup_rate_mean": _mean([x for x in (_to_float(r.get("cleanup_rate", "")) for r in rows) if x is not None]),
        "collision_rate_mean": _mean([x for x in (_to_float(r.get("collision_rate", "")) for r in rows) if x is not None]),
    }


def _write_md_table(path: Path, header: list[str], rows: list[list[object]]) -> None:
    lines: list[str] = []
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join("" if v is None else str(v) for v in r) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _fmt_pct(x: float | None) -> str:
    if x is None:
        return ""
    return f"{100.0*float(x):.1f}%"


def _fmt_f(x: float | None, nd: int = 2) -> str:
    if x is None:
        return ""
    return f"{float(x):.{int(nd)}f}"


def export_main(*, out_dir: Path, runs: list[RunSpec], difficulty: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # A compact method-level table aligned with paper/shuaijun-iros26-paper/docs/suggestion.md.
    header = [
        "method",
        "go_to_goal SR",
        "station RMS err (m)",
        "route wp err (m)",
        "scan coverage",
        "pipeline score",
        "cleanup rate",
        "energy",
    ]
    table_rows: list[list[object]] = []
    per_task_rows: list[list[object]] = []
    per_task_header = ["method", "task", "eps", "SR", "Tsucc_s", "E_mean", "Viol_mean", "collision_rate"]

    for rs in runs:
        rows = _load_rows(_find_summary_csv(rs.root))
        g = _group_rows(rows, difficulty=difficulty)

        def agg(task: str) -> dict[str, Any]:
            rr = g.get((task, difficulty), [])
            return _agg_task(rr) if rr else {"eps": 0, "SR": None}

        go = agg("go_to_goal_current")
        st = agg("station_keeping")
        rt = agg("route_following_waypoints")
        sc = agg("area_scan_terrain_recon")
        pl = agg("pipeline_inspection_leak_detection")
        cl = agg("surface_pollution_cleanup_multiagent")

        energy_all = _mean([x for x in (_to_float(r.get("energy_proxy", "")) for r in rows) if x is not None])

        table_rows.append(
            [
                rs.label,
                _fmt_pct(go.get("SR")),
                _fmt_f(st.get("station_rms_error_m"), 2),
                _fmt_f(rt.get("waypoint_track_mean_error_m"), 2),
                _fmt_f(sc.get("coverage_mean"), 2),
                _fmt_f(pl.get("pipeline_score_mean"), 2),
                _fmt_f(cl.get("cleanup_rate_mean"), 2),
                _fmt_f(energy_all, 1),
            ]
        )

        # Per-task breakdown (same difficulty).
        for (task, diff), rr in sorted(g.items()):
            if diff != difficulty:
                continue
            a = _agg_task(rr)
            per_task_rows.append(
                [
                    rs.label,
                    task,
                    int(a["eps"]),
                    _fmt_pct(a["SR"]),
                    _fmt_f(a["Tsucc_s"], 1),
                    _fmt_f(a["E_mean"], 1),
                    _fmt_f(a["Viol_mean"], 2),
                    _fmt_f(a["collision_rate_mean"], 3),
                ]
            )

    _write_md_table(out_dir / f"table_main_{difficulty}.md", header, table_rows)
    _write_md_table(out_dir / f"table_per_task_{difficulty}.md", per_task_header, per_task_rows)


def export_currents(*, out_dir: Path, runs: list[RunSpec], difficulty: str, tasks: list[str]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Expect labels like "heuristic_cg0", "heuristic_cg1", ... for grouping.
    by_method: dict[str, dict[str, RunSpec]] = {}
    for rs in runs:
        lab = rs.label
        cg = ""
        method = lab
        if "_cg" in lab:
            method, cg = lab.split("_cg", 1)
            cg = "cg" + cg
        by_method.setdefault(method, {})[cg or "cg?"] = rs

    cgs = sorted({cg for m in by_method.values() for cg in m.keys()})
    header = ["method", *cgs]
    rows_out: list[list[object]] = []
    for method, m in sorted(by_method.items()):
        r = [method]
        for cg in cgs:
            rs = m.get(cg)
            if rs is None:
                r.append("")
                continue
            rows = _load_rows(_find_summary_csv(rs.root))
            g = _group_rows(rows, difficulty=difficulty)
            srs: list[float] = []
            for t in tasks:
                rr = g.get((t, difficulty), [])
                if not rr:
                    continue
                a = _agg_task(rr)
                if a.get("SR") is not None:
                    srs.append(float(a["SR"]))
            r.append(_fmt_pct(_mean(srs)))
        rows_out.append(r)
    _write_md_table(out_dir / f"table_currentsweep_{difficulty}.md", header, rows_out)


def export_disturbances(*, out_dir: Path, runs: list[RunSpec], difficulty: str, tasks: list[str]) -> None:
    """Table 2-style robustness: success rate under no/mild/strong/tidal disturbances."""
    out_dir.mkdir(parents=True, exist_ok=True)

    def _cond_key(raw: str) -> str | None:
        s = str(raw or "").strip().lower()
        if not s:
            return None
        s = s.replace("-", "_").replace(" ", "_")
        if s in {"no", "none", "nocurrent", "no_current", "cg0", "current0"}:
            return "No Current"
        if s in {"mild", "mildcurrent", "mild_current", "cg1", "current1"}:
            return "Mild Current"
        if s in {"strong", "strongcurrent", "strong_current", "cg2", "current2"}:
            return "Strong Current"
        if s in {"tide", "tidal", "tidal_disturbance", "tidal_disturb"}:
            return "Tidal Disturbance"
        return None

    by_method: dict[str, dict[str, RunSpec]] = {}
    for rs in runs:
        lab = str(rs.label)
        method = lab
        cond_raw = ""
        if "__" in lab:
            method, cond_raw = lab.split("__", 1)
        elif "_cg" in lab:
            method, cg = lab.split("_cg", 1)
            cond_raw = "cg" + cg
        elif ":" in lab:
            method, cond_raw = lab.split(":", 1)
        cond = _cond_key(cond_raw)
        if cond is None:
            raise SystemExit(f"disturbances: could not infer condition from label {lab!r}; use method__no_current etc.")
        by_method.setdefault(str(method), {})[cond] = rs

    conds = ["No Current", "Mild Current", "Strong Current", "Tidal Disturbance"]
    header = ["method", *conds]
    rows_out: list[list[object]] = []
    for method, m in sorted(by_method.items()):
        r: list[object] = [method]
        for cond in conds:
            rs = m.get(cond)
            if rs is None:
                r.append("")
                continue
            rows = _load_rows(_find_summary_csv(rs.root))
            g = _group_rows(rows, difficulty=difficulty)
            srs: list[float] = []
            for t in tasks:
                rr = g.get((t, difficulty), [])
                if not rr:
                    continue
                a = _agg_task(rr)
                if a.get("SR") is not None:
                    srs.append(float(a["SR"]))
            r.append(_fmt_pct(_mean(srs)))
        rows_out.append(r)
    _write_md_table(out_dir / f"table_disturbances_{difficulty}.md", header, rows_out)


def export_scaling(*, out_dir: Path, runs: list[RunSpec], task: str, difficulty: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Expect labels like "N02", "N04", ...
    rows_out: list[list[object]] = []
    header = ["N", "eps", "SR", "cleanup_rate", "collision_rate", "Tsucc_s (succ only)"]
    for rs in runs:
        # pull N from label, fallback to reading n_agents from first row
        rows = _load_rows(_find_summary_csv(rs.root))
        g = _group_rows(rows, difficulty=difficulty)
        rr = g.get((task, difficulty), [])
        a = _agg_task(rr) if rr else {"eps": 0, "SR": None}
        n = rs.label
        if rows:
            try:
                n = str(int(float(rows[0].get("n_agents", "0") or "0")))
            except Exception:
                n = rs.label
        rows_out.append(
            [
                n,
                int(a.get("eps", 0) or 0),
                _fmt_pct(a.get("SR")),
                _fmt_f(a.get("cleanup_rate_mean"), 2),
                _fmt_f(a.get("collision_rate_mean"), 3),
                _fmt_f(a.get("Tsucc_s"), 1),
            ]
        )
    # sort by N if possible
    def _key(r: list[object]) -> int:
        try:
            return int(str(r[0]))
        except Exception:
            return 10**9

    rows_out.sort(key=_key)
    _write_md_table(out_dir / f"table_scaling_{task}_{difficulty}.md", header, rows_out)


def export_planning_suite(*, out_dir: Path, runs: list[RunSpec], difficulty: str) -> None:
    """LLM-planner-focused table for the planning-sensitive tasks (cleanup/scan/pipeline)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    header = [
        "method",
        "cleanup SR",
        "cleanup rate",
        "scan SR",
        "scan coverage",
        "pipeline SR",
        "pipeline score",
        "collision_rate (cleanup)",
    ]
    rows_out: list[list[object]] = []

    for rs in runs:
        rows = _load_rows(_find_summary_csv(rs.root))
        g = _group_rows(rows, difficulty=difficulty)

        def agg(task: str) -> dict[str, Any]:
            rr = g.get((task, difficulty), [])
            return _agg_task(rr) if rr else {"eps": 0, "SR": None}

        cl = agg("surface_pollution_cleanup_multiagent")
        sc = agg("area_scan_terrain_recon")
        pl = agg("pipeline_inspection_leak_detection")

        rows_out.append(
            [
                rs.label,
                _fmt_pct(cl.get("SR")),
                _fmt_f(cl.get("cleanup_rate_mean"), 2),
                _fmt_pct(sc.get("SR")),
                _fmt_f(sc.get("coverage_mean"), 2),
                _fmt_pct(pl.get("SR")),
                _fmt_f(pl.get("pipeline_score_mean"), 2),
                _fmt_f(cl.get("collision_rate_mean"), 3),
            ]
        )

    _write_md_table(out_dir / f"table_planning_suite_{difficulty}.md", header, rows_out)


def export_planning_suite_cost(*, out_dir: Path, runs: list[RunSpec], difficulty: str) -> None:
    """LLM comparison with efficiency metrics (latency/tokens) for planning-suite tasks."""
    out_dir.mkdir(parents=True, exist_ok=True)
    header = [
        "method",
        "cleanup SR",
        "scan SR",
        "pipeline SR",
        "llm uncached calls",
        "llm latency (ms/call)",
        "llm prompt toks/call",
        "llm output toks/call",
    ]
    rows_out: list[list[object]] = []
    tasks = ["surface_pollution_cleanup_multiagent", "area_scan_terrain_recon", "pipeline_inspection_leak_detection"]

    for rs in runs:
        rows = _load_rows(_find_summary_csv(rs.root))
        g = _group_rows(rows, difficulty=difficulty)

        def agg(task: str) -> dict[str, Any]:
            rr = g.get((task, difficulty), [])
            return _agg_task(rr) if rr else {"eps": 0, "SR": None}

        cl = agg("surface_pollution_cleanup_multiagent")
        sc = agg("area_scan_terrain_recon")
        pl = agg("pipeline_inspection_leak_detection")

        # Aggregate LLM efficiency over all planning-suite episodes (ignore cached calls).
        uncached_calls = 0.0
        latency_ms_total = 0.0
        prompt_toks_total = 0.0
        out_toks_total = 0.0
        for t in tasks:
            rr = g.get((t, difficulty), [])
            for r in rr:
                try:
                    uncached_calls += float(r.get("llm_uncached_calls", "") or 0.0)
                except Exception:
                    pass
                try:
                    latency_ms_total += float(r.get("llm_latency_ms_total", "") or 0.0)
                except Exception:
                    pass
                try:
                    prompt_toks_total += float(r.get("llm_prompt_tokens_total", "") or 0.0)
                except Exception:
                    pass
                try:
                    out_toks_total += float(r.get("llm_output_tokens_total", "") or 0.0)
                except Exception:
                    pass
        lat_per = (latency_ms_total / uncached_calls) if uncached_calls > 0 else None
        pt_per = (prompt_toks_total / uncached_calls) if uncached_calls > 0 else None
        ot_per = (out_toks_total / uncached_calls) if uncached_calls > 0 else None

        rows_out.append(
            [
                rs.label,
                _fmt_pct(cl.get("SR")),
                _fmt_pct(sc.get("SR")),
                _fmt_pct(pl.get("SR")),
                _fmt_f(uncached_calls, 1) if uncached_calls > 0 else "",
                _fmt_f(lat_per, 1),
                _fmt_f(pt_per, 1),
                _fmt_f(ot_per, 1),
            ]
        )

    _write_md_table(out_dir / f"table_planning_suite_cost_{difficulty}.md", header, rows_out)


def export_difficulty_ladder(*, out_dir: Path, runs: list[RunSpec], tasks: list[str]) -> None:
    """Table 4-style ladder: per-task success over easy/medium/hard for a task subset."""
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ["method", "task", "difficulty", "n_agents", "eps", "SR"]
    rows_out: list[list[object]] = []
    diffs = ["easy", "medium", "hard"]

    for rs in runs:
        rows = _load_rows(_find_summary_csv(rs.root))
        for t in tasks:
            for d in diffs:
                rr = [r for r in rows if str(r.get("task", "")).strip() == str(t) and str(r.get("difficulty", "")).strip() == str(d)]
                if not rr:
                    continue
                a = _agg_task(rr)
                # n_agents is constant within rr; take the first.
                try:
                    n = str(int(float(rr[0].get("n_agents", "0") or "0")))
                except Exception:
                    n = ""
                rows_out.append([rs.label, t, d, n, int(a.get("eps", 0) or 0), _fmt_pct(a.get("SR"))])

    _write_md_table(out_dir / "table_difficulty_ladder.md", header, rows_out)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Export paper-ready Markdown tables from headless run_matrix summary.csv.")
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--run", action="append", default=[], help="Run spec label=path (repeatable).")
    ap.add_argument("--table", type=str, required=True, choices=["main", "currents", "disturbances", "scaling", "planning_suite", "planning_suite_cost", "difficulty_ladder"])
    ap.add_argument("--difficulty", type=str, default="hard", choices=["easy", "medium", "hard"])
    ap.add_argument("--tasks", type=str, default="", help="Comma list for currentsweep (default: canonical 10).")
    ap.add_argument("--task", type=str, default="surface_pollution_cleanup_multiagent", help="Task id for scaling table.")
    ap.add_argument("--dump-json", action="store_true", help="Also dump a machine-readable JSON next to the md.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    runs = _parse_runs([str(x) for x in list(args.run or [])])
    if not runs:
        raise SystemExit("at least one --run is required")

    if str(args.table) == "main":
        export_main(out_dir=out_dir, runs=runs, difficulty=str(args.difficulty))
    elif str(args.table) == "currents":
        tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
        if not tasks:
            tasks = [
                "go_to_goal_current",
                "station_keeping",
                "surface_pollution_cleanup_multiagent",
                "underwater_pollution_lift_5uuv",
                "fish_herding_8uuv",
                "area_scan_terrain_recon",
                "pipeline_inspection_leak_detection",
                "route_following_waypoints",
                "depth_profile_tracking",
                "formation_transit_multiagent",
            ]
        export_currents(out_dir=out_dir, runs=runs, difficulty=str(args.difficulty), tasks=tasks)
    elif str(args.table) == "disturbances":
        tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
        if not tasks:
            tasks = [
                "go_to_goal_current",
                "station_keeping",
                "surface_pollution_cleanup_multiagent",
                "underwater_pollution_lift_5uuv",
                "fish_herding_8uuv",
                "area_scan_terrain_recon",
                "pipeline_inspection_leak_detection",
                "route_following_waypoints",
                "depth_profile_tracking",
                "formation_transit_multiagent",
            ]
        export_disturbances(out_dir=out_dir, runs=runs, difficulty=str(args.difficulty), tasks=tasks)
    elif str(args.table) == "planning_suite":
        export_planning_suite(out_dir=out_dir, runs=runs, difficulty=str(args.difficulty))
    elif str(args.table) == "planning_suite_cost":
        export_planning_suite_cost(out_dir=out_dir, runs=runs, difficulty=str(args.difficulty))
    elif str(args.table) == "difficulty_ladder":
        tasks = [t.strip() for t in str(args.tasks).split(",") if t.strip()]
        if not tasks:
            tasks = [
                "go_to_goal_current",
                "route_following_waypoints",
                "area_scan_terrain_recon",
                "pipeline_inspection_leak_detection",
                "surface_pollution_cleanup_multiagent",
            ]
        export_difficulty_ladder(out_dir=out_dir, runs=runs, tasks=tasks)
    else:
        export_scaling(out_dir=out_dir, runs=runs, task=str(args.task), difficulty=str(args.difficulty))

    if bool(args.dump_json):
        (out_dir / "export_meta.json").write_text(
            json.dumps(
                {
                    "table": str(args.table),
                    "difficulty": str(args.difficulty),
                    "runs": [{"label": r.label, "root": str(r.root)} for r in runs],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
