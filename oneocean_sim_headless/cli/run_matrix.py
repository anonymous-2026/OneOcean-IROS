#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from oneocean_sim_headless.controllers import preset_controller
from oneocean_sim_headless.env import EnvConfig, HeadlessOceanEnv
from oneocean_sim_headless.tasks import preset_task
from oneocean_sim_headless.validators import validate_run_dir


def shlex_quote(s: str) -> str:
    # Minimal shlex.quote replacement to avoid importing shlex (keep file lightweight).
    if not s:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@%+=:,./-")
    if all(c in safe for c in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _git_state() -> dict[str, Any]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        sha = ""
    try:
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip())
    except Exception:
        dirty = None
    return {"sha": sha, "dirty": dirty}


def _write_env_snapshot(root: Path) -> None:
    try:
        txt = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], stderr=subprocess.STDOUT, text=True)
    except Exception as e:
        txt = f"# pip freeze unavailable: {type(e).__name__}: {e}\n"
    (root / "env_pip_freeze.txt").write_text(txt, encoding="utf-8")


def _append_run_index(*, root: Path, meta: dict[str, Any]) -> None:
    index_path = Path("runs") / "index.jsonl"
    if not index_path.exists():
        return
    entry = {
        "time_local": meta.get("time_local", ""),
        "kind": "headless_matrix",
        "path": str(root),
        "cmd": meta.get("cmd", ""),
        "git": meta.get("git", {}),
    }
    with index_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


@dataclass(frozen=True)
class Scenario:
    task: str
    difficulty: str
    controller: str
    pollution_model: str
    n_agents: int
    seed: int


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run a headless experiment matrix and aggregate results (H1).")
    ap.add_argument("--drift-npz", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--seeds", type=str, default="0-9", help="Seed range like '0-9' or list like '0,1,2'")
    ap.add_argument("--tasks", type=str, default="go_to_goal_current,station_keeping,pollution_localization,pollution_containment_multiagent")
    ap.add_argument("--difficulties", type=str, default="easy,medium,hard")
    ap.add_argument("--pollution-models", type=str, default="gaussian", help="Comma list: gaussian,ocpnet_3d")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--tile-x", type=float, default=600.0)
    ap.add_argument("--tile-z", type=float, default=600.0)
    ap.add_argument("--constraint-mode", type=str, default="hard", choices=["off", "hard"])
    ap.add_argument("--bathy-mode", type=str, default="off", choices=["off", "hard"])
    ap.add_argument("--seafloor-clearance-m", type=float, default=1.0)
    return ap.parse_args()


def _parse_seeds(spec: str) -> list[int]:
    s = spec.strip()
    if "," in s:
        return [int(x) for x in s.split(",") if x.strip()]
    if "-" in s:
        a, b = s.split("-", 1)
        lo = int(a.strip())
        hi = int(b.strip())
        if hi < lo:
            lo, hi = hi, lo
        return list(range(lo, hi + 1))
    return [int(s)]


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path("runs") / "headless" / f"matrix_{stamp}").resolve()
    root.mkdir(parents=True, exist_ok=True)

    meta: dict[str, Any] = {
        "time_local": datetime.now().isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "cmd": " ".join([shlex_quote(a) for a in sys.argv]),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": str(Path.cwd().resolve()),
        "out_dir": str(root),
    }
    meta["git"] = _git_state()
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_env_snapshot(root)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    diffs = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    pmods = [p.strip() for p in args.pollution_models.split(",") if p.strip()]
    seeds = _parse_seeds(str(args.seeds))

    # Task-specific default controllers and agent counts.
    defaults = {
        "go_to_goal_current": {"controller": "go_to_goal", "n_agents": 2},
        "station_keeping": {"controller": "station_keep", "n_agents": 2},
        "pollution_localization": {"controller": "plume_gradient", "n_agents": 8},
        "pollution_containment_multiagent": {"controller": "containment_ring", "n_agents": 10},
    }

    scenarios: list[Scenario] = []
    for task in tasks:
        if task not in defaults:
            raise ValueError(f"Unknown task: {task}")
        for diff in diffs:
            for pm in pmods:
                for seed in seeds:
                    scenarios.append(
                        Scenario(
                            task=task,
                            difficulty=diff,
                            controller=str(defaults[task]["controller"]),
                            pollution_model=pm,
                            n_agents=int(defaults[task]["n_agents"]),
                            seed=int(seed),
                        )
                    )

    (root / "matrix_config.json").write_text(
        json.dumps(
            {
                "drift_npz": str(Path(args.drift_npz).expanduser()),
                "dt_s": float(args.dt),
                "tile_size_x_m": float(args.tile_x),
                "tile_size_z_m": float(args.tile_z),
                "tasks": tasks,
                "difficulties": diffs,
                "pollution_models": pmods,
                "seeds": seeds,
                "scenario_count": int(len(scenarios)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = []
    t_all = time.time()
    for idx, sc in enumerate(scenarios):
        run_dir = root / f"{sc.task}" / f"{sc.difficulty}" / f"{sc.pollution_model}" / f"seed_{sc.seed:03d}"
        run_dir.mkdir(parents=True, exist_ok=True)

        env_cfg = EnvConfig(
            drift_cache_npz=str(Path(args.drift_npz).expanduser()),
            pollution_model=str(sc.pollution_model),  # type: ignore[arg-type]
            dt_s=float(args.dt),
            tile_size_x_m=float(args.tile_x),
            tile_size_z_m=float(args.tile_z),
            constraint_mode=str(args.constraint_mode),  # type: ignore[arg-type]
            bathy_mode=str(args.bathy_mode),  # type: ignore[arg-type]
            seafloor_clearance_m=float(args.seafloor_clearance_m),
        )
        task_cfg = preset_task(kind=str(sc.task), difficulty=str(sc.difficulty))  # type: ignore[arg-type]
        ctrl_cfg = preset_controller(kind=str(sc.controller), max_speed_mps=env_cfg.max_speed_mps)  # type: ignore[arg-type]

        t0 = time.time()
        env = HeadlessOceanEnv(env_cfg, out_dir=run_dir, seed=int(sc.seed), n_agents=int(sc.n_agents))
        env.reset(task=task_cfg, controller=ctrl_cfg)
        done = False
        last = {}
        steps = 0
        while not done:
            done, info = env.step()
            last = info
            steps += 1
        env.close()

        metrics = {
            "task": sc.task,
            "difficulty": sc.difficulty,
            "controller": sc.controller,
            "pollution_model": sc.pollution_model,
            "seed": int(sc.seed),
            "n_agents": int(sc.n_agents),
            "steps": int(steps),
            "dt_s": float(args.dt),
            "success": bool(last.get("success", False)),
            "time_to_success_s": env.time_to_success_s,
            "energy_proxy": float(env.energy_proxy),
            "constraint_violations": int(env.constraint_violations),
            "final": last,
        }
        # Mirror the runner convention (metrics.json + metrics.csv).
        try:
            env.rec.write_metrics(metrics)
        except Exception:
            (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

        if bool(args.validate):
            vr = validate_run_dir(run_dir)
            (run_dir / "validation.json").write_text(json.dumps(asdict(vr), indent=2), encoding="utf-8")
            if not vr.ok:
                print(json.dumps({"scenario_index": idx, "run_dir": str(run_dir), "ok": False, "reason": vr.reason}, indent=2))
                return 2

        rows.append(
            {
                "task": sc.task,
                "difficulty": sc.difficulty,
                "controller": sc.controller,
                "pollution_model": sc.pollution_model,
                "n_agents": int(sc.n_agents),
                "seed": int(sc.seed),
                "success": bool(metrics["success"]),
                "time_to_success_s": metrics["time_to_success_s"],
                "steps": int(steps),
                "dt_s": float(args.dt),
                "energy_proxy": metrics["energy_proxy"],
                "constraint_violations": metrics["constraint_violations"],
                "best_dist_to_goal_m": last.get("best_dist_to_goal_m", None),
                "source_error_m": last.get("source_error_m", None),
                "mass_frac": last.get("mass_frac", None),
                "target": last.get("target", None),
                "elapsed_s": float(time.time() - t0),
                "run_dir": str(run_dir),
            }
        )

        if (idx + 1) % 25 == 0 or (idx + 1) == len(scenarios):
            print(json.dumps({"done": int(idx + 1), "total": int(len(scenarios))}, indent=2))

    # Write CSV.
    out_csv = root / "matrix_results.csv"
    cols = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Aggregate summary (success rate by task/difficulty/model).
    # Use nested dicts so JSON serialization stays valid (tuple keys are not JSON-serializable).
    summary: dict[str, Any] = {}
    for r in rows:
        task = str(r["task"])
        diff = str(r["difficulty"])
        pm = str(r["pollution_model"])
        n_agents = str(int(r["n_agents"]))
        entry = (
            summary.setdefault(task, {})
            .setdefault(diff, {})
            .setdefault(pm, {})
            .setdefault(n_agents, {"count": 0, "success": 0, "mean_energy": 0.0, "mean_steps": 0.0})
        )
        entry["count"] += 1
        entry["success"] += int(bool(r["success"]))
        entry["mean_energy"] += float(r["energy_proxy"])
        entry["mean_steps"] += float(r["steps"])
    for _task, v_task in summary.items():
        for _diff, v_diff in v_task.items():
            for _pm, v_pm in v_diff.items():
                for _n, v in v_pm.items():
                    c = max(1, int(v["count"]))
                    v["success_rate"] = float(v["success"]) / float(c)
                    v["mean_energy"] = float(v["mean_energy"]) / float(c)
                    v["mean_steps"] = float(v["mean_steps"]) / float(c)

    (root / "matrix_summary.json").write_text(json.dumps({"summary": summary, "elapsed_s": float(time.time() - t_all)}, indent=2), encoding="utf-8")
    _append_run_index(root=root, meta=meta)
    print(json.dumps({"out_dir": str(root), "csv": str(out_csv), "elapsed_s": float(time.time() - t_all)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
