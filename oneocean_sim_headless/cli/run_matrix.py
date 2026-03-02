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
from oneocean_sim_headless.tasks import CANONICAL_TASKS_10, preset_task
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
    ap.add_argument("--preset", type=str, default="", choices=["", "smoke", "hero", "hero_full10"], help="Convenience presets (override tasks/difficulties/seeds/episodes).")
    ap.add_argument("--seeds", type=str, default="0-9", help="Seed range like '0-9' or list like '0,1,2'")
    ap.add_argument("--episodes", type=int, default=1, help="Episodes per seed (writes episode subfolders when >1).")
    ap.add_argument("--seed-step", type=int, default=1, help="Seed increment per episode within a base seed.")
    ap.add_argument("--shard-index", type=int, default=0, help="Optional sharding for parallel runs (0 <= shard_index < shard_count).")
    ap.add_argument("--shard-count", type=int, default=1, help="Optional sharding for parallel runs (number of shards).")
    ap.add_argument("--tasks", type=str, default="go_to_goal_current,station_keeping,pollution_localization,pollution_containment_multiagent")
    ap.add_argument("--difficulties", type=str, default="easy,medium,hard")
    ap.add_argument("--pollution-models", type=str, default="gaussian", help="Comma list: gaussian,ocpnet_3d")
    ap.add_argument("--n-agents", type=int, default=-1, help="Override n_agents for all scenarios (useful for scaling sweeps).")
    ap.add_argument("--controller-override", type=str, default="", help="Override controller kind for all tasks (e.g., mlp_bc).")
    ap.add_argument("--bc-weights-npz", type=str, default="", help="Weights for controller=mlp_bc (exported bc_mlp_v1_weights.npz).")
    ap.add_argument("--llm-model-path", type=str, default="", help="Local LLM model path for controller=llm_planner (no download).")
    ap.add_argument("--llm-cache-dir", type=str, default="", help="Cache directory for deterministic LLM outputs (JSON per state).")
    ap.add_argument("--llm-call-stride-steps", type=int, default=30, help="Max frequency of LLM calls (steps).")
    ap.add_argument("--llm-max-new-tokens", type=int, default=192, help="Generation budget for LLM JSON outputs.")
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--tile-x", type=float, default=600.0)
    ap.add_argument("--tile-z", type=float, default=600.0)
    ap.add_argument("--constraint-mode", type=str, default="hard", choices=["off", "hard"])
    ap.add_argument("--bathy-mode", type=str, default="off", choices=["off", "hard"])
    ap.add_argument("--seafloor-clearance-m", type=float, default=1.0)
    ap.add_argument("--current-gain", type=float, default=1.0, help="Scale dataset currents (ablation/stress-test).")
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

    shard_index = int(args.shard_index)
    shard_count = int(args.shard_count)
    if shard_count < 1:
        raise SystemExit("--shard-count must be >= 1")
    if shard_index < 0 or shard_index >= shard_count:
        raise SystemExit(f"--shard-index must satisfy 0 <= shard_index < shard_count (got {shard_index} / {shard_count})")

    meta: dict[str, Any] = {
        "time_local": datetime.now().isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "cmd": " ".join([shlex_quote(a) for a in sys.argv]),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": str(Path.cwd().resolve()),
        "out_dir": str(root),
        "shard": {"index": int(shard_index), "count": int(shard_count)},
    }
    meta["git"] = _git_state()
    (root / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    _write_env_snapshot(root)

    if str(args.preset).strip() == "smoke":
        args.tasks = ",".join(list(CANONICAL_TASKS_10))
        args.difficulties = "easy,medium,hard"
        args.pollution_models = "gaussian"
        args.seeds = "0"
        args.episodes = 1
    elif str(args.preset).strip() == "hero":
        # Designed to hit: seeds>=5 and episodes>=10 (interpreting episodes as seeds×episodes).
        # Keep small enough to run routinely.
        args.tasks = ",".join(
            [
                "go_to_goal_current",
                "station_keeping",
                "surface_pollution_cleanup_multiagent",
                "area_scan_terrain_recon",
                "formation_transit_multiagent",
                "pipeline_inspection_leak_detection",
                "route_following_waypoints",
                "depth_profile_tracking",
            ]
        )
        args.difficulties = "medium,hard"
        args.pollution_models = "gaussian"
        args.seeds = "0-4"
        args.episodes = 2
    elif str(args.preset).strip() == "hero_full10":
        # Full canonical 10-task suite for paper-scale reporting.
        args.tasks = ",".join(list(CANONICAL_TASKS_10))
        args.difficulties = "medium,hard"
        args.pollution_models = "gaussian"
        args.seeds = "0-9"
        args.episodes = 2

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    diffs = [d.strip() for d in args.difficulties.split(",") if d.strip()]
    pmods = [p.strip() for p in args.pollution_models.split(",") if p.strip()]
    seeds = _parse_seeds(str(args.seeds))
    episodes = int(max(1, int(args.episodes)))
    seed_step = int(max(1, int(args.seed_step)))

    # Task-specific default controllers and agent counts.
    defaults = {
        "go_to_goal_current": {"controller": "go_to_goal", "n_agents": 2},
        "station_keeping": {"controller": "station_keep", "n_agents": 2},
        "surface_pollution_cleanup_multiagent": {"controller": "go_to_goal", "n_agents": 10},
        "underwater_pollution_lift_5uuv": {"controller": "go_to_goal", "n_agents": 5},
        "fish_herding_8uuv": {"controller": "go_to_goal", "n_agents": 8},
        # Scanning is inherently multi-agent in our paper setup; default to N=8 for reliable coverage.
        "area_scan_terrain_recon": {"controller": "go_to_goal", "n_agents": 8},
        "pipeline_inspection_leak_detection": {"controller": "go_to_goal", "n_agents": 2},
        "route_following_waypoints": {"controller": "go_to_goal", "n_agents": 2},
        "depth_profile_tracking": {"controller": "go_to_goal", "n_agents": 2},
        "formation_transit_multiagent": {"controller": "go_to_goal", "n_agents": 10},
        # legacy/internal
        "pollution_localization": {"controller": "plume_gradient", "n_agents": 8},
        "pollution_containment_multiagent": {"controller": "containment_ring", "n_agents": 10},
    }
    if str(args.preset).strip() == "hero":
        # Hero runs should showcase multi-agent settings (N=8/10) even for simple tasks.
        for k in ("go_to_goal_current", "station_keeping", "area_scan_terrain_recon", "pipeline_inspection_leak_detection", "route_following_waypoints", "depth_profile_tracking"):
            if k in defaults:
                defaults[k]["n_agents"] = 8
    elif str(args.preset).strip() == "hero_full10":
        # Same as hero: use N=8/10 for paper-facing multi-agent evaluation (except fixed-N tasks).
        for k in ("go_to_goal_current", "station_keeping", "area_scan_terrain_recon", "pipeline_inspection_leak_detection", "route_following_waypoints", "depth_profile_tracking"):
            if k in defaults:
                defaults[k]["n_agents"] = 8
    # Global override (e.g., scaling sweeps). Tasks with required_n_agents will fail fast in env.reset.
    if int(args.n_agents) > 0:
        for v in defaults.values():
            v["n_agents"] = int(args.n_agents)

    scenarios: list[Scenario] = []
    for task in tasks:
        if task not in defaults:
            raise ValueError(f"Unknown task: {task}")
        for diff in diffs:
            for pm in pmods:
                for seed in seeds:
                    ctrl = str(defaults[task]["controller"])
                    if str(args.controller_override).strip():
                        ctrl = str(args.controller_override).strip()
                    scenarios.append(
                        Scenario(
                            task=task,
                            difficulty=diff,
                            controller=ctrl,
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
                "constraint_mode": str(args.constraint_mode),
                "bathy_mode": str(args.bathy_mode),
                "seafloor_clearance_m": float(args.seafloor_clearance_m),
                "preset": str(args.preset),
                "tasks": tasks,
                "difficulties": diffs,
                "pollution_models": pmods,
                "seeds": seeds,
                "episodes": int(episodes),
                "seed_step": int(seed_step),
                "shard_index": int(shard_index),
                "shard_count": int(shard_count),
                "scenario_count": int(len(scenarios)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    rows = []
    t_all = time.time()
    for idx, sc in enumerate(scenarios):
        if int(shard_count) > 1 and (int(idx) % int(shard_count)) != int(shard_index):
            continue
        base_run_dir = root / f"{sc.task}" / f"{sc.difficulty}" / f"{sc.pollution_model}" / f"n{int(sc.n_agents)}" / f"seed_{sc.seed:03d}"
        base_run_dir.mkdir(parents=True, exist_ok=True)

        env_cfg = EnvConfig(
            drift_cache_npz=str(Path(args.drift_npz).expanduser()),
            pollution_model=str(sc.pollution_model),  # type: ignore[arg-type]
            dt_s=float(args.dt),
            tile_size_x_m=float(args.tile_x),
            tile_size_z_m=float(args.tile_z),
            constraint_mode=str(args.constraint_mode),  # type: ignore[arg-type]
            bathy_mode=str(args.bathy_mode),  # type: ignore[arg-type]
            seafloor_clearance_m=float(args.seafloor_clearance_m),
            current_gain=float(args.current_gain),
        )
        task_cfg = preset_task(kind=str(sc.task), difficulty=str(sc.difficulty))  # type: ignore[arg-type]
        ctrl_cfg = preset_controller(
            kind=str(sc.controller),  # type: ignore[arg-type]
            max_speed_mps=env_cfg.max_speed_mps,
            bc_weights_npz=str(args.bc_weights_npz),
            llm_model_path=str(args.llm_model_path),
            llm_cache_dir=str(args.llm_cache_dir),
            llm_call_stride_steps=int(args.llm_call_stride_steps),
            llm_max_new_tokens=int(args.llm_max_new_tokens),
        )

        for ep in range(int(episodes)):
            seed = int(sc.seed) + ep * int(seed_step)
            run_dir = base_run_dir if int(episodes) == 1 else (base_run_dir / f"episode_{ep:03d}")
            run_dir.mkdir(parents=True, exist_ok=True)

            t0 = time.time()
            env = HeadlessOceanEnv(env_cfg, out_dir=run_dir, seed=int(seed), n_agents=int(sc.n_agents))
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
                "seed": int(seed),
                "base_seed": int(sc.seed),
                "episode_index": int(ep),
                "n_agents": int(sc.n_agents),
                "steps": int(steps),
                "dt_s": float(args.dt),
                "success": bool(last.get("success", False)),
                "time_to_success_s": env.time_to_success_s,
                "energy_proxy": float(env.energy_proxy),
                "constraint_violations": int(env.constraint_violations),
                "final": last,
            }
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
                    "seed": int(seed),
                    "base_seed": int(sc.seed),
                    "episode_index": int(ep),
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
                    "waypoint_index": last.get("waypoint_index", None),
                    "formation_err_m": last.get("formation_err_m", None),
                    "coverage": last.get("coverage", None),
                    "leaks_detected": last.get("leaks_detected", None),
                    "fish_progress": last.get("fish_progress", None),
                    "lift_phase": last.get("lift_phase", None),
                    "elapsed_s": float(time.time() - t0),
                    "run_dir": str(run_dir),
                }
            )

        if (idx + 1) % 25 == 0 or (idx + 1) == len(scenarios):
            print(json.dumps({"done": int(idx + 1), "total": int(len(scenarios))}, indent=2))

    # Write summary.csv (paper-facing contract) and keep matrix_results.csv for backward compat.
    out_csv = root / "summary.csv"
    cols = list(rows[0].keys()) if rows else []
    with out_csv.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (root / "matrix_results.csv").write_text(out_csv.read_text(encoding="utf-8"), encoding="utf-8")

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
    (root / "results_manifest.json").write_text(
        json.dumps(
            {
                "track": "h1_headless_matrix",
                "out_dir": str(root),
                "meta_json": str((root / "meta.json").resolve()),
                "matrix_config_json": str((root / "matrix_config.json").resolve()),
                "summary_csv": str(out_csv.resolve()),
                "matrix_summary_json": str((root / "matrix_summary.json").resolve()),
                "scenario_count": int(len(scenarios)),
                "rows": int(len(rows)),
                "elapsed_s": float(time.time() - t_all),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _append_run_index(root=root, meta=meta)
    print(json.dumps({"out_dir": str(root), "csv": str(out_csv), "elapsed_s": float(time.time() - t_all)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
