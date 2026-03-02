from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from ..controllers import ControllerConfig, preset_controller
from ..env import EnvConfig, HeadlessOceanEnv
from ..tasks import CANONICAL_TASKS_10, TaskConfig, preset_task
from ..validators import validate_run_dir


def _shlex_quote(s: str) -> str:
    if not s:
        return "''"
    safe = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_@%+=:,./-")
    if all(c in safe for c in s):
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def _git_state() -> dict[str, object]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        sha = ""
    try:
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL, text=True).strip())
    except Exception:
        dirty = None
    return {"sha": sha, "dirty": dirty}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="H1 headless runner (no UI): simulate + record MIMIR-inspired streams.")
    ap.add_argument("--drift-npz", type=str, required=True, help="Drift cache .npz exported from combined_environment.nc variants")
    ap.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            *CANONICAL_TASKS_10,
            # legacy/internal
            "pollution_localization",
            "pollution_containment_multiagent",
        ],
    )
    ap.add_argument("--difficulty", type=str, default="medium", choices=["easy", "medium", "hard"])
    ap.add_argument("--controller", type=str, required=True, choices=["go_to_goal", "station_keep", "plume_gradient", "containment_ring", "mlp_bc"])
    ap.add_argument("--bc-weights-npz", type=str, default="", help="Required for controller=mlp_bc: path to exported bc_mlp_v1_weights.npz")
    ap.add_argument("--pollution-model", type=str, default="gaussian", choices=["gaussian", "ocpnet_3d"])
    ap.add_argument("--n-agents", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1, help="Number of episodes to run (writes episode subfolders when >1).")
    ap.add_argument("--seed-step", type=int, default=1, help="Seed increment per episode.")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--dynamics-model", type=str, default="kinematic", choices=["kinematic", "3dof", "6dof"])
    ap.add_argument("--constraint-mode", type=str, default="hard", choices=["off", "hard"], help="Hard constraints using land_mask (invalid regions).")
    ap.add_argument("--bathy-mode", type=str, default="off", choices=["off", "hard"], help="Hard constraints using elevation vs agent depth (touchdown/too-shallow).")
    ap.add_argument("--seafloor-clearance-m", type=float, default=1.0, help="Minimum clearance above seafloor when bathy-mode=hard.")
    ap.add_argument("--max-steps", type=int, default=-1, help="Override max steps; -1 uses task preset.")
    ap.add_argument("--success-radius", type=float, default=-1.0, help="Override success radius (meters); -1 uses task preset.")
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--validate", action="store_true", help="Validate recording integrity after run.")
    ap.add_argument("--render", action="store_true", help="Write a tiny top-down MP4 + keyframe PNG into each episode dir.")
    ap.add_argument("--render-stride", type=int, default=2, help="Frame stride when rendering (2 => every 2nd step).")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"h1_headless_{args.task}_{stamp}_seed{int(args.seed)}_n{int(args.n_agents)}"
    base_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path("runs") / "headless" / batch_id).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "time_local": datetime.now().isoformat(timespec="seconds"),
        "argv": list(sys.argv),
        "cmd": " ".join([_shlex_quote(a) for a in sys.argv]),
        "python_executable": sys.executable,
        "python_version": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "cwd": str(Path.cwd().resolve()),
        "git": _git_state(),
    }

    env_cfg = EnvConfig(
        drift_cache_npz=str(Path(args.drift_npz).expanduser()),
        pollution_model=str(args.pollution_model),
        dt_s=float(args.dt),
        dynamics_model=str(args.dynamics_model),  # type: ignore[arg-type]
        constraint_mode=str(args.constraint_mode),  # type: ignore[arg-type]
        bathy_mode=str(args.bathy_mode),  # type: ignore[arg-type]
        seafloor_clearance_m=float(args.seafloor_clearance_m),
    )
    preset = preset_task(kind=str(args.task), difficulty=str(args.difficulty))  # type: ignore[arg-type]
    if int(args.max_steps) >= 1:
        preset = TaskConfig(**{**preset.to_dict(), "max_steps": int(args.max_steps)})
    if float(args.success_radius) > 0:
        preset = TaskConfig(**{**preset.to_dict(), "success_radius_m": float(args.success_radius)})
    task_cfg = preset
    ctrl_cfg = preset_controller(kind=str(args.controller), max_speed_mps=env_cfg.max_speed_mps, bc_weights_npz=str(args.bc_weights_npz))  # type: ignore[arg-type]

    batch_metrics = []
    for ep in range(int(max(1, args.episodes))):
        seed = int(args.seed) + ep * int(args.seed_step)
        out_dir = base_dir if int(args.episodes) == 1 else (base_dir / f"episode_{ep:03d}")
        out_dir.mkdir(parents=True, exist_ok=True)

        t0 = time.time()
        env = HeadlessOceanEnv(env_cfg, out_dir=out_dir, seed=seed, n_agents=int(args.n_agents))
        env.reset(task=task_cfg, controller=ctrl_cfg)

        done = False
        last_info = {}
        steps = 0
        while not done:
            done, info = env.step()
            last_info = info
            steps += 1
        env.close()

        success = bool(last_info.get("success", False))
        metrics = {
            "task": str(args.task),
            "difficulty": str(args.difficulty),
            "controller": str(args.controller),
            "pollution_model": str(args.pollution_model),
            "seed": int(seed),
            "n_agents": int(args.n_agents),
            "steps": int(steps),
            "dt_s": float(args.dt),
            "success": bool(success),
            "time_to_success_s": float(env.time_to_success_s) if env.time_to_success_s is not None else None,
            "energy_proxy": float(env.energy_proxy),
            "constraint_violations": int(env.constraint_violations),
            "elapsed_s": float(time.time() - t0),
            "final": last_info,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        try:
            env.rec.write_metrics(metrics)
        except Exception:
            pass

        if bool(args.validate):
            res = validate_run_dir(out_dir)
            (out_dir / "validation.json").write_text(json.dumps({"ok": res.ok, "reason": res.reason}, indent=2), encoding="utf-8")
            if not res.ok:
                print(json.dumps({"out_dir": str(out_dir), "ok": False, "reason": res.reason}, indent=2))
                return 2

        batch_metrics.append(metrics)

        if bool(args.render):
            try:
                from ..render import render_topdown_rollout  # local import (optional dep: imageio)

                render_topdown_rollout(run_dir=out_dir, out_mp4=out_dir / "rollout.mp4", out_keyframe=out_dir / "keyframe.png", stride=int(args.render_stride))
            except Exception as e:
                (out_dir / "render_error.txt").write_text(f"{type(e).__name__}: {e}\n", encoding="utf-8")

    # Root-level artifacts for paper-ready aggregation.
    (base_dir / "run_meta_root.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (base_dir / "batch_metrics.json").write_text(json.dumps(batch_metrics, indent=2), encoding="utf-8")
    # summary.csv (row = episode)
    cols = sorted({k for m in batch_metrics for k in m.keys()})
    with (base_dir / "summary.csv").open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for m in batch_metrics:
            w.writerow({k: m.get(k) for k in cols})
    # results_manifest.json (root index)
    manifest = {
        "track": "h1_headless",
        "out_dir": str(base_dir),
        "created_at_local": meta["time_local"],
        "git": meta["git"],
        "cmd": meta["cmd"],
        "drift_npz": str(Path(args.drift_npz).expanduser().resolve()),
        "task": str(args.task),
        "difficulty": str(args.difficulty),
        "controller": str(args.controller),
        "pollution_model": str(args.pollution_model),
        "n_agents": int(args.n_agents),
        "dt_s": float(args.dt),
        "constraint_mode": str(args.constraint_mode),
        "bathy_mode": str(args.bathy_mode),
        "seafloor_clearance_m": float(args.seafloor_clearance_m),
        "episodes": int(len(batch_metrics)),
        "episode_dirs": [str((base_dir if int(args.episodes) == 1 else (base_dir / f"episode_{i:03d}")).resolve()) for i in range(int(max(1, args.episodes)))],
        "summary_csv": str((base_dir / "summary.csv").resolve()),
    }
    (base_dir / "results_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(base_dir), "episodes": int(len(batch_metrics))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
