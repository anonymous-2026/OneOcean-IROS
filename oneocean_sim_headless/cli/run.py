from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from ..controllers import ControllerConfig
from ..env import EnvConfig, HeadlessOceanEnv
from ..tasks import TaskConfig
from ..validators import validate_run_dir


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="H1 headless runner (no UI): simulate + record MIMIR-inspired streams.")
    ap.add_argument("--drift-npz", type=str, required=True, help="Drift cache .npz exported from combined_environment.nc variants")
    ap.add_argument("--task", type=str, required=True, choices=[
        "go_to_goal_current",
        "station_keeping",
        "pollution_localization",
        "pollution_containment_multiagent",
    ])
    ap.add_argument("--controller", type=str, required=True, choices=["go_to_goal", "station_keep", "plume_gradient", "containment_ring"])
    ap.add_argument("--pollution-model", type=str, default="gaussian", choices=["gaussian", "ocpnet_3d"])
    ap.add_argument("--n-agents", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=1, help="Number of episodes to run (writes episode subfolders when >1).")
    ap.add_argument("--seed-step", type=int, default=1, help="Seed increment per episode.")
    ap.add_argument("--dt", type=float, default=1.0)
    ap.add_argument("--max-steps", type=int, default=240)
    ap.add_argument("--out-dir", type=str, default="")
    ap.add_argument("--validate", action="store_true", help="Validate recording integrity after run.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_id = f"h1_headless_{args.task}_{stamp}_seed{int(args.seed)}_n{int(args.n_agents)}"
    base_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (Path("runs") / "headless" / batch_id).resolve()
    base_dir.mkdir(parents=True, exist_ok=True)

    env_cfg = EnvConfig(drift_cache_npz=str(Path(args.drift_npz).expanduser()), pollution_model=str(args.pollution_model), dt_s=float(args.dt))
    task_cfg = TaskConfig(kind=str(args.task), max_steps=int(args.max_steps))
    ctrl_cfg = ControllerConfig(kind=str(args.controller), max_speed_mps=env_cfg.max_speed_mps)

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

        metrics = {
            "task": str(args.task),
            "controller": str(args.controller),
            "pollution_model": str(args.pollution_model),
            "seed": int(seed),
            "n_agents": int(args.n_agents),
            "steps": int(steps),
            "dt_s": float(args.dt),
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

    (base_dir / "batch_metrics.json").write_text(json.dumps(batch_metrics, indent=2), encoding="utf-8")
    print(json.dumps({"out_dir": str(base_dir), "episodes": int(len(batch_metrics))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
