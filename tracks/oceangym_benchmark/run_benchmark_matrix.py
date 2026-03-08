from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


def _tag_now_local() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_ssl_cert_file(env: dict[str, str]) -> None:
    try:
        import certifi  # type: ignore

        env.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        return


@dataclass(frozen=True)
class RunSpec:
    name: str
    args: list[str]


def _run_one(*, python: str, repo_root: Path, spec: RunSpec, out_root: Path, dry_run: bool) -> None:
    out_dir = (out_root / spec.name).resolve()
    cmd = [python, "tracks/oceangym_benchmark/run_task_suite.py", "--out_dir", str(out_dir)] + list(spec.args)
    env = dict(os.environ)
    _ensure_ssl_cert_file(env)

    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "run_index.jsonl").open("a", encoding="utf-8").write(json.dumps({"name": spec.name, "cmd": cmd}) + "\n")

    print("[oceangym-benchmark] run:", spec.name)
    print("[oceangym-benchmark] out:", out_dir)
    if dry_run:
        print("[oceangym-benchmark] cmd:", " ".join(cmd))
        return
    subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


def _matrix(*, strong_scale: float, current_npz: str, episodes_nav: int, episodes_plume: int, n_multiagent: int) -> list[RunSpec]:
    worlds_all = ["Dam-HoveringCamera", "PierHarbor-HoveringCamera", "OpenWater-HoveringCamera", "SimpleUnderwater-Hovering"]
    worlds_key = ["PierHarbor-HoveringCamera", "OpenWater-HoveringCamera"]

    common = [
        "--preset",
        "ocean_worlds_camera",
        "--no_media",
        "--current_npz",
        current_npz,
        "--current_depth_m",
        "0.494025",
        "--fog_density",
        "0.006",
        "--n_multiagent",
        str(int(n_multiagent)),
    ]

    specs: list[RunSpec] = []

    # Slice A: difficulty ladder (4 single-agent tasks × 3 difficulties).
    ladder_tasks = [
        "go_to_goal_current",
        "station_keeping",
        "route_following_waypoints",
        "depth_profile_tracking",
    ]
    for diff in ["easy", "medium", "hard"]:
        specs.append(
            RunSpec(
                name=f"A_difficulty_ladder_{diff}",
                args=common
                + [
                    "--difficulty",
                    diff,
                    "--episodes",
                    str(int(episodes_nav)),
                    "--ctrl_profile",
                    "default",
                    "--dataset_days_per_sim_second",
                    "0.0",
                    "--current_scale",
                    "1.0",
                    "--tasks",
                    *ladder_tasks,
                ],
            )
        )

    # Slice B: robustness under currents (2 tasks × 3 current scales × 2 controller profiles × optional time-varying).
    robust_tasks = ["go_to_goal_current", "route_following_waypoints"]
    for ds_days, worlds in [("0.0", worlds_all), ("0.1", worlds_key)]:
        for scale in ["0.0", "1.0", str(float(strong_scale))]:
            for prof in ["default", "weak"]:
                tag = f"B_robust_ds{ds_days}_scale{scale}_ctrl{prof}"
                specs.append(
                    RunSpec(
                        name=tag,
                        args=common
                        + [
                            "--difficulty",
                            "medium",
                            "--episodes",
                            str(int(episodes_nav)),
                            "--ctrl_profile",
                            prof,
                            "--dataset_days_per_sim_second",
                            ds_days,
                            "--current_scale",
                            scale,
                            "--scenarios",
                            *worlds,
                            "--tasks",
                            *robust_tasks,
                        ],
                    )
                )

    # Slice C: pollution model comparison (localization + containment).
    plume_tasks = [
        "surface_pollution_cleanup_multiagent__localization",
        "surface_pollution_cleanup_multiagent__containment",
    ]
    for model in ["analytic", "ocpnet_3d"]:
        for scale in ["1.0", str(float(strong_scale))]:
            specs.append(
                RunSpec(
                    name=f"C_pollution_model_{model}_scale{scale}",
                    args=common
                    + [
                        "--difficulty",
                        "medium",
                        "--episodes",
                        str(int(episodes_plume)),
                        "--ctrl_profile",
                        "default",
                        "--dataset_days_per_sim_second",
                        "0.0",
                        "--current_scale",
                        scale,
                        "--pollution_model",
                        model,
                        "--scenarios",
                        *worlds_all,
                        "--tasks",
                        *plume_tasks,
                    ],
                )
            )

    # Slice D: multi-agent scaling (N sweep) on key worlds.
    scaling_tasks = [
        "formation_transit_multiagent",
        "surface_pollution_cleanup_multiagent__containment",
    ]
    for n in [2, 4, 8, 10]:
        specs.append(
            RunSpec(
                name=f"D_scaling_N{n}",
                args=common
                + [
                    "--difficulty",
                    "medium",
                    "--episodes",
                    str(int(episodes_plume)),
                    "--ctrl_profile",
                    "default",
                    "--dataset_days_per_sim_second",
                    "0.0",
                    "--current_scale",
                    "1.0",
                    "--n_multiagent",
                    str(int(n)),
                    "--pollution_model",
                    "analytic",
                    "--scenarios",
                    *worlds_key,
                    "--tasks",
                    *scaling_tasks,
                ],
            )
        )

    return specs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root", default=None, help="Run root folder under runs/oceangym_benchmark/ (default: oceangym_latest_<timestamp>).")
    ap.add_argument("--dry_run", action="store_true")
    ap.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional list of slice prefixes to run (e.g., A_difficulty_ladder B_robust D_scaling). Default: run all.",
    )
    ap.add_argument("--strong_scale", type=float, default=2.5)
    ap.add_argument("--episodes_nav", type=int, default=5)
    ap.add_argument("--episodes_plume", type=int, default=3)
    ap.add_argument("--n_multiagent", type=int, default=8)
    ap.add_argument(
        "--current_npz",
        default="runs/_cache/data_grounding/currents/cmems_center_uovo.npz",
        help="Data-grounded current series exported from combined_environment.nc.",
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parents[2]
    python = sys.executable

    out_root = Path(args.out_root) if args.out_root else Path("runs") / "oceangym_benchmark" / f"oceangym_latest_{_tag_now_local()}"
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    specs = _matrix(
        strong_scale=float(args.strong_scale),
        current_npz=str(args.current_npz),
        episodes_nav=int(args.episodes_nav),
        episodes_plume=int(args.episodes_plume),
        n_multiagent=int(args.n_multiagent),
    )
    (out_root / "latest_plan.json").write_text(json.dumps([s.__dict__ for s in specs], indent=2) + "\n", encoding="utf-8")

    only = [str(x).strip() for x in (args.only or []) if str(x).strip()]
    if only:
        specs = [s for s in specs if any(s.name.startswith(pfx) for pfx in only)]

    for spec in specs:
        _run_one(python=python, repo_root=repo_root, spec=spec, out_root=out_root, dry_run=bool(args.dry_run))

    print("[oceangym-benchmark] done:", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
