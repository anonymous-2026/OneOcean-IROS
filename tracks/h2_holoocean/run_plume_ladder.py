from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _tag_now_local() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


@dataclass(frozen=True)
class LadderLevel:
    name: str
    num_agents: int
    current_force_scale: float
    pollution_domain_xy_m: float
    localize_seconds: float
    contain_seconds: float
    contain_radius_m: float


_DEFAULT_LEVELS: dict[str, LadderLevel] = {
    "easy": LadderLevel(
        name="easy",
        num_agents=2,
        current_force_scale=4.0,
        pollution_domain_xy_m=90.0,
        localize_seconds=14.0,
        contain_seconds=12.0,
        contain_radius_m=8.0,
    ),
    "medium": LadderLevel(
        name="medium",
        num_agents=6,
        current_force_scale=6.5,
        pollution_domain_xy_m=120.0,
        localize_seconds=18.0,
        contain_seconds=16.0,
        contain_radius_m=10.0,
    ),
    "hero": LadderLevel(
        name="hero",
        num_agents=10,
        current_force_scale=8.0,
        pollution_domain_xy_m=160.0,
        localize_seconds=25.0,
        contain_seconds=20.0,
        contain_radius_m=10.0,
    ),
    "hard": LadderLevel(
        name="hard",
        num_agents=10,
        current_force_scale=12.0,
        pollution_domain_xy_m=220.0,
        localize_seconds=35.0,
        contain_seconds=28.0,
        contain_radius_m=14.0,
    ),
}


def _read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _run_one(
    *,
    run_py: Path,
    out_dir: Path,
    scenario: str,
    dataset_variant: str | None,
    combined_nc: str | None,
    time_index: int,
    depth_index: int,
    seed: int,
    level: LadderLevel,
    pollution_model: str,
    fps: int,
    render_quality: int,
    env: dict[str, str],
) -> dict:
    cmd: list[str] = [
        sys.executable,
        str(run_py),
        "--out-dir",
        str(out_dir),
        "--scenario",
        scenario,
        "--num-agents",
        str(level.num_agents),
        "--seed",
        str(seed),
        "--time-index",
        str(time_index),
        "--depth-index",
        str(depth_index),
        "--pollution-model",
        pollution_model,
        "--pollution-warmup-s",
        str(float(env.get("POLLUTION_WARMUP_S", "20"))),
        "--pollution-update-period-s",
        str(float(env.get("POLLUTION_UPDATE_PERIOD_S", "2"))),
        "--pollution-domain-xy-m",
        str(level.pollution_domain_xy_m),
        "--current-force-scale",
        str(level.current_force_scale),
        "--localize-seconds",
        str(level.localize_seconds),
        "--contain-seconds",
        str(level.contain_seconds),
        "--contain-radius-m",
        str(level.contain_radius_m),
        "--fps",
        str(fps),
        "--render-quality",
        str(render_quality),
    ]
    if dataset_variant:
        cmd += ["--dataset-variant", dataset_variant]
    if combined_nc:
        cmd += ["--combined-nc", combined_nc]
    subprocess.run(cmd, check=True, env=env)
    res = _read_json(out_dir / "results_manifest.json")
    res["ladder_level"] = {
        "name": level.name,
        "num_agents": level.num_agents,
        "current_force_scale": level.current_force_scale,
        "pollution_domain_xy_m": level.pollution_domain_xy_m,
        "localize_seconds": level.localize_seconds,
        "contain_seconds": level.contain_seconds,
        "contain_radius_m": level.contain_radius_m,
    }
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", type=str, default=os.environ.get("OUT_DIR", f"runs/h2_holoocean/plume_ladder_{_tag_now_local()}"))
    ap.add_argument("--scenario", type=str, default=os.environ.get("SCENARIO_NAME", "PierHarbor-HoveringCamera"))
    ap.add_argument("--levels", type=str, default=os.environ.get("LEVELS", "easy,hero,hard"), help="Comma-separated from: easy,medium,hero,hard")
    ap.add_argument("--seed", type=int, default=int(os.environ.get("SEED", "0")))
    ap.add_argument("--time-index", type=int, default=int(os.environ.get("TIME_INDEX", "0")))
    ap.add_argument("--depth-index", type=int, default=int(os.environ.get("DEPTH_INDEX", "0")))
    ap.add_argument("--dataset-variant", type=str, default=os.environ.get("DATASET_VARIANT", "tiny"))
    ap.add_argument("--combined-nc", type=str, default=os.environ.get("COMBINED_NC", ""))
    ap.add_argument("--pollution-model", type=str, default=os.environ.get("POLLUTION_MODEL", "ocpnet_3d"), choices=["gaussian", "ocpnet_3d"])
    ap.add_argument("--pollution-warmup-s", type=float, default=float(os.environ.get("POLLUTION_WARMUP_S", "20")))
    ap.add_argument("--pollution-update-period-s", type=float, default=float(os.environ.get("POLLUTION_UPDATE_PERIOD_S", "2")))
    ap.add_argument("--fps", type=int, default=int(os.environ.get("FPS", "20")))
    ap.add_argument("--render-quality", type=int, default=int(os.environ.get("RENDER_QUALITY", "3")))
    args = ap.parse_args()

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_py = (Path(__file__).resolve().parent / "run_plume_tasks.py").resolve()
    levels = [s.strip() for s in str(args.levels).split(",") if s.strip()]
    unknown = [s for s in levels if s not in _DEFAULT_LEVELS]
    if unknown:
        raise SystemExit(f"Unknown --levels entries: {unknown}. Expected subset of: {sorted(_DEFAULT_LEVELS.keys())}")

    dataset_variant = str(args.dataset_variant).strip() or None
    combined_nc = str(args.combined_nc).strip() or None

    env = os.environ.copy()
    env["POLLUTION_WARMUP_S"] = str(float(args.pollution_warmup_s))
    env["POLLUTION_UPDATE_PERIOD_S"] = str(float(args.pollution_update_period_s))

    results: dict[str, dict] = {}
    for i, lvl_name in enumerate(levels):
        level = _DEFAULT_LEVELS[lvl_name]
        child = out_dir / f"level_{level.name}_n{level.num_agents}_seed{int(args.seed) + i}"
        child.mkdir(parents=True, exist_ok=True)
        results[level.name] = _run_one(
            run_py=run_py,
            out_dir=child,
            scenario=str(args.scenario),
            dataset_variant=dataset_variant,
            combined_nc=combined_nc,
            time_index=int(args.time_index),
            depth_index=int(args.depth_index),
            seed=int(args.seed) + i,
            level=level,
            pollution_model=str(args.pollution_model),
            fps=int(args.fps),
            render_quality=int(args.render_quality),
            env=env,
        )

    manifest = {
        "created_at_utc": _utc_now(),
        "track": "h2_holoocean",
        "scenario": str(args.scenario),
        "dataset_variant": dataset_variant,
        "combined_nc_override": combined_nc,
        "time_index": int(args.time_index),
        "depth_index": int(args.depth_index),
        "pollution_model": str(args.pollution_model),
        "pollution_warmup_s": float(args.pollution_warmup_s),
        "pollution_update_period_s": float(args.pollution_update_period_s),
        "fps": int(args.fps),
        "render_quality": int(args.render_quality),
        "levels": levels,
        "child_runs": {k: v.get("metrics_json") for k, v in results.items()},
        "results": results,
        "command_hint": (
            "cd oneocean(iros-2026-code) && "
            f"{sys.executable} tracks/h2_holoocean/run_plume_ladder.py "
            f"--scenario {str(args.scenario)} --levels {str(args.levels)} --dataset-variant {dataset_variant or ''}".strip()
        ),
    }
    (out_dir / "results_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
