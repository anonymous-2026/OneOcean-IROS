from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from ..fields import AdvectedGaussianPlume, DatasetCurrentField, FieldMapping
from ..media import Mp4Writer
from ..rpc import AirSimRpc, RpcAddress, make_pose_xyz_yaw, pose_to_xyz_yaw


def _vehicle_names(base: str, n: int) -> List[str]:
    return [f"{base}{i:02d}" for i in range(n)]


def _clip_norm(vx: float, vy: float, vmax: float) -> Tuple[float, float]:
    s = math.hypot(vx, vy)
    if s <= vmax:
        return vx, vy
    if s < 1e-9:
        return 0.0, 0.0
    return vx / s * vmax, vy / s * vmax


def _write_json(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="H4 UNav-Sim plume tasks (external physics + dataset-driven drift).")
    parser.add_argument("--ip", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=41451)
    parser.add_argument("--timeout-s", type=float, default=60.0)
    parser.add_argument("--vehicle-base", default="Rov")
    parser.add_argument("--vehicle-count", type=int, default=8)
    parser.add_argument("--record-vehicle", default="Rov00")
    parser.add_argument("--camera-name", default="front_right_custom")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--steps", type=int, default=320)
    parser.add_argument("--dt", type=float, default=0.10)
    parser.add_argument("--speed-mps", type=float, default=1.2)
    parser.add_argument("--current-gain", type=float, default=1.0)
    parser.add_argument("--ignore-collision", action="store_true")

    parser.add_argument(
        "--dataset-path",
        default="/data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc",
    )
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--depth-index", type=int, default=0)
    parser.add_argument("--domain-width-m", type=float, default=120.0)
    parser.add_argument("--domain-height-m", type=float, default=120.0)
    parser.add_argument("--origin-x", type=float, default=0.0)
    parser.add_argument("--origin-y", type=float, default=0.0)

    parser.add_argument("--plume-x", type=float, default=60.0)
    parser.add_argument("--plume-y", type=float, default=60.0)
    parser.add_argument("--plume-sigma-m", type=float, default=8.0)
    parser.add_argument("--plume-advect-gain", type=float, default=1.0)

    parser.add_argument("--output-dir", default="runs/h4_unavsim_plume_hero_v1")
    return parser


def _task_localize(
    poses: Dict[str, Tuple[float, float, float, float]],
    plume: AdvectedGaussianPlume,
) -> Tuple[float, float]:
    # Global best-probe greedy target.
    best_v = -1.0
    best_xy = plume.center_xy  # fallback: true center (used only if probes fail)
    for x, y, *_ in poses.values():
        v = plume.concentration(x, y)
        if v > best_v:
            best_v = v
            best_xy = (x, y)
    return best_xy


def _task_contain_targets(center_xy: Tuple[float, float], names: List[str], radius_m: float = 18.0) -> Dict[str, Tuple[float, float]]:
    cx, cy = center_xy
    out: Dict[str, Tuple[float, float]] = {}
    for idx, name in enumerate(names):
        ang = 2.0 * math.pi * (idx / max(1, len(names)))
        out[name] = (cx + radius_m * math.cos(ang), cy + radius_m * math.sin(ang))
    return out


def main() -> int:
    args = build_parser().parse_args()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    address = RpcAddress(ip=args.ip, port=args.port)
    rpc = AirSimRpc(address=address, timeout_s=args.timeout_s)
    rpc.confirm_connection()

    names = _vehicle_names(args.vehicle_base, args.vehicle_count)
    for name in names:
        rpc.enable_api_control(True, name)
        rpc.arm_disarm(True, name)

    mapping = FieldMapping(width_m=args.domain_width_m, height_m=args.domain_height_m, origin_x=args.origin_x, origin_y=args.origin_y)
    currents = DatasetCurrentField(
        nc_path=args.dataset_path,
        mapping=mapping,
        time_index=args.time_index,
        depth_index=args.depth_index,
    )
    plume = AdvectedGaussianPlume(
        current_field=currents,
        center_xy=(args.plume_x, args.plume_y),
        sigma_m=args.plume_sigma_m,
        advect_gain=args.plume_advect_gain,
    )

    run_config = {
        "track": "H4_UNAV_SIM",
        "address": {"ip": args.ip, "port": args.port},
        "vehicles": names,
        "record_vehicle": args.record_vehicle,
        "camera_name": args.camera_name,
        "dataset_path": args.dataset_path,
        "time_index": args.time_index,
        "depth_index": args.depth_index,
        "mapping": mapping.__dict__,
        "plume": {"x": args.plume_x, "y": args.plume_y, "sigma_m": args.plume_sigma_m, "advect_gain": args.plume_advect_gain},
        "steps": args.steps,
        "dt": args.dt,
        "speed_mps": args.speed_mps,
        "current_gain": args.current_gain,
        "ignore_collision": bool(args.ignore_collision),
    }
    _write_json(out_dir / "run_config.json", run_config)

    # Initialize poses from sim.
    poses: Dict[str, Tuple[float, float, float, float]] = {}
    for name in names:
        pose = rpc.sim_get_vehicle_pose(name)
        poses[name] = pose_to_xyz_yaw(pose)

    video_path = out_dir / "rollout.mp4"
    traj_rows = []
    energy = 0.0
    collision_count = 0

    with Mp4Writer(video_path, fps=args.fps) as writer:
        for step in range(args.steps):
            t = step * args.dt
            plume.step(args.dt)
            plume_center = plume.center_xy

            # Task policy switches half-way: localization then containment.
            if step < args.steps // 2:
                target_xy = _task_localize(poses, plume)
                per_agent_targets = {name: target_xy for name in names}
            else:
                per_agent_targets = _task_contain_targets(plume_center, names, radius_m=18.0)

            for name in names:
                x, y, z, yaw = poses[name]
                tx, ty = per_agent_targets[name]
                vx = (tx - x) * 0.6
                vy = (ty - y) * 0.6
                vx, vy = _clip_norm(vx, vy, args.speed_mps)

                u, v = currents.sample_uv(x, y)
                vx_total = vx + args.current_gain * u
                vy_total = vy + args.current_gain * v

                x2 = x + vx_total * args.dt
                y2 = y + vy_total * args.dt
                z2 = z
                yaw2 = math.atan2(vy_total, vx_total) if (abs(vx_total) + abs(vy_total)) > 1e-6 else yaw

                poses[name] = (x2, y2, z2, yaw2)
                rpc.sim_set_vehicle_pose(make_pose_xyz_yaw(x2, y2, z2, yaw2), ignore_collision=args.ignore_collision, vehicle_name=name)

                energy += float(vx * vx + vy * vy) * args.dt
                if rpc.sim_get_collision_info(name).get("has_collided"):
                    collision_count += 1

                traj_rows.append(
                    {
                        "step": step,
                        "t": t,
                        "vehicle": name,
                        "x": x2,
                        "y": y2,
                        "z": z2,
                        "yaw": yaw2,
                        "u": u,
                        "v": v,
                        "plume_c": plume.concentration(x2, y2),
                        "plume_cx": plume_center[0],
                        "plume_cy": plume_center[1],
                    }
                )

            # Capture one camera stream for media.
            try:
                png = rpc.sim_get_image(args.camera_name, 0, args.record_vehicle, False)
                if png:
                    writer.append_png_bytes(png)
            except Exception:
                # Do not crash the whole run if image capture is temporarily unavailable.
                pass

            time.sleep(max(0.0, args.dt * 0.25))

    # Metrics
    final_center = plume.center_xy
    dists = [math.hypot(poses[n][0] - final_center[0], poses[n][1] - final_center[1]) for n in names]
    mean_dist = float(np.mean(dists))
    success = mean_dist < 10.0
    metrics = {
        "success": bool(success),
        "mean_distance_to_plume_center_m": mean_dist,
        "energy_proxy": float(energy),
        "collision_count": int(collision_count),
        "vehicles": int(len(names)),
    }
    _write_json(out_dir / "metrics.json", metrics)

    # Minimal manifests (expanded by later stages once we have real UE media and finalized tasks).
    _write_json(
        out_dir / "media_manifest.json",
        {
            "rollout_mp4": "rollout.mp4",
            "camera_name": args.camera_name,
            "record_vehicle": args.record_vehicle,
            "fps": int(args.fps),
        },
    )
    _write_json(out_dir / "results_manifest.json", {"metrics": metrics, "run_config": "run_config.json"})

    # Trajectory log
    (out_dir / "trajectories.jsonl").write_text("\n".join(json.dumps(r) for r in traj_rows) + "\n", encoding="utf-8")
    currents.close()
    print(json.dumps({"output_dir": str(out_dir), "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

