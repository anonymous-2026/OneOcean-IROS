from __future__ import annotations

import importlib.util
import json
import math
import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskCfg:
    num_agents: int = 10
    dt: float = 0.02
    steps_localize: int = 220
    steps_contain: int = 260
    fps: int = 20
    width: int = 1280
    height: int = 720
    world_span: float = 30.0
    z: float = 2.0
    plume_sigma0: float = 1.8
    plume_sigma_growth: float = 0.004
    success_radius: float = 2.0
    contain_radius: float = 7.0
    capture_stride: int = 3
    seabed_height_span: float = 6.0


def _marinegym_src_from_env() -> Path:
    p = os.environ.get("MARINEGYM_SRC")
    if p:
        return Path(p).expanduser().resolve()
    return Path("runs/_cache/external_scenes/marinegym/MarineGym-main").expanduser().resolve()


def _load_patcher():
    patcher_path = Path(__file__).with_name("patch_marinegym_for_isaacsim51.py")
    spec = importlib.util.spec_from_file_location("_mg_patch", patcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import patcher: {patcher_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _nearest_idx(arr, val: float) -> int:
    import numpy as np

    if arr.size == 1:
        return 0
    i = int(np.searchsorted(arr, val))
    if i <= 0:
        return 0
    if i >= arr.size:
        return int(arr.size - 1)
    return i - 1 if abs(val - float(arr[i - 1])) <= abs(float(arr[i]) - val) else i


def _float_image_from_annotator(float_annotator):
    import numpy as np

    data = float_annotator.get_data()
    try:
        arr = np.frombuffer(data, dtype=np.float32).reshape(*data.shape)
        return arr
    except Exception:
        arr = np.asarray(data)
        if hasattr(data, "shape") and tuple(arr.shape) != tuple(data.shape):
            try:
                arr = arr.reshape(*data.shape)
            except Exception:
                pass
        return arr.astype(np.float32, copy=False)


def _rgb_from_annotator(rgb_annotator):
    import numpy as np

    data = rgb_annotator.get_data()
    arr = np.asarray(data)
    if getattr(arr, "ndim", 0) == 3 and arr.shape[-1] >= 3:
        return arr[:, :, :3].astype(np.uint8, copy=False)
    return np.frombuffer(data, dtype=np.uint8)


def _apply_underwater_postprocess(rgb_u8, distance_m):
    import numpy as np

    rgb = rgb_u8.astype(np.float32) / 255.0
    d = distance_m.astype(np.float32)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[:, :, 0]
    d = np.nan_to_num(d, nan=60.0, posinf=60.0, neginf=0.0)
    d = np.clip(d, 0.0, 80.0)

    beta = 0.090
    t = np.exp(-beta * d)
    t3 = t[:, :, None]
    water = np.array([0.05, 0.20, 0.40], dtype=np.float32)
    rgb = rgb * t3 + water * (1.0 - t3)

    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, size=d.shape).astype(np.float32)
    noise = (noise - noise.min()) / max(1e-6, float(noise.ptp()))
    haze = (1.0 - t) ** 1.35
    rgb = np.clip(rgb + 0.06 * haze[:, :, None] * (noise[:, :, None] - 0.5), 0.0, 1.0)

    return (rgb * 255.0 + 0.5).astype(np.uint8)


def _plume_concentration(xy, center_xy, sigma: float):
    import numpy as np

    d = xy - center_xy
    r2 = (d * d).sum(axis=-1)
    return np.exp(-0.5 * r2 / max(1e-8, sigma * sigma))


def _maybe_add_bathymetry_seafloor(stage, ds, *, world_span: float, height_span: float) -> tuple[str | None, str | None]:
    import numpy as np

    var_name = None
    for cand in ("elevation", "bathymetry", "depth_bathymetry", "topography", "topo"):
        if cand in ds:
            var_name = cand
            break
    if var_name is None:
        return None, None

    arr = ds[var_name]
    if "time" in arr.dims:
        arr = arr.isel(time=0)
    if "depth" in arr.dims:
        arr = arr.isel(depth=0)
    vals = np.asarray(arr.values)
    if vals.ndim != 2:
        return None, None

    n0, n1 = int(vals.shape[0]), int(vals.shape[1])
    stride = max(1, int(max(n0, n1) / 160))
    vals = vals[::stride, ::stride].astype(np.float32, copy=False)
    nlat, nlon = int(vals.shape[0]), int(vals.shape[1])
    if nlat < 2 or nlon < 2:
        return None, None

    depth = -vals if float(np.nanmean(vals)) < 0.0 else vals
    depth = np.nan_to_num(depth, nan=float(np.nanmedian(depth)))
    p10, p90 = np.percentile(depth, [10, 90]).astype(np.float32)
    denom = float(max(1e-6, p90 - p10))
    dnorm = np.clip((depth - float(p10)) / denom, 0.0, 1.0)
    z = -0.8 - float(height_span) * dnorm

    xs = np.linspace(-world_span, world_span, nlon, dtype=np.float32)
    ys = np.linspace(-world_span, world_span, nlat, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    points = np.stack([xx, yy, z], axis=-1).reshape(-1, 3)
    indices = []
    counts = []
    for i in range(nlat - 1):
        for j in range(nlon - 1):
            v0 = i * nlon + j
            v1 = v0 + 1
            v2 = v0 + nlon + 1
            v3 = v0 + nlon
            indices.extend([v0, v1, v2, v0, v2, v3])
            counts.extend([3, 3])

    from pxr import Gf, UsdGeom, UsdPhysics

    path = "/World/Seafloor"
    mesh = UsdGeom.Mesh.Define(stage, path)
    mesh.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points])
    mesh.CreateFaceVertexCountsAttr(counts)
    mesh.CreateFaceVertexIndicesAttr(indices)
    mesh.CreateDoubleSidedAttr(True)
    mesh.CreateDisplayColorAttr([Gf.Vec3f(0.36, 0.35, 0.30)])
    try:
        UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    except Exception:
        pass
    return path, var_name


def main() -> int:
    out_dir = Path(os.environ.get("OUT_DIR", "runs/h6_marinegym/marinegym_plume_tasks")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TaskCfg(
        num_agents=int(os.environ.get("NUM_AGENTS", "10")),
        steps_localize=int(os.environ.get("STEPS_LOCALIZE", "220")),
        steps_contain=int(os.environ.get("STEPS_CONTAIN", "260")),
        fps=int(os.environ.get("FPS", "20")),
    )

    marinegym_src = _marinegym_src_from_env()
    if not marinegym_src.exists():
        raise SystemExit(
            f"Missing MarineGym source at {marinegym_src}. "
            "Run: python3 tracks/h6_marinegym/fetch_marinegym_source.py"
        )

    patcher = _load_patcher()
    patcher.patch_marinegym(marinegym_src)
    sys.path.insert(0, str(marinegym_src))

    combined_nc = os.environ.get(
        "COMBINED_NC",
        "/data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/tiny/combined/combined_environment.nc",
    )

    from isaacsim import SimulationApp

    experience = "/home/shuaijun/isaacsim/apps/isaacsim.exp.base.python.kit"
    simapp_cfg = {
        "headless": True,
        "width": cfg.width,
        "height": cfg.height,
        "renderer": "RayTracedLighting",
        "multi_gpu": False,
        "max_gpu_count": 1,
        "extra_args": [
            "--/validate/p2p/enabled=false",
            "--/validate/p2p/memoryCheck/enabled=false",
            "--/validate/iommu/enabled=false",
            "--/validate/wait=0",
        ],
    }

    (out_dir / "run_meta.json").write_text(
        json.dumps(
            {
                "track": "H6_MarineGym",
                "out_dir": str(out_dir),
                "marinegym_src": str(marinegym_src),
                "combined_nc": combined_nc,
                "num_agents": cfg.num_agents,
                "steps_localize": cfg.steps_localize,
                "steps_contain": cfg.steps_contain,
                "simapp_cfg": simapp_cfg,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    simulation_app = SimulationApp(simapp_cfg, experience=experience)
    try:
        import numpy as np
        import torch
        import xarray as xr
        import imageio.v3 as iio

        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep

        from omni.isaac.core.simulation_context import SimulationContext
        from omni.isaac.core.utils.extensions import enable_extension
        from omni.isaac.core.utils.stage import set_stage_up_axis, set_stage_units
        from omni.isaac.core.utils.viewports import set_camera_view

        from marinegym.robots.drone import UnderwaterVehicle
        from marinegym.robots.robot import RobotBase
        from marinegym.utils.torch import quat_axis

        for ext in (
            "omni.kit.window.toolbar",
            "omni.kit.viewport.rtx",
            "omni.kit.viewport.pxr",
            "omni.kit.viewport.bundle",
            "omni.kit.window.status_bar",
            "omni.replicator.isaac",
        ):
            try:
                enable_extension(ext)
            except Exception:
                pass

        simulation_app.update()
        omni.usd.get_context().new_stage()
        simulation_app.update()
        try:
            set_stage_up_axis("z")
        except Exception:
            pass
        try:
            set_stage_units(1.0)
        except Exception:
            pass
        stage = omni.usd.get_context().get_stage()

        ds = xr.open_dataset(combined_nc)
        if "uo" not in ds or "vo" not in ds:
            raise RuntimeError(f"combined dataset missing uo/vo: {combined_nc}")
        uo = ds["uo"]
        vo = ds["vo"]
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        depths = ds["depth"].values
        lon_min, lon_max = float(lons.min()), float(lons.max())
        lat_min, lat_max = float(lats.min()), float(lats.max())

        seafloor_path, seafloor_var = _maybe_add_bathymetry_seafloor(stage, ds, world_span=cfg.world_span, height_span=cfg.seabed_height_span)

        def current_uv(world_xy, depth_m: float) -> np.ndarray:
            x, y = float(world_xy[0]), float(world_xy[1])
            lon = lon_min + (x / (2 * cfg.world_span) + 0.5) * (lon_max - lon_min)
            lat = lat_min + (y / (2 * cfg.world_span) + 0.5) * (lat_max - lat_min)
            di = _nearest_idx(depths, float(np.clip(depth_m, float(depths.min()), float(depths.max()))))
            la = _nearest_idx(lats, lat)
            lo = _nearest_idx(lons, lon)
            uu = float(uo.isel(time=0, depth=di, latitude=la, longitude=lo).values)
            vv = float(vo.isel(time=0, depth=di, latitude=la, longitude=lo).values)
            return np.array([uu, vv], dtype=float)

        sim = SimulationContext(
            stage_units_in_meters=1.0,
            physics_dt=cfg.dt,
            rendering_dt=cfg.dt,
            backend="torch",
            sim_params={
                "dt": cfg.dt,
                "substeps": 1,
                "gravity": [0, 0, -9.81],
                "replicate_physics": False,
                "use_flatcache": True,
                "use_gpu_pipeline": True,
                "device": "cuda:0",
                "solver_type": 1,
                "use_gpu": True,
                "enable_stabilization": True,
                "enable_scene_query_support": True,
            },
            physics_prim_path="/physicsScene",
            device="cuda:0",
        )

        drone, _ = UnderwaterVehicle.make("BlueROV")
        translations = []
        for i in range(cfg.num_agents):
            a = 2 * math.pi * i / max(1, cfg.num_agents)
            translations.append((7.0 * math.cos(a), 7.0 * math.sin(a), cfg.z))
        drone.spawn(translations=translations)
        sim.reset()
        RobotBase._envs_positions = torch.zeros(1, 1, 3, device=sim._device)
        drone.initialize()

        rotor_cfg = drone.params.get("rotor_configuration", {})
        fc = torch.tensor(rotor_cfg.get("force_constants", [4.4e-7] * int(rotor_cfg.get("num_rotors", 6))), device=sim._device)
        rpm_max = torch.tensor(rotor_cfg.get("max_rotation_velocities", [3900.0] * int(rotor_cfg.get("num_rotors", 6))), device=sim._device).clamp(min=1.0)
        thrust_max = torch.clamp((fc / 4.4e-7) * 9.81 * (4.7368e-07 * rpm_max.square() - 1.9275e-04 * rpm_max + 8.4452e-02), min=1.0)

        kp = torch.tensor([3.0, 3.0, 6.0], device=sim._device)
        kd = torch.tensor([3.5, 3.5, 5.0], device=sim._device)
        kd_ang = torch.tensor([0.6, 0.6, 0.6], device=sim._device)
        f_clip = torch.tensor([65.0, 65.0, 85.0], device=sim._device)

        def compute_thruster_cmds(target_pos_world: torch.Tensor) -> torch.Tensor:
            pos_w, _ = drone.get_world_poses(clone=True)
            vel6_w = drone.get_velocities(clone=True)
            lin_v = vel6_w[..., :3]
            ang_v = vel6_w[..., 3:]
            err = target_pos_world - pos_w
            f_w = torch.clamp(kp * err - kd * lin_v, -f_clip, f_clip)
            tau_w = -kd_ang * ang_v

            rotor_pos_w, rotor_rot_w = drone.rotors_view.get_world_poses()
            r_w = rotor_pos_w - pos_w.unsqueeze(-2)
            axis_w = quat_axis(rotor_rot_w.reshape(-1, 4), axis=0).reshape(*rotor_rot_w.shape[:-1], 3)
            tau_per = torch.linalg.cross(r_w, axis_w)
            A = torch.cat([axis_w, tau_per], dim=-1).transpose(-1, -2)
            w = torch.cat([f_w, tau_w], dim=-1).unsqueeze(-1)
            thrusts = (torch.linalg.pinv(A) @ w).squeeze(-1)
            return torch.clamp(thrusts / thrust_max.view(1, 1, -1), -1.0, 1.0)

        set_camera_view(eye=np.array([11.0, -11.0, 6.0]), target=np.array([0.0, 0.0, 2.0]))
        render_product = rep.create.render_product("/OmniverseKit_Persp", (cfg.width, cfg.height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        dist = rep.AnnotatorRegistry.get_annotator("distance_to_camera", device="cpu")
        rgb.attach([render_product])
        dist.attach([render_product])

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        simulation_app.update()

        plume_center = np.array([0.0, 0.0], dtype=float)
        best_c = -1.0
        best_xy = None
        est_center = None

        localize_dir = out_dir / "task_plume_localize"
        localize_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        collisions = 0
        energy = 0.0
        success_step = None

        for t in range(cfg.steps_localize):
            theta = 2 * math.pi * (t / max(1, cfg.steps_localize))
            plume_center = plume_center + cfg.dt * current_uv(plume_center, depth_m=10.0)
            sigma = cfg.plume_sigma0 + cfg.plume_sigma_growth * t

            r = 2.0 + 0.05 * t
            desired = []
            for i in range(cfg.num_agents):
                phase = 2 * math.pi * i / max(1, cfg.num_agents)
                desired.append([r * math.cos(theta + phase), r * math.sin(theta + phase), cfg.z])
            desired_t = torch.tensor(desired, device=sim._device, dtype=torch.float32).unsqueeze(0)

            pos, _ = drone.get_world_poses(clone=True)
            pos_np = pos[0].detach().cpu().numpy()
            flow = np.zeros((cfg.num_agents, 6), dtype=np.float32)
            for i in range(cfg.num_agents):
                uv = current_uv(pos_np[i, :2], depth_m=10.0)
                flow[i, 0] = float(uv[0])
                flow[i, 1] = float(uv[1])
            drone.flow_vels[0, :, :] = torch.tensor(flow, device=sim._device)

            cmds = compute_thruster_cmds(desired_t)
            energy += float((cmds * cmds).sum().detach().cpu().item())
            drone.apply_action(cmds)

            c_here = _plume_concentration(pos_np[:, :2], plume_center, sigma)
            top_i = int(np.argmax(c_here))
            if float(c_here[top_i]) > best_c:
                best_c = float(c_here[top_i])
                best_xy = pos_np[top_i, :2].copy()
            if t > 20 and best_xy is not None:
                est_center = best_xy
                if success_step is None and float(np.linalg.norm(est_center - plume_center)) <= cfg.success_radius:
                    success_step = int(t)

            for i in range(cfg.num_agents):
                for j in range(i + 1, cfg.num_agents):
                    if float(np.linalg.norm(pos_np[i, :2] - pos_np[j, :2])) < 0.35:
                        collisions += 1
                        break

            set_camera_view(eye=np.array([12.0 * math.cos(theta + 0.6), 12.0 * math.sin(theta + 0.6), 6.0]), target=np.array([0.0, 0.0, cfg.z]))
            sim.step(render=True)
            simulation_app.update()
            if t % cfg.capture_stride == 0:
                im = _rgb_from_annotator(rgb)
                dm = _float_image_from_annotator(dist)
                if getattr(im, "ndim", 0) == 3 and im.shape[-1] >= 3:
                    frames.append(_apply_underwater_postprocess(im[:, :, :3], dm))

        iio.imwrite(localize_dir / "frame_000.png", frames[-1])
        iio.imwrite(localize_dir / "rollout.gif", frames, duration=1.0 / cfg.fps, loop=0)
        (localize_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "success": success_step is not None,
                    "time_to_success_steps": success_step,
                    "best_probe_concentration": best_c,
                    "collision_count_proxy": collisions,
                    "energy_proxy": float(energy),
                    "num_agents": cfg.num_agents,
                    "combined_nc": combined_nc,
                    "combined_vars_used": ["uo", "vo"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        contain_dir = out_dir / "task_plume_contain"
        contain_dir.mkdir(parents=True, exist_ok=True)
        center = plume_center if est_center is None else est_center
        frames = []
        collisions = 0
        energy = 0.0
        leakage = 0.0
        rng = np.random.default_rng(1)

        for t in range(cfg.steps_contain):
            theta = 2 * math.pi * (t / max(1, cfg.steps_contain))
            plume_center = plume_center + cfg.dt * current_uv(plume_center, depth_m=10.0)
            sigma = cfg.plume_sigma0 + cfg.plume_sigma_growth * (cfg.steps_localize + t)

            desired = []
            for i in range(cfg.num_agents):
                phase = 2 * math.pi * i / max(1, cfg.num_agents) + 0.25 * math.sin(theta)
                desired.append([float(center[0]) + cfg.contain_radius * math.cos(phase), float(center[1]) + cfg.contain_radius * math.sin(phase), cfg.z])
            desired_t = torch.tensor(desired, device=sim._device, dtype=torch.float32).unsqueeze(0)

            pos, _ = drone.get_world_poses(clone=True)
            pos_np = pos[0].detach().cpu().numpy()
            flow = np.zeros((cfg.num_agents, 6), dtype=np.float32)
            for i in range(cfg.num_agents):
                uv = current_uv(pos_np[i, :2], depth_m=10.0)
                flow[i, 0] = float(uv[0])
                flow[i, 1] = float(uv[1])
            drone.flow_vels[0, :, :] = torch.tensor(flow, device=sim._device)

            cmds = compute_thruster_cmds(desired_t)
            energy += float((cmds * cmds).sum().detach().cpu().item())
            drone.apply_action(cmds)

            samples = rng.uniform(-10, 10, size=(256, 2))
            c = _plume_concentration(samples, plume_center, sigma)
            r = np.linalg.norm(samples - plume_center.reshape(1, 2), axis=1)
            leakage += float(c[r > cfg.contain_radius].mean())

            for i in range(cfg.num_agents):
                for j in range(i + 1, cfg.num_agents):
                    if float(np.linalg.norm(pos_np[i, :2] - pos_np[j, :2])) < 0.35:
                        collisions += 1
                        break

            set_camera_view(eye=np.array([12.0 * math.cos(theta + 0.8), 12.0 * math.sin(theta + 0.8), 6.0]), target=np.array([float(center[0]), float(center[1]), cfg.z]))
            sim.step(render=True)
            simulation_app.update()
            if t % cfg.capture_stride == 0:
                im = _rgb_from_annotator(rgb)
                dm = _float_image_from_annotator(dist)
                if getattr(im, "ndim", 0) == 3 and im.shape[-1] >= 3:
                    frames.append(_apply_underwater_postprocess(im[:, :, :3], dm))

        iio.imwrite(contain_dir / "frame_000.png", frames[-1])
        iio.imwrite(contain_dir / "rollout.gif", frames, duration=1.0 / cfg.fps, loop=0)
        (contain_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "success": True,
                    "leakage_proxy_mean": float(leakage / max(1, cfg.steps_contain)),
                    "collision_count_proxy": collisions,
                    "energy_proxy": float(energy),
                    "num_agents": cfg.num_agents,
                    "contain_radius": cfg.contain_radius,
                    "combined_nc": combined_nc,
                    "combined_vars_used": ["uo", "vo"],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        (out_dir / "media_manifest.json").write_text(
            json.dumps(
                {
                    "track": "H6_MarineGym",
                    "marinegym_src": str(marinegym_src),
                    "combined_nc": combined_nc,
                    "combined_vars_used": ["uo", "vo"],
                    "seafloor_prim": seafloor_path,
                    "seafloor_var": seafloor_var,
                    "tasks": [
                        {"name": "plume_localize", "png": str(localize_dir / "frame_000.png"), "gif": str(localize_dir / "rollout.gif")},
                        {"name": "plume_contain", "png": str(contain_dir / "frame_000.png"), "gif": str(contain_dir / "rollout.gif")},
                    ],
                    "cmd": "OUT_DIR=... NUM_AGENTS=10 /home/shuaijun/isaacsim/python.sh tracks/h6_marinegym/run_marinegym_plume_tasks.py",
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        (out_dir / "results_manifest.json").write_text(
            json.dumps(
                {
                    "track": "H6_MarineGym",
                    "results": [
                        {"task": "plume_localize", "metrics_json": str(localize_dir / "metrics.json")},
                        {"task": "plume_contain", "metrics_json": str(contain_dir / "metrics.json")},
                    ],
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(str(localize_dir / "rollout.gif"))
        print(str(contain_dir / "rollout.gif"))
        return 0

    except Exception:
        (out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())

