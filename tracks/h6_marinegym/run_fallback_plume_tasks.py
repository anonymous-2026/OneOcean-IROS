from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TaskCfg:
    num_agents: int = 10
    dt: float = 0.2
    steps_localize: int = 200
    steps_contain: int = 240
    fps: int = 20
    width: int = 1280
    height: int = 720
    world_span: float = 30.0  # maps to lon/lat bbox
    depth_z: float = -2.0
    # plume
    plume_sigma0: float = 1.4
    plume_sigma_growth: float = 0.004
    success_radius: float = 1.5
    contain_radius: float = 6.5


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


def _rgb_from_annotator(rgb_annotator):
    import numpy as np

    data = rgb_annotator.get_data()
    return np.frombuffer(data, dtype=np.uint8).reshape(*data.shape)[:, :, :3]


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


def _apply_underwater_postprocess(rgb_u8, distance_m):
    import numpy as np

    rgb = rgb_u8.astype(np.float32) / 255.0
    d = distance_m.astype(np.float32)
    if d.ndim == 3 and d.shape[-1] == 1:
        d = d[:, :, 0]

    d = np.nan_to_num(d, nan=60.0, posinf=60.0, neginf=0.0)
    d = np.clip(d, 0.0, 80.0)

    beta = 0.095
    t = np.exp(-beta * d)
    t3 = t[:, :, None]
    water = np.array([0.05, 0.20, 0.40], dtype=np.float32)
    rgb = rgb * t3 + water * (1.0 - t3)

    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, size=d.shape).astype(np.float32)
    noise = (noise - noise.min()) / max(1e-6, float(noise.ptp()))
    haze = (1.0 - t) ** 1.35
    rgb = np.clip(rgb + 0.07 * haze[:, :, None] * (noise[:, :, None] - 0.5), 0.0, 1.0)

    return (rgb * 255.0 + 0.5).astype(np.uint8)


def _plume_concentration(xy, center_xy, sigma: float):
    import numpy as np

    d = xy - center_xy
    r2 = (d * d).sum(axis=-1)
    return np.exp(-0.5 * r2 / max(1e-8, sigma * sigma))


def main() -> int:
    out_dir = Path(os.environ.get("OUT_DIR", "runs/h6_marinegym/fallback_plume_tasks")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TaskCfg(
        num_agents=int(os.environ.get("NUM_AGENTS", "10")),
        steps_localize=int(os.environ.get("STEPS_LOCALIZE", "200")),
        steps_contain=int(os.environ.get("STEPS_CONTAIN", "240")),
        fps=int(os.environ.get("FPS", "20")),
    )

    combined_nc = os.environ.get(
        "COMBINED_NC",
        "/data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/tiny/combined/combined_environment.nc",
    )

    # Isaac Sim requirement: instantiate SimulationApp before importing other omni.* modules.
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

    simulation_app = SimulationApp(simapp_cfg, experience=experience)
    try:
        import numpy as np
        import xarray as xr
        import imageio.v3 as iio

        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep

        from isaacsim.core.api import World
        from isaacsim.core.api.objects import VisualCuboid
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import set_stage_up_axis, set_stage_units
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import UsdGeom, Gf, UsdLux

        # Load currents from combined dataset (must-use-data policy).
        ds = xr.open_dataset(combined_nc)
        if "uo" not in ds or "vo" not in ds:
            raise RuntimeError(f"combined dataset missing uo/vo: {combined_nc}")
        uo = ds["uo"]
        vo = ds["vo"]
        lats = ds["latitude"].values
        lons = ds["longitude"].values
        depths = ds["depth"].values

        lon_min = float(lons.min())
        lon_max = float(lons.max())
        lat_min = float(lats.min())
        lat_max = float(lats.max())

        def current_uv(world_xy, world_z) -> np.ndarray:
            # world x/y -> lon/lat linearly over world_span.
            x, y = float(world_xy[0]), float(world_xy[1])
            lon = lon_min + (x / (2 * cfg.world_span) + 0.5) * (lon_max - lon_min)
            lat = lat_min + (y / (2 * cfg.world_span) + 0.5) * (lat_max - lat_min)
            depth = float(np.clip(-float(world_z), float(depths.min()), float(depths.max())))
            di = _nearest_idx(depths, depth)
            la = _nearest_idx(lats, lat)
            lo = _nearest_idx(lons, lon)
            uu = float(uo.isel(time=0, depth=di, latitude=la, longitude=lo).values)
            vv = float(vo.isel(time=0, depth=di, latitude=la, longitude=lo).values)
            return np.array([uu, vv], dtype=float)

        # Enable viewport + RTX extensions (best-effort).
        for ext in ("omni.kit.viewport.rtx", "omni.kit.viewport.pxr", "omni.kit.viewport.bundle", "omni.replicator.core"):
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

        world = World(backend="numpy", stage_units_in_meters=1.0)
        world.reset()

        stage = omni.usd.get_context().get_stage()

        # Lighting: bluish, soft.
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(650.0)
        dome.CreateColorAttr((0.35, 0.55, 1.0))
        sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
        sun.CreateIntensityAttr(2600.0)
        sun.CreateColorAttr((0.55, 0.75, 1.0))
        sun.AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 0.0))

        # Seafloor mesh (cheap “texture” via vertex color).
        grid_n = 90
        size = cfg.world_span
        xs = np.linspace(-size, size, grid_n)
        ys = np.linspace(-size, size, grid_n)
        xv, yv = np.meshgrid(xs, ys, indexing="xy")
        h = (
            0.6 * np.sin(0.18 * xv) * np.cos(0.16 * yv)
            + 0.25 * np.sin(0.42 * xv + 0.2) * np.sin(0.36 * yv - 0.4)
        )
        z = -5.0 + h
        points = np.stack([xv, yv, z], axis=-1).reshape(-1, 3)
        face_vertex_counts = []
        face_vertex_indices = []
        for j in range(grid_n - 1):
            for i in range(grid_n - 1):
                i0 = j * grid_n + i
                i1 = i0 + 1
                i2 = i0 + grid_n
                i3 = i2 + 1
                face_vertex_counts.extend([3, 3])
                face_vertex_indices.extend([i0, i1, i3, i0, i3, i2])
        mesh = UsdGeom.Mesh.Define(stage, "/World/SeaFloorMesh")
        mesh.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points])
        mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
        mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateSubdivisionSchemeAttr("none")
        c = np.clip((h - h.min()) / (h.ptp() + 1e-8), 0.0, 1.0)
        base = np.stack([0.18 + 0.30 * c, 0.16 + 0.26 * c, 0.12 + 0.18 * c], axis=-1).reshape(-1, 3)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(float(r), float(g), float(b)) for r, g, b in base])

        # Obstacles: rocks (used for collision metric)
        rng = np.random.default_rng(1)
        rocks = []
        for k in range(28):
            x = float(rng.uniform(-12, 12))
            y = float(rng.uniform(-12, 12))
            rz = float(rng.uniform(-4.6, -3.4))
            radius = float(rng.uniform(0.45, 1.25))
            sx = float(rng.uniform(0.8, 1.8))
            sy = float(rng.uniform(0.8, 1.8))
            sz = float(rng.uniform(0.6, 1.6))
            rock = UsdGeom.Sphere.Define(stage, f"/World/Rock_{k:02d}")
            rock.CreateRadiusAttr(radius)
            c = np.array([0.10, 0.12, 0.16]) + rng.uniform(0.0, 0.06, size=3)
            rock.CreateDisplayColorAttr([Gf.Vec3f(float(c[0]), float(c[1]), float(c[2]))])
            xf = UsdGeom.Xformable(rock)
            xf.AddTranslateOp().Set(Gf.Vec3f(x, y, rz))
            xf.AddScaleOp().Set(Gf.Vec3f(sx, sy, sz))
            rocks.append((np.array([x, y], dtype=float), float(radius * max(sx, sy))))

        # Agents
        agents = []
        agent_pos = []
        for i in range(cfg.num_agents):
            a = 2 * math.pi * i / max(1, cfg.num_agents)
            pos = np.array([7.5 * math.cos(a), 7.5 * math.sin(a), cfg.depth_z], dtype=float)
            agent = VisualCuboid(
                prim_path=f"/World/Agent_{i:02d}",
                name=f"Agent_{i:02d}",
                position=pos,
                scale=np.array([0.5, 0.22, 0.18]),
                color=np.array([0.95, 0.3, 0.1]) if i == 0 else np.array([0.2, 0.7, 0.9]),
            )
            agents.append(agent)
            agent_pos.append(pos)
        agent_pos = np.stack(agent_pos, axis=0)

        # Camera + replicator render product
        set_camera_view(eye=np.array([13.0, -13.0, 3.6]), target=np.array([0.0, 0.0, -2.6]))
        render_product = rep.create.render_product("/OmniverseKit_Persp", (cfg.width, cfg.height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb.attach([render_product])
        dist = rep.AnnotatorRegistry.get_annotator("distance_to_camera", device="cpu")
        dist.attach([render_product])

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        simulation_app.update()

        # ---- Task 1: plume localization ----
        localize_dir = out_dir / "task_plume_localize"
        localize_dir.mkdir(parents=True, exist_ok=True)

        plume_center = np.array([0.0, 0.0], dtype=float)
        est_center = None
        best_c = -1.0
        best_xy = None
        collisions = 0
        energy = 0.0
        success_step = None

        frames = []
        for t in range(cfg.steps_localize):
            theta = 2 * math.pi * (t / max(1, cfg.steps_localize))

            # Advect plume center using current at plume location.
            uvp = current_uv(plume_center, cfg.depth_z)
            plume_center = plume_center + cfg.dt * uvp

            sigma = cfg.plume_sigma0 + cfg.plume_sigma_growth * t

            # Simple multi-agent search: each agent runs a phased spiral; use max probe to estimate.
            for i in range(cfg.num_agents):
                phase = 2 * math.pi * i / max(1, cfg.num_agents)
                r = 2.0 + 0.06 * t
                desired_xy = np.array([r * math.cos(theta + phase), r * math.sin(theta + phase)], dtype=float)
                uv = current_uv(agent_pos[i, :2], agent_pos[i, 2])
                v_cmd = 0.12 * (desired_xy - agent_pos[i, :2]) + 0.18 * uv
                agent_pos[i, :2] = agent_pos[i, :2] + v_cmd
                agent_pos[i, 2] = cfg.depth_z
                energy += float((v_cmd * v_cmd).sum())

                # collision proxy
                for rock_xy, rock_r in rocks:
                    if float(np.linalg.norm(agent_pos[i, :2] - rock_xy)) < rock_r:
                        collisions += 1
                        break

                # probe concentration
                c_here = float(_plume_concentration(agent_pos[i, :2], plume_center, sigma))
                if c_here > best_c:
                    best_c = c_here
                    best_xy = agent_pos[i, :2].copy()

                agents[i].set_world_pose(position=agent_pos[i])

            # Estimate center as best probe location (baseline).
            if best_xy is not None:
                est_center = best_xy.copy()
                if success_step is None and float(np.linalg.norm(est_center - plume_center)) <= cfg.success_radius:
                    success_step = t

            # camera orbit
            eye = np.array([15.0 * math.cos(theta), 15.0 * math.sin(theta), 3.8])
            tgt = np.array([0.0, 0.0, -2.6])
            set_camera_view(eye=eye, target=tgt)

            world.step(render=True)
            simulation_app.update()
            if t % 2 == 0:
                im = _rgb_from_annotator(rgb)
                try:
                    dm = _float_image_from_annotator(dist)
                    im = _apply_underwater_postprocess(im, dm)
                except Exception:
                    pass
                frames.append(im)

        png_path = localize_dir / "frame_000.png"
        gif_path = localize_dir / "rollout.gif"
        iio.imwrite(png_path, frames[-1])
        iio.imwrite(gif_path, frames, duration=1.0 / cfg.fps, loop=0)

        metrics_localize = {
            "success": success_step is not None,
            "time_to_success_steps": success_step,
            "best_probe_concentration": best_c,
            "collision_count_proxy": collisions,
            "energy_proxy": float(energy),
            "num_agents": cfg.num_agents,
            "combined_nc": combined_nc,
            "combined_vars_used": ["uo", "vo"],
        }
        (localize_dir / "metrics.json").write_text(json.dumps(metrics_localize, indent=2), encoding="utf-8")

        # ---- Task 2: containment / cleanup (multi-agent, N=8/10 hero) ----
        contain_dir = out_dir / "task_plume_contain"
        contain_dir.mkdir(parents=True, exist_ok=True)

        # Reset agents to a ring.
        for i in range(cfg.num_agents):
            a = 2 * math.pi * i / max(1, cfg.num_agents)
            agent_pos[i] = np.array([cfg.contain_radius * math.cos(a), cfg.contain_radius * math.sin(a), cfg.depth_z], dtype=float)
            agents[i].set_world_pose(position=agent_pos[i])

        frames = []
        collisions = 0
        energy = 0.0
        leakage = 0.0
        for t in range(cfg.steps_contain):
            theta = 2 * math.pi * (t / max(1, cfg.steps_contain))

            # Advect plume center.
            uvp = current_uv(plume_center, cfg.depth_z)
            plume_center = plume_center + cfg.dt * uvp
            sigma = cfg.plume_sigma0 + cfg.plume_sigma_growth * (cfg.steps_localize + t)

            # Formation control: maintain ring around estimated center; current perturbs.
            for i in range(cfg.num_agents):
                a = 2 * math.pi * i / max(1, cfg.num_agents) + 0.25 * math.sin(theta)
                desired_xy = plume_center + cfg.contain_radius * np.array([math.cos(a), math.sin(a)], dtype=float)
                uv = current_uv(agent_pos[i, :2], agent_pos[i, 2])
                v_cmd = 0.16 * (desired_xy - agent_pos[i, :2]) + 0.22 * uv
                agent_pos[i, :2] = agent_pos[i, :2] + v_cmd
                agent_pos[i, 2] = cfg.depth_z
                energy += float((v_cmd * v_cmd).sum())

                for rock_xy, rock_r in rocks:
                    if float(np.linalg.norm(agent_pos[i, :2] - rock_xy)) < rock_r:
                        collisions += 1
                        break

                agents[i].set_world_pose(position=agent_pos[i])

            # Leakage proxy: average concentration outside ring radius.
            samples = rng.uniform(-10, 10, size=(256, 2))
            c = _plume_concentration(samples, plume_center, sigma)
            r = np.linalg.norm(samples - plume_center.reshape(1, 2), axis=1)
            leakage += float(c[r > cfg.contain_radius].mean())

            eye = np.array([15.0 * math.cos(theta + 0.7), 15.0 * math.sin(theta + 0.7), 3.8])
            tgt = np.array([float(plume_center[0]), float(plume_center[1]), -2.6])
            set_camera_view(eye=eye, target=tgt)

            world.step(render=True)
            simulation_app.update()
            if t % 2 == 0:
                im = _rgb_from_annotator(rgb)
                try:
                    dm = _float_image_from_annotator(dist)
                    im = _apply_underwater_postprocess(im, dm)
                except Exception:
                    pass
                frames.append(im)

        png_path = contain_dir / "frame_000.png"
        gif_path = contain_dir / "rollout.gif"
        iio.imwrite(png_path, frames[-1])
        iio.imwrite(gif_path, frames, duration=1.0 / cfg.fps, loop=0)

        metrics_contain = {
            "success": True,  # proxy task; always runs
            "leakage_proxy_mean": float(leakage / max(1, cfg.steps_contain)),
            "collision_count_proxy": collisions,
            "energy_proxy": float(energy),
            "num_agents": cfg.num_agents,
            "contain_radius": cfg.contain_radius,
            "combined_nc": combined_nc,
            "combined_vars_used": ["uo", "vo"],
        }
        (contain_dir / "metrics.json").write_text(json.dumps(metrics_contain, indent=2), encoding="utf-8")

        # manifests
        media_manifest = {
            "track": "H6_MarineGym_FallbackIsaacSim",
            "combined_nc": combined_nc,
            "combined_vars_used": ["uo", "vo"],
            "tasks": [
                {"name": "plume_localize", "png": str(localize_dir / "frame_000.png"), "gif": str(localize_dir / "rollout.gif")},
                {"name": "plume_contain", "png": str(contain_dir / "frame_000.png"), "gif": str(contain_dir / "rollout.gif")},
            ],
            "cmd_hint": "OUT_DIR=... /home/shuaijun/isaacsim/python.sh tracks/h6_marinegym/run_fallback_plume_tasks.py",
        }
        (out_dir / "media_manifest.json").write_text(json.dumps(media_manifest, indent=2), encoding="utf-8")

        results_manifest = {
            "track": "H6_MarineGym_FallbackIsaacSim",
            "results": [
                {"task": "plume_localize", "metrics_json": str(localize_dir / "metrics.json")},
                {"task": "plume_contain", "metrics_json": str(contain_dir / "metrics.json")},
            ],
        }
        (out_dir / "results_manifest.json").write_text(json.dumps(results_manifest, indent=2), encoding="utf-8")

        # Print key artifacts for quick verification.
        print(str(localize_dir / "rollout.gif"))
        print(str(contain_dir / "rollout.gif"))
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
