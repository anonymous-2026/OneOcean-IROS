from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SceneCfg:
    num_agents: int = 10
    frames: int = 240
    fps: int = 20
    width: int = 1280
    height: int = 720
    radius: float = 12.0


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

    # Replace non-finite values with a far distance.
    d = np.nan_to_num(d, nan=60.0, posinf=60.0, neginf=0.0)
    d = np.clip(d, 0.0, 80.0)

    # Beer-Lambert-ish attenuation + haze tint (cheap but effective).
    beta = 0.085  # higher = murkier
    t = np.exp(-beta * d)
    t3 = t[:, :, None]
    water = np.array([0.06, 0.22, 0.42], dtype=np.float32)
    rgb = rgb * t3 + water * (1.0 - t3)

    # Add mild suspended particulate haze (screen-space noise scaled by distance).
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, 1.0, size=d.shape).astype(np.float32)
    noise = (noise - noise.min()) / max(1e-6, float(noise.ptp()))
    haze = (1.0 - t) ** 1.4
    rgb = np.clip(rgb + 0.06 * haze[:, :, None] * (noise[:, :, None] - 0.5), 0.0, 1.0)

    return (rgb * 255.0 + 0.5).astype(np.uint8)


def main() -> int:
    out_dir = Path(os.environ.get("OUT_DIR", "runs/h6_marinegym/fallback_scene")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = SceneCfg(
        num_agents=int(os.environ.get("NUM_AGENTS", "10")),
        frames=int(os.environ.get("FRAMES", "240")),
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
        # This host hangs at startup unless we disable these validations.
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
        import imageio.v3 as iio
        import xarray as xr

        import omni.usd
        import omni.timeline
        import omni.replicator.core as rep

        from isaacsim.core.api import World
        from isaacsim.core.api.objects import VisualCuboid
        from isaacsim.core.utils.extensions import enable_extension
        from isaacsim.core.utils.stage import set_stage_up_axis, set_stage_units
        from isaacsim.core.utils.viewports import set_camera_view
        from pxr import UsdGeom, Gf, UsdLux

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

        # Optional: load our combined dataset for current-driven drift.
        ds = None
        uo = vo = None
        lats = lons = depths = times = None
        try:
            if combined_nc and Path(combined_nc).exists():
                ds = xr.open_dataset(combined_nc)
                if "uo" in ds and "vo" in ds:
                    uo = ds["uo"]
                    vo = ds["vo"]
                    lats = ds["latitude"].values
                    lons = ds["longitude"].values
                    depths = ds["depth"].values
                    times = ds["time"].values
        except Exception:
            ds = None

        def _nearest_idx(arr: np.ndarray, val: float) -> int:
            if arr.size == 1:
                return 0
            i = int(np.searchsorted(arr, val))
            if i <= 0:
                return 0
            if i >= arr.size:
                return int(arr.size - 1)
            return i - 1 if abs(val - float(arr[i - 1])) <= abs(float(arr[i]) - val) else i

        # Lighting: bluish, soft.
        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(600.0)
        dome.CreateColorAttr((0.4, 0.6, 1.0))
        sun = UsdLux.DistantLight.Define(stage, "/World/SunLight")
        sun.CreateIntensityAttr(2500.0)
        sun.CreateColorAttr((0.6, 0.8, 1.0))
        sun.AddRotateXYZOp().Set(Gf.Vec3f(315.0, 0.0, 0.0))

        # Seafloor: a procedurally-displaced mesh with per-vertex color (cheap “texture”).
        grid_n = 90
        size = 30.0
        xs = np.linspace(-size, size, grid_n)
        ys = np.linspace(-size, size, grid_n)
        xv, yv = np.meshgrid(xs, ys, indexing="xy")
        # pseudo-noise height
        h = (
            0.6 * np.sin(0.18 * xv) * np.cos(0.16 * yv)
            + 0.25 * np.sin(0.42 * xv + 0.2) * np.sin(0.36 * yv - 0.4)
        )
        z = -5.0 + h

        points = np.stack([xv, yv, z], axis=-1).reshape(-1, 3)
        # quads -> triangles
        face_vertex_counts = []
        face_vertex_indices = []
        for j in range(grid_n - 1):
            for i in range(grid_n - 1):
                i0 = j * grid_n + i
                i1 = i0 + 1
                i2 = i0 + grid_n
                i3 = i2 + 1
                # two triangles (i0,i1,i3) and (i0,i3,i2)
                face_vertex_counts.extend([3, 3])
                face_vertex_indices.extend([i0, i1, i3, i0, i3, i2])

        mesh = UsdGeom.Mesh.Define(stage, "/World/SeaFloorMesh")
        mesh.CreatePointsAttr([Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])) for p in points])
        mesh.CreateFaceVertexCountsAttr(face_vertex_counts)
        mesh.CreateFaceVertexIndicesAttr(face_vertex_indices)
        mesh.CreateDoubleSidedAttr(True)
        mesh.CreateSubdivisionSchemeAttr("none")
        # color variation: sandy -> rocky via height
        c = np.clip((h - h.min()) / (h.ptp() + 1e-8), 0.0, 1.0)
        base = np.stack([0.20 + 0.25 * c, 0.18 + 0.22 * c, 0.14 + 0.18 * c], axis=-1).reshape(-1, 3)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(float(r), float(g), float(b)) for r, g, b in base])

        # Obstacles: “rocks”
        rng = np.random.default_rng(0)

        def _define_rock(path: str, position, radius: float, scale_xyz, color_rgb):
            rock = UsdGeom.Sphere.Define(stage, path)
            rock.CreateRadiusAttr(float(radius))
            rock.CreateDisplayColorAttr([Gf.Vec3f(float(color_rgb[0]), float(color_rgb[1]), float(color_rgb[2]))])
            xf = UsdGeom.Xformable(rock)
            xf.AddTranslateOp().Set(Gf.Vec3f(float(position[0]), float(position[1]), float(position[2])))
            xf.AddScaleOp().Set(Gf.Vec3f(float(scale_xyz[0]), float(scale_xyz[1]), float(scale_xyz[2])))

        for k in range(35):
            pos = np.array([rng.uniform(-12, 12), rng.uniform(-12, 12), rng.uniform(-4.8, -3.2)], dtype=float)
            radius = float(rng.uniform(0.35, 1.1))
            scale = np.array([rng.uniform(0.7, 1.7), rng.uniform(0.7, 1.7), rng.uniform(0.5, 1.5)], dtype=float)
            color = np.array([0.10, 0.12, 0.16]) + rng.uniform(0.0, 0.06, size=3)
            _define_rock(f"/World/Rock_{k:02d}", pos, radius, scale, color)

        # Particulates: slow “bubbles/suspended matter” (many tiny cuboids).
        bubble_paths = []
        for k in range(220):
            x = rng.uniform(-15, 15)
            y = rng.uniform(-15, 15)
            z0 = rng.uniform(-5.0, 2.0)
            bub = UsdGeom.Sphere.Define(stage, f"/World/Bubble_{k:03d}")
            bub.CreateRadiusAttr(float(rng.uniform(0.012, 0.028)))
            bub.CreateDisplayColorAttr([Gf.Vec3f(0.82, 0.92, 1.0)])
            xf = UsdGeom.Xformable(bub)
            xf.AddTranslateOp().Set(Gf.Vec3f(float(x), float(y), float(z0)))
            xf.AddScaleOp().Set(
                Gf.Vec3f(float(rng.uniform(0.8, 1.6)), float(rng.uniform(0.8, 1.6)), float(rng.uniform(0.8, 1.6)))
            )
            bubble_paths.append((bub.GetPath().pathString, z0))

        # Multi-agent “vehicles”: dynamic-looking proxies (visual for now, but 3D + moving).
        agents = []
        agent_pos = []
        for i in range(cfg.num_agents):
            a = 2 * math.pi * i / max(1, cfg.num_agents)
            pos = np.array([cfg.radius * math.cos(a), cfg.radius * math.sin(a), -2.0])
            agent = VisualCuboid(
                prim_path=f"/World/Agent_{i:02d}",
                name=f"Agent_{i:02d}",
                position=pos,
                scale=np.array([0.5, 0.22, 0.18]),
                color=np.array([0.95, 0.3, 0.1]) if i == 0 else np.array([0.2, 0.7, 0.9]),
            )
            agents.append(agent)
            agent_pos.append(pos.astype(float))
        agent_pos = np.stack(agent_pos, axis=0)

        # Camera + replicator render product
        set_camera_view(eye=np.array([12.0, -12.0, 3.0]), target=np.array([0.0, 0.0, -2.5]))
        render_product = rep.create.render_product("/OmniverseKit_Persp", (cfg.width, cfg.height))
        rgb = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu")
        rgb.attach([render_product])
        dist = rep.AnnotatorRegistry.get_annotator("distance_to_camera", device="cpu")
        dist.attach([render_product])

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        simulation_app.update()

        frames = []
        lon_min = float(lons.min()) if lons is not None else None
        lon_max = float(lons.max()) if lons is not None else None
        lat_min = float(lats.min()) if lats is not None else None
        lat_max = float(lats.max()) if lats is not None else None

        for t in range(cfg.frames):
            # Orbit camera for parallax
            theta = 2 * math.pi * (t / max(1, cfg.frames))
            eye = np.array([14.0 * math.cos(theta), 14.0 * math.sin(theta), 3.5])
            tgt = np.array([0.0, 0.0, -2.6])
            set_camera_view(eye=eye, target=tgt)

            # Agents swirl + drift
            for i, agent in enumerate(agents):
                a = 2 * math.pi * i / max(1, cfg.num_agents) + 0.8 * theta
                drift = 0.3 * math.sin(0.6 * theta + i)
                desired = np.array(
                    [
                        (cfg.radius + drift) * math.cos(a),
                        (cfg.radius + drift) * math.sin(a),
                        -2.0 + 0.2 * math.sin(theta + i),
                    ],
                    dtype=float,
                )

                # Current-driven advection from our dataset (nearest-neighbor sampling).
                adv = np.zeros(3, dtype=float)
                if uo is not None and vo is not None and lats is not None and lons is not None and depths is not None:
                    # Map world x/y to lon/lat bounds.
                    x, y, zc = agent_pos[i]
                    lon = lon_min + (x / (2 * size) + 0.5) * (lon_max - lon_min)
                    lat = lat_min + (y / (2 * size) + 0.5) * (lat_max - lat_min)
                    depth = float(np.clip(-zc, float(depths.min()), float(depths.max())))
                    ti = 0
                    di = _nearest_idx(depths, depth)
                    la = _nearest_idx(lats, lat)
                    lo = _nearest_idx(lons, lon)
                    uu = float(uo.isel(time=ti, depth=di, latitude=la, longitude=lo).values)
                    vv = float(vo.isel(time=ti, depth=di, latitude=la, longitude=lo).values)
                    adv[:2] = np.array([uu, vv], dtype=float)

                # Simple proportional “controller” to loosely follow desired path + advection.
                k = 0.04
                agent_pos[i] = agent_pos[i] + k * (desired - agent_pos[i]) + 0.15 * adv
                agent.set_world_pose(position=agent_pos[i])

            # Bubbles rise and wrap
            for (path, z0) in bubble_paths:
                prim = stage.GetPrimAtPath(path)
                if not prim.IsValid():
                    continue
                xform = UsdGeom.Xformable(prim)
                ops = xform.GetOrderedXformOps()
                if not ops:
                    continue
                # assume translate is first
                tr = ops[0].Get()
                z_new = float(tr[2]) + 0.01 + 0.01 * math.sin(theta * 3.0)
                if z_new > 2.5:
                    z_new = -5.0
                ops[0].Set(Gf.Vec3d(tr[0], tr[1], z_new))

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

        timeline.stop()

        png_path = out_dir / "frame_000.png"
        gif_path = out_dir / "rollout.gif"
        iio.imwrite(png_path, frames[-1])
        iio.imwrite(gif_path, frames, duration=1.0 / cfg.fps, loop=0)

        (out_dir / "media_manifest.json").write_text(
            json.dumps(
                {
                    "track": "H6_MarineGym_FallbackIsaacSim",
                    "note": "Fallback scene (multi-agent visuals) while porting MarineGym code to Isaac Sim 5.1.",
                    "combined_nc": combined_nc if (combined_nc and Path(combined_nc).exists()) else None,
                    "combined_vars_used": ["uo", "vo"] if uo is not None and vo is not None else [],
                    "num_agents": cfg.num_agents,
                    "frames_saved": len(frames),
                    "experience": experience,
                    "simapp_cfg": simapp_cfg,
                    "png": str(png_path),
                    "gif": str(gif_path),
                    "cmd_hint": "OUT_DIR=... /home/shuaijun/isaacsim/python.sh tracks/h6_marinegym/run_fallback_multiagent_scene.py",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(str(png_path))
        print(str(gif_path))
        return 0
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
