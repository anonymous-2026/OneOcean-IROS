from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np


def _look_at_quat_wxyz(camera_pos: np.ndarray, target_pos: np.ndarray, up: np.ndarray) -> np.ndarray:
    from pxr import Gf

    cam = np.asarray(camera_pos, dtype=float).reshape(3)
    tgt = np.asarray(target_pos, dtype=float).reshape(3)
    up = np.asarray(up, dtype=float).reshape(3)

    eye = Gf.Vec3d(float(cam[0]), float(cam[1]), float(cam[2]))
    target = Gf.Vec3d(float(tgt[0]), float(tgt[1]), float(tgt[2]))
    up_axis = Gf.Vec3d(float(up[0]), float(up[1]), float(up[2]))
    q = Gf.Matrix4d().SetLookAt(eye, target, up_axis).GetInverse().ExtractRotation().GetQuat()
    imag = q.GetImaginary()
    quat = np.array([float(q.GetReal()), float(imag[0]), float(imag[1]), float(imag[2])], dtype=float)
    quat = quat / max(1e-12, float(np.linalg.norm(quat)))
    return quat


def _json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless OceanSim UW camera + sonar demo (writes PNG/NPY).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup_frames", type=int, default=20)
    parser.add_argument("--no_sonar", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit_cpu_threads", type=int, default=32)
    args = parser.parse_args()

    out_dir = args.out.expanduser().resolve()
    (out_dir / "uw_camera").mkdir(parents=True, exist_ok=True)
    (out_dir / "sonar").mkdir(parents=True, exist_ok=True)

    oceansim_ext_root = Path("/home/shuaijun/isaacsim/extsUser/OceanSim").resolve()
    args_json = {}
    for k, v in vars(args).items():
        if isinstance(v, Path):
            args_json[k] = str(v)
        else:
            args_json[k] = v
    debug_state: dict = {
        "ts_unix": time.time(),
        "args": args_json,
        "oceansim_ext_root": str(oceansim_ext_root),
        "cwd": os.getcwd(),
        "env_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    _json_dump(out_dir / "debug_state.json", debug_state)

    oceansim_exts_folder = oceansim_ext_root.parent if oceansim_ext_root.exists() else None

    from isaacsim import SimulationApp

    extra_args = [
        "--/validate/p2p/enabled=false",
        "--/validate/p2p/memoryCheck/enabled=false",
        "--/validate/iommu/enabled=false",
        "--/renderer/multiGpu/enabled=false",
        "--/renderer/multiGpu/maxGpuCount=1",
        f"--/renderer/activeGpu={int(args.gpu)}",
        f"--/physics/cudaDevice={int(args.gpu)}",
    ]
    if oceansim_exts_folder is not None:
        # Make the extension discoverable to Kit so `enable_extension("OceanSim")` can work.
        extra_args = ["--ext-folder", str(oceansim_exts_folder)] + extra_args
        debug_state["kit_ext_folder_added"] = str(oceansim_exts_folder)
        _json_dump(out_dir / "debug_state.json", debug_state)

    launcher_cfg = {
        "headless": True,
        "multi_gpu": False,
        "limit_cpu_threads": int(args.limit_cpu_threads),
        "active_gpu": int(args.gpu),
        "physics_gpu": int(args.gpu),
        "extra_args": extra_args,
    }
    simulation_app = SimulationApp(launcher_cfg)
    t0 = time.time()
    try:
        import omni.timeline
        import omni.usd

        from isaacsim.core.utils.extensions import enable_extension
        from pxr import Gf, UsdGeom, UsdLux

        try:
            enable_extension("omni.sensors.nv.camera")
        except Exception:
            pass
        try:
            enable_extension("omni.replicator.core")
        except Exception:
            pass
        if oceansim_ext_root.exists():
            try:
                enable_extension("OceanSim")
            except Exception:
                # The extension may still be importable if Kit already mounted extsUser.
                pass

        simulation_app.update()
        omni.usd.get_context().new_stage()
        simulation_app.update()

        stage = omni.usd.get_context().get_stage()
        assert stage is not None
        UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
        UsdGeom.SetStageMetersPerUnit(stage, 1.0)
        world_xf = UsdGeom.Xform.Define(stage, "/World")
        stage.SetDefaultPrim(world_xf.GetPrim())

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(500.0)
        dome.CreateColorAttr((0.45, 0.7, 1.0))

        floor = UsdGeom.Cube.Define(stage, "/World/SeaFloor")
        floor.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, -2.0))
        floor.AddScaleOp().Set(Gf.Vec3f(100.0, 100.0, 0.2))

        rng = np.random.default_rng(0)
        for i in range(14):
            rock = UsdGeom.Sphere.Define(stage, f"/World/Rock_{i:02d}")
            rock.AddTranslateOp().Set(
                Gf.Vec3d(float(rng.uniform(-10, 10)), float(rng.uniform(-10, 10)), float(rng.uniform(-1.9, -1.0)))
            )
            rock.AddScaleOp().Set(Gf.Vec3f(float(rng.uniform(0.4, 1.6)), float(rng.uniform(0.4, 1.6)), float(rng.uniform(0.2, 1.2))))

        # Particulate matter ("silt") to help the scene read as underwater (parallax + suspended particles).
        silt_ops: list[tuple[UsdGeom.Sphere, "UsdGeom.XformOp"]] = []
        for i in range(160):
            p = UsdGeom.Sphere.Define(stage, f"/World/Silt/Silt_{i:03d}")
            t_op = p.AddTranslateOp()
            t_op.Set(Gf.Vec3d(float(rng.uniform(-12, 12)), float(rng.uniform(-12, 12)), float(rng.uniform(-2.0, 3.0))))
            p.AddScaleOp().Set(Gf.Vec3f(float(rng.uniform(0.02, 0.05))))
            silt_ops.append((p, t_op))

        uuv = UsdGeom.Cube.Define(stage, "/World/UUV")
        uuv_t = uuv.AddTranslateOp()
        uuv_s = uuv.AddScaleOp()
        uuv_s.Set(Gf.Vec3f(0.6, 0.25, 0.25))
        uuv_t.Set(Gf.Vec3d(-6.0, 0.0, -1.2))

        from isaacsim.oceansim.sensors.UW_Camera import UW_Camera

        cam_pos = np.array([9.0, 7.0, 1.8], dtype=float)
        cam_tgt = np.array([0.0, 0.0, -1.3], dtype=float)
        cam_q = _look_at_quat_wxyz(cam_pos, cam_tgt, up=np.array([0.0, 0.0, 1.0], dtype=float))

        uw_cam = UW_Camera(
            prim_path="/World/UWCamera",
            name="UWCamera",
            resolution=(1280, 720),
            position=cam_pos.tolist(),
            orientation=cam_q.tolist(),
        )
        uw_cam.initialize(viewport=False, writing_dir=str(out_dir / "uw_camera"))

        sonar = None
        if not args.no_sonar:
            from isaacsim.oceansim.sensors.ImagingSonarSensor import ImagingSonarSensor

            sonar = ImagingSonarSensor(
                prim_path="/World/ImagingSonar",
                name="ImagingSonar",
                position=[2.0, 0.0, -1.0],
                orientation=[1.0, 0.0, 0.0, 0.0],
                hori_res=512,
                range_res=0.05,
                max_range=12.0,
                min_range=0.5,
                angular_res=1.0,
                hori_fov=120.0,
                vert_fov=20.0,
            )
            sonar.sonar_initialize(output_dir=str(out_dir / "sonar"), viewport=False, include_unlabelled=True)

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        simulation_app.update()

        for _ in range(int(args.warmup_frames)):
            simulation_app.update()
            uw_cam.render()

        frames_written = 0
        for i in range(int(args.frames)):
            theta = 2.0 * np.pi * (i / max(1.0, float(args.frames)))
            x = -6.0 + 12.0 * (i / max(1.0, float(max(1, args.frames - 1))))
            y = 2.0 * np.sin(theta)
            z = -1.25 + 0.15 * np.cos(theta)
            uuv_t.Set(Gf.Vec3d(float(x), float(y), float(z)))

            # Slow silt drift for parallax.
            for _, op in silt_ops:
                v = op.Get()
                new_y = float(v[1]) + 0.01
                if new_y > 12.0:
                    new_y = -12.0
                op.Set(Gf.Vec3d(float(v[0]), new_y, float(v[2])))

            simulation_app.update()
            uw_cam.render()
            if sonar is not None:
                sonar.make_sonar_data()
            frames_written += 1

        timeline.stop()

        try:
            uw_cam.close()
        except Exception:
            pass
        try:
            if sonar is not None:
                sonar.close()
        except Exception:
            pass

        try:
            import omni.replicator.core as rep

            rep.orchestrator.wait_until_complete()
        except Exception:
            pass

        manifest = {
            "track": "H5_OceanSim",
            "isaacsim_root": "/home/shuaijun/isaacsim",
            "oceansim_ext_root": str(oceansim_ext_root) if oceansim_ext_root.exists() else None,
            "frames": int(args.frames),
            "warmup_frames": int(args.warmup_frames),
            "frames_written": int(frames_written),
            "uw_camera_dir": str(out_dir / "uw_camera"),
            "sonar_dir": None if args.no_sonar else str(out_dir / "sonar"),
            "cmd_hint": "This script disables Kit P2P/IOMMU validation to avoid multi-GPU startup delays.",
        }
        _json_dump(out_dir / "media_manifest.json", manifest)
        _json_dump(
            out_dir / "results_manifest.json",
            {
                "track": "H5_OceanSim",
                "frames_written": int(frames_written),
                "uw_camera_png_count": len(list((out_dir / "uw_camera").glob("*.png"))),
                "sonar_npy_count": len(list((out_dir / "sonar").glob("*.npy"))),
            },
        )
        return 0
    except Exception as exc:
        (out_dir / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        print(f"[H5 OceanSim] ERROR: {exc}\n{traceback.format_exc()}")
        return 2
    finally:
        simulation_app.close()
        print(f"[done] elapsed_s={time.time()-t0:.1f} out={out_dir}")


if __name__ == "__main__":
    raise SystemExit(main())
