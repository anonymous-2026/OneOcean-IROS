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


def _look_at_quat_world_wxyz(camera_pos: np.ndarray, target_pos: np.ndarray, up: np.ndarray) -> np.ndarray:
    """
    Build a quaternion in Isaac Sim "world camera" convention:
    - +X forward, +Z up, +Y left.
    This matches Camera.set_world_pose(..., camera_axes="world").
    """
    cam = np.asarray(camera_pos, dtype=float).reshape(3)
    tgt = np.asarray(target_pos, dtype=float).reshape(3)
    up = np.asarray(up, dtype=float).reshape(3)

    forward = tgt - cam
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-9:
        forward = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        forward = forward / forward_norm

    up_norm = float(np.linalg.norm(up))
    if up_norm < 1e-9:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        up = up / up_norm

    # Camera Y axis is LEFT. Use left = up x forward.
    left = np.cross(up, forward)
    left_norm = float(np.linalg.norm(left))
    if left_norm < 1e-9:
        # forward ~ up; choose an arbitrary left
        left = np.array([0.0, 1.0, 0.0], dtype=float)
    else:
        left = left / left_norm

    up_ortho = np.cross(forward, left)
    up_ortho_norm = float(np.linalg.norm(up_ortho))
    if up_ortho_norm < 1e-9:
        up_ortho = up
    else:
        up_ortho = up_ortho / up_ortho_norm

    # Rotation matrix mapping camera(world-convention) axes -> world axes.
    # Columns are world vectors of camera basis: [forward, left, up].
    R = np.stack([forward, left, up_ortho], axis=1)

    # Convert rotation matrix to quaternion (w,x,y,z), right-handed.
    tr = float(np.trace(R))
    if tr > 0.0:
        S = np.sqrt(tr + 1.0) * 2.0
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif (R[0, 0] > R[1, 1]) and (R[0, 0] > R[2, 2]):
        S = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S

    quat = np.array([qw, qx, qy, qz], dtype=float)
    quat = quat / max(1e-12, float(np.linalg.norm(quat)))
    return quat


def _maybe_to_numpy(arr):
    if arr is None:
        return None
    if isinstance(arr, np.ndarray):
        return arr
    if hasattr(arr, "numpy"):
        try:
            return arr.numpy()
        except Exception:
            pass
    try:
        return np.asarray(arr)
    except Exception:
        return None


def _json_dump(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Headless OceanSim UW camera + sonar demo (writes PNG/NPY).")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--usd",
        type=str,
        default="",
        help="Optional USD path to load as scene. If empty, uses OceanSim's official MHL example scene.",
    )
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--warmup_frames", type=int, default=20)
    parser.add_argument("--debug_raw_frames", type=int, default=0, help="Dump raw LdrColor + depth for first N frames.")
    parser.add_argument(
        "--camera_preset",
        choices=("isometric", "forward_y", "topdown"),
        default="isometric",
        help="Camera preset to help get readable frames in headless mode.",
    )
    parser.add_argument("--no_sonar", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--limit_cpu_threads", type=int, default=32)
    args = parser.parse_args()

    out_dir = args.out.expanduser().resolve()
    (out_dir / "uw_camera").mkdir(parents=True, exist_ok=True)
    (out_dir / "sonar").mkdir(parents=True, exist_ok=True)
    if int(args.debug_raw_frames) > 0:
        (out_dir / "raw_camera").mkdir(parents=True, exist_ok=True)

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
        from pxr import Gf, Usd, UsdGeom, UsdLux

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

        from isaacsim.core.utils.stage import add_reference_to_stage

        # Load an external scene (official OceanSim MHL by default).
        if str(args.usd).strip():
            scene_usd = str(args.usd).strip()
            scene_prim = "/World/scene"
            add_reference_to_stage(usd_path=scene_usd, prim_path=scene_prim)
        else:
            from isaacsim.oceansim.utils.assets_utils import get_oceansim_assets_path

            assets_root = Path(get_oceansim_assets_path())
            scene_usd = str((assets_root / "collected_MHL" / "mhl_scaled.usd").resolve())
            scene_prim = "/World/mhl"
            add_reference_to_stage(usd_path=scene_usd, prim_path=scene_prim)

        simulation_app.update()
        simulation_app.update()

        dome = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
        dome.CreateIntensityAttr(250.0)
        dome.CreateColorAttr((0.35, 0.6, 1.0))

        key_light = UsdLux.DistantLight.Define(stage, "/World/KeyLight")
        key_light.CreateIntensityAttr(60000.0)
        key_light.CreateColorAttr((0.95, 0.98, 1.0))

        from isaacsim.oceansim.sensors.UW_Camera import UW_Camera

        cam_tgt = np.array([0.0, 0.0, 0.0], dtype=float)
        dist = 20.0
        try:
            prim = stage.GetPrimAtPath(scene_prim)
            if prim.IsValid():
                bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), includedPurposes=[UsdGeom.Tokens.default_])
                world_box = bbox_cache.ComputeWorldBound(prim)
                box_range = world_box.ComputeAlignedBox().GetRange()
                mn = np.array([box_range.GetMin()[0], box_range.GetMin()[1], box_range.GetMin()[2]], dtype=float)
                mx = np.array([box_range.GetMax()[0], box_range.GetMax()[1], box_range.GetMax()[2]], dtype=float)
                cam_tgt = 0.5 * (mn + mx)
                diag = float(np.linalg.norm(mx - mn))
                dist = max(5.0, 0.8 * diag)
        except Exception:
            pass

        if args.camera_preset == "topdown":
            cam_pos = cam_tgt + np.array([0.0, -0.2 * dist, 0.9 * dist], dtype=float)
        elif args.camera_preset == "forward_y":
            cam_pos = cam_tgt + np.array([0.0, -1.2 * dist, 0.15 * dist], dtype=float)
        else:
            cam_pos = cam_tgt + np.array([0.9 * dist, -0.9 * dist, 0.35 * dist], dtype=float)

        cam_q_world = _look_at_quat_world_wxyz(cam_pos, cam_tgt, up=np.array([0.0, 0.0, 1.0], dtype=float))

        cam_light = UsdLux.SphereLight.Define(stage, "/World/CameraLight")
        cam_light.AddTranslateOp().Set(Gf.Vec3d(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])))
        cam_light.CreateIntensityAttr(35000.0)
        cam_light.CreateRadiusAttr(2.0)
        cam_light.CreateColorAttr((0.95, 0.98, 1.0))

        uw_cam = UW_Camera(
            prim_path="/World/UWCamera",
            name="UWCamera",
            resolution=(1280, 720),
            position=cam_pos.tolist(),
            orientation=[1.0, 0.0, 0.0, 0.0],
        )
        try:
            cam_prim = stage.GetPrimAtPath("/World/UWCamera")
            usd_cam = UsdGeom.Camera(cam_prim)
            usd_cam.CreateClippingRangeAttr().Set((0.05, 250.0))
            usd_cam.CreateFocalLengthAttr().Set(18.0)
        except Exception:
            pass
        # NOTE: OceanSim's UW_Camera currently maps UW_param indices as:
        # - backscatter_value = [0:3]
        # - attenuation_coeff = [6:9]  (swapped vs docstring)
        # - backscatter_coeff = [3:6]  (swapped vs docstring)
        # We set a mild haze but avoid the "solid green wash".
        uw_param = np.array(
            [
                0.02,
                0.06,
                0.08,  # backscatter_value (0..1)
                0.010,
                0.012,
                0.016,  # backscatter_coeff (interpreted from [3:6])
                0.020,
                0.030,
                0.050,  # attenuation_coeff (interpreted from [6:9])
            ],
            dtype=float,
        )
        uw_cam.initialize(UW_param=uw_param, viewport=False, writing_dir=str(out_dir / "uw_camera"))
        try:
            uw_cam.set_world_pose(position=cam_pos.tolist(), orientation=cam_q_world.tolist(), camera_axes="world")
        except Exception:
            pass

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
            # Simple camera orbit to create visible motion without modifying the scene content.
            theta = 2.0 * np.pi * (i / max(1.0, float(max(1, args.frames - 1))))
            orbit = cam_tgt + np.array([np.cos(theta), np.sin(theta), 0.0], dtype=float) * (0.25 * dist)
            cam_pos_i = orbit + (cam_pos - cam_tgt)
            cam_q_i = _look_at_quat_world_wxyz(cam_pos_i, cam_tgt, up=np.array([0.0, 0.0, 1.0], dtype=float))
            try:
                uw_cam.set_world_pose(position=cam_pos_i.tolist(), orientation=cam_q_i.tolist(), camera_axes="world")
                cam_light.GetPrim().GetAttribute("xformOp:translate").Set(
                    Gf.Vec3d(float(cam_pos_i[0]), float(cam_pos_i[1]), float(cam_pos_i[2]))
                )
            except Exception:
                pass

            simulation_app.update()
            if int(args.debug_raw_frames) > 0 and i < int(args.debug_raw_frames):
                try:
                    from PIL import Image

                    raw_rgba = None
                    raw_depth = None
                    annot_rgba = getattr(uw_cam, "_rgba_annot", None)
                    annot_depth = getattr(uw_cam, "_depth_annot", None)
                    if annot_rgba is not None:
                        raw_rgba = annot_rgba.get_data()
                    if annot_depth is not None:
                        raw_depth = annot_depth.get_data()

                    rgba_np = _maybe_to_numpy(raw_rgba)
                    depth_np = _maybe_to_numpy(raw_depth)

                    if rgba_np is not None and getattr(rgba_np, "shape", None) is not None and rgba_np.size != 0:
                        Image.fromarray(rgba_np.astype("uint8"), mode="RGBA").save(
                            out_dir / "raw_camera" / f"ldr_{i:04d}.png"
                        )

                    if depth_np is not None and getattr(depth_np, "shape", None) is not None and depth_np.size != 0:
                        depth_np = np.asarray(depth_np, dtype=np.float32)
                        np.save(out_dir / "raw_camera" / f"depth_{i:04d}.npy", depth_np)
                        finite = np.isfinite(depth_np)
                        if finite.any():
                            d = depth_np[finite]
                            d_min = float(np.percentile(d, 1.0))
                            d_max = float(np.percentile(d, 99.0))
                            if d_max <= d_min + 1e-6:
                                d_max = d_min + 1.0
                            depth_vis = np.zeros_like(depth_np, dtype=np.uint8)
                            depth_norm = (np.clip(depth_np, d_min, d_max) - d_min) / (d_max - d_min)
                            depth_vis[finite] = (255.0 * (1.0 - depth_norm[finite])).astype(np.uint8)
                            Image.fromarray(depth_vis, mode="L").save(out_dir / "raw_camera" / f"depth_{i:04d}.png")
                except Exception:
                    pass
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
            "scene_usd": scene_usd,
            "scene_prim_path": scene_prim,
            "camera_preset": args.camera_preset,
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
