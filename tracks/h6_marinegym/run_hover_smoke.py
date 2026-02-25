from __future__ import annotations

import json
import os
import sys
import traceback
import importlib.util
from pathlib import Path

from omegaconf import OmegaConf


def _marinegym_src_from_env() -> Path:
    # Prefer an explicit env var; fall back to cached path under repo.
    p = os.environ.get("MARINEGYM_SRC")
    if p:
        return Path(p).expanduser().resolve()
    return Path("runs/_cache/external_scenes/marinegym/MarineGym-main").expanduser().resolve()


def main() -> int:
    out_dir = Path(os.environ.get("OUT_DIR", "runs/h6_marinegym/_tmp")).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    marinegym_src = _marinegym_src_from_env()
    if not marinegym_src.exists():
        raise SystemExit(
            f"Missing MarineGym source at {marinegym_src}. "
            "Run: python3 tracks/h6_marinegym/fetch_marinegym_source.py"
        )

    patcher_path = Path(__file__).with_name("patch_marinegym_for_isaacsim51.py")
    spec = importlib.util.spec_from_file_location("_mg_patch", patcher_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to import patcher: {patcher_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.patch_marinegym(marinegym_src)

    sys.path.insert(0, str(marinegym_src))

    # Isaac Sim requirement: instantiate SimulationApp before importing other omni.* modules.
    from isaacsim import SimulationApp

    experience = "/home/shuaijun/isaacsim/apps/isaacsim.exp.base.python.kit"
    simapp_cfg = {
        "headless": True,
        "width": 1280,
        "height": 720,
        "renderer": "RayTracedLighting",
        "multi_gpu": False,
        "max_gpu_count": 1,
        # This machine hangs at startup unless we disable these validations.
        "extra_args": [
            "--/validate/p2p/enabled=false",
            "--/validate/p2p/memoryCheck/enabled=false",
            "--/validate/iommu/enabled=false",
            "--/validate/wait=0",
        ],
    }

    simulation_app = SimulationApp(simapp_cfg, experience=experience)
    try:
        import imageio.v3 as iio

        from marinegym.envs.single.hover import Hover

        sim_cfg = {
            "dt": 0.016,
            "substeps": 1,
            "gravity": [0, 0, -9.81],
            "replicate_physics": False,
            "use_flatcache": True,
            "use_gpu_pipeline": True,
            "device": "cuda:0",
            "solver_type": 1,
            "use_gpu": True,
            "bounce_threshold_velocity": 0.2,
            "friction_offset_threshold": 0.04,
            "friction_correlation_distance": 0.025,
            "enable_stabilization": True,
        }
        env_cfg = {"num_envs": 1, "env_spacing": 6, "max_episode_length": 120}
        viewer_cfg = {"resolution": [1280, 720], "eye": [8, 0, 6], "lookat": [0, 0, 3]}

        disturbances = {
            "evaluate": {
                "flow": {
                    "enable_flow": True,
                    "max_flow_velocity": [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
                    "flow_velocity_gaussian_noise": [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
                },
                "payload": {"enable_payload": False, "mass": [0.01, 0.2], "z": [-0.1, 0.1]},
            },
            "train": {
                "flow": {
                    "enable_flow": False,
                    "max_flow_velocity": [0.5, 0.5, 0.5, 0.0, 0.0, 0.0],
                    "flow_velocity_gaussian_noise": [0.1, 0.1, 0.1, 0.0, 0.0, 0.0],
                },
                "payload": {"enable_payload": False, "mass": [0.01, 0.2], "z": [-0.1, 0.1]},
            },
        }
        randomization = {
            "evaluate": {"enable_randomization": False, "body": {}, "rotor": {}},
            "train": {"enable_randomization": False, "body": {}, "rotor": {}},
        }

        task_cfg = {
            "name": "Hover",
            "env": env_cfg,
            "sim": sim_cfg,
            "drone_model": {"name": "BlueROV", "controller": "LeePositionController"},
            "force_sensor": False,
            "time_encoding": True,
            "reward_effort_weight": 0.1,
            "reward_action_smoothness_weight": 0.0,
            "reward_distance_scale": 1.2,
            "action_transform": None,
            "disturbances": disturbances,
            "randomization": randomization,
        }
        cfg = OmegaConf.create(
            {
                "headless": True,
                "enable_livestream": False,
                "mode": "evaluate",
                "sim": sim_cfg,
                "env": env_cfg,
                "viewer": viewer_cfg,
                "task": task_cfg,
            }
        )

        env = Hover(cfg, headless=True)
        td = env.reset()
        frames = []
        for i in range(90):
            td = env.rand_step(td)
            if i % 3 == 0:
                frames.append(env.render("rgb_array"))

        png_path = out_dir / "frame_000.png"
        gif_path = out_dir / "rollout.gif"
        iio.imwrite(png_path, frames[-1])
        iio.imwrite(gif_path, frames, duration=0.05, loop=0)

        (out_dir / "media_manifest.json").write_text(
            json.dumps(
                {
                    "track": "H6_MarineGym",
                    "marinegym_src": str(marinegym_src),
                    "experience": experience,
                    "simapp_cfg": simapp_cfg,
                    "frames": len(frames),
                    "png": str(png_path),
                    "gif": str(gif_path),
                    "cmd_hint": "OUT_DIR=... /home/shuaijun/isaacsim/python.sh tracks/h6_marinegym/run_hover_smoke.py",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        print(str(png_path))
        print(str(gif_path))
        return 0

    except Exception as e:
        err_path = out_dir / "error.txt"
        err_path.write_text(
            f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}",
            encoding="utf-8",
        )
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
