#!/usr/bin/env python3
"""
Preview-render *full* third-party underwater/ocean scenes from UUV Simulator.

Goal: help select an external, already-made scene to adopt/adapt (per project requirements),
without committing any large third-party binaries into our repos.

This script:
1) Loads selected UUV Gazebo world/model assets from a local checkout.
2) Converts referenced meshes (.dae/.stl) to .obj via trimesh (cached under runs/_cache/).
3) Builds a lightweight MuJoCo MJCF scene composed of static geoms.
4) Renders PNGs + an orbit GIF (headless EGL) into runs/.

Run with the MuJoCo venv (recommended):
  MUJOCO_GL=egl /data/private/user2/workspace/robosuite_learning/.venv/bin/python \
    tools/external_scenes/uuv_world_preview.py --world herkules_ship_wreck
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from xml.etree import ElementTree as ET

import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh


@dataclass(frozen=True)
class Pose6D:
    x: float
    y: float
    z: float
    roll: float
    pitch: float
    yaw: float


def _parse_pose6d(text: str | None) -> Pose6D:
    if not text:
        return Pose6D(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    parts = [p for p in text.replace("\n", " ").split(" ") if p.strip()]
    vals = [float(v) for v in parts]
    if len(vals) != 6:
        raise ValueError(f"Expected 6 pose values, got {len(vals)} from {text!r}")
    return Pose6D(*vals)


def _rpy_to_quat_wxyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return np.array([w, x, y, z], dtype=np.float64)


def _quat_mul_wxyz(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def _quat_rotate_vec_wxyz(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
    q_conj = np.array([w, -x, -y, -z], dtype=np.float64)
    r = _quat_mul_wxyz(_quat_mul_wxyz(q, qv), q_conj)
    return r[1:]


def _compose_pose(parent: Pose6D, child: Pose6D) -> tuple[np.ndarray, np.ndarray]:
    p_pos = np.array([parent.x, parent.y, parent.z], dtype=np.float64)
    c_pos = np.array([child.x, child.y, child.z], dtype=np.float64)
    p_q = _rpy_to_quat_wxyz(parent.roll, parent.pitch, parent.yaw)
    c_q = _rpy_to_quat_wxyz(child.roll, child.pitch, child.yaw)
    out_q = _quat_mul_wxyz(p_q, c_q)
    out_pos = p_pos + _quat_rotate_vec_wxyz(p_q, c_pos)
    return out_pos, out_q


def _safe_mkdir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _resolve_uuv_model_dir(uuv_root: Path, model_name: str) -> Path:
    candidates = [
        uuv_root / "uuv_gazebo_worlds" / "models" / model_name,
        uuv_root / "uuv_tutorials" / "uuv_tutorial_seabed_world" / "models" / model_name,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(f"Cannot locate model://{model_name} under {uuv_root}")


def _mesh_uri_to_path(uuv_root: Path, uri: str) -> Path:
    uri = uri.strip()
    if uri.startswith("model://"):
        rest = uri[len("model://") :]
        model, rel = rest.split("/", 1)
        return _resolve_uuv_model_dir(uuv_root, model) / rel
    if uri.startswith("file://"):
        raise FileNotFoundError(
            f"Unsupported mesh URI {uri!r} (points to Gazebo Media/ assets not vendored here)"
        )
    raise FileNotFoundError(f"Unsupported mesh URI {uri!r}")


def _convert_mesh_to_obj(src: Path, dst_obj: Path, scale_xyz: np.ndarray) -> None:
    if dst_obj.exists():
        return
    _safe_mkdir(dst_obj.parent)
    mesh = trimesh.load(src, force="mesh")
    if scale_xyz is not None:
        mesh.apply_scale(scale_xyz.astype(np.float64))
    mesh.export(dst_obj)


@dataclass(frozen=True)
class StaticGeom:
    name: str
    kind: str  # "mesh" or "box"
    pos: np.ndarray  # (3,)
    quat_wxyz: np.ndarray  # (4,)
    mesh_obj: Path | None = None
    box_size: np.ndarray | None = None  # full size xyz (SDF size), not halfsize
    rgba: tuple[float, float, float, float] = (0.25, 0.35, 0.45, 1.0)


def _parse_model_static_geoms(
    uuv_root: Path, model_name: str, cache_mesh_dir: Path
) -> list[StaticGeom]:
    model_dir = _resolve_uuv_model_dir(uuv_root, model_name)
    sdf_path = model_dir / "model.sdf"
    if not sdf_path.exists():
        raise FileNotFoundError(f"Missing {sdf_path}")

    tree = ET.parse(sdf_path)
    root = tree.getroot()
    geoms: list[StaticGeom] = []

    for link in root.findall(".//link"):
        link_pose = _parse_pose6d((link.findtext("pose") or "").strip())

        # Parse both visual and collision; prefer visual for meshes, collision for boxes.
        for tag in ("visual", "collision"):
            for item in link.findall(f"./{tag}"):
                item_name = item.get("name") or f"{model_name}_{tag}"
                item_pose = _parse_pose6d((item.findtext("pose") or "").strip())
                geom = item.find("./geometry")
                if geom is None:
                    continue

                mesh = geom.find("./mesh")
                box = geom.find("./box")

                if mesh is not None:
                    uri = mesh.findtext("./uri")
                    if not uri:
                        continue
                    src = _mesh_uri_to_path(uuv_root, uri)
                    scale_text = mesh.findtext("./scale")
                    scale_xyz = (
                        np.array([float(v) for v in scale_text.split()], dtype=np.float64)
                        if scale_text
                        else np.array([1.0, 1.0, 1.0], dtype=np.float64)
                    )
                    # Use a unique basename so multiple models with the same source stem don't collide.
                    dst_obj = cache_mesh_dir / f"{model_name}__{src.stem}.obj"
                    _convert_mesh_to_obj(src, dst_obj, scale_xyz)

                    pos, quat = _compose_pose(link_pose, item_pose)
                    geoms.append(
                        StaticGeom(
                            name=f"{model_name}__{tag}__{item_name}",
                            kind="mesh",
                            pos=pos,
                            quat_wxyz=quat,
                            mesh_obj=dst_obj,
                        )
                    )
                    continue

                if box is not None:
                    size_text = box.findtext("./size")
                    if not size_text:
                        continue
                    size = np.array([float(v) for v in size_text.split()], dtype=np.float64)
                    pos, quat = _compose_pose(link_pose, item_pose)
                    geoms.append(
                        StaticGeom(
                            name=f"{model_name}__{tag}__{item_name}",
                            kind="box",
                            pos=pos,
                            quat_wxyz=quat,
                            box_size=size,
                            rgba=(0.02, 0.06, 0.10, 0.25) if "surface" in item_name else (0.05, 0.08, 0.12, 1.0),
                        )
                    )
                    continue

    return geoms


def _parse_world_includes(world_path: Path) -> list[tuple[str, Pose6D]]:
    tree = ET.parse(world_path)
    root = tree.getroot()
    includes: list[tuple[str, Pose6D]] = []
    for inc in root.findall(".//include"):
        uri = (inc.findtext("./uri") or "").strip()
        if not uri.startswith("model://"):
            continue
        model_name = uri[len("model://") :].strip()
        pose = _parse_pose6d((inc.findtext("./pose") or "").strip())
        includes.append((model_name, pose))
    return includes


def _build_mjcf_for_geoms(
    geoms: Iterable[StaticGeom], meshdir: Path, cameras: list[dict], off_w: int, off_h: int
) -> str:
    asset_meshes = []
    world_geoms = []
    added_mesh_names: set[str] = set()

    for g in geoms:
        if g.kind == "mesh" and g.mesh_obj is not None:
            mesh_name = g.mesh_obj.stem
            if mesh_name not in added_mesh_names:
                asset_meshes.append(
                    f"<mesh name='{mesh_name}' file='{g.mesh_obj.name}'/>"
                )
                added_mesh_names.add(mesh_name)
            world_geoms.append(
                "<geom "
                f"name='{g.name}' type='mesh' mesh='{mesh_name}' "
                f"pos='{g.pos[0]} {g.pos[1]} {g.pos[2]}' "
                f"quat='{g.quat_wxyz[0]} {g.quat_wxyz[1]} {g.quat_wxyz[2]} {g.quat_wxyz[3]}' "
                f"rgba='{g.rgba[0]} {g.rgba[1]} {g.rgba[2]} {g.rgba[3]}' "
                "contype='0' conaffinity='0'/>"
            )
        elif g.kind == "box" and g.box_size is not None:
            half = 0.5 * g.box_size
            world_geoms.append(
                "<geom "
                f"name='{g.name}' type='box' "
                f"size='{half[0]} {half[1]} {half[2]}' "
                f"pos='{g.pos[0]} {g.pos[1]} {g.pos[2]}' "
                f"quat='{g.quat_wxyz[0]} {g.quat_wxyz[1]} {g.quat_wxyz[2]} {g.quat_wxyz[3]}' "
                f"rgba='{g.rgba[0]} {g.rgba[1]} {g.rgba[2]} {g.rgba[3]}' "
                "contype='0' conaffinity='0'/>"
            )

    cam_xml = []
    for cam in cameras:
        cam_xml.append(
            f"<camera name='{cam['name']}' pos='{cam['pos']}' xyaxes='{cam['xyaxes']}'/>"
        )

    mjcf = "\n".join(
        [
            "<mujoco model='uuv_preview'>",
            f"  <compiler angle='radian' meshdir='{meshdir.as_posix()}'/>",
            "  <visual>",
            f"    <global offwidth='{off_w}' offheight='{off_h}'/>",
            "  </visual>",
            "  <option gravity='0 0 -9.81' integrator='Euler'/>",
            "  <worldbody>",
            "    <light pos='0 0 200' dir='0 0 -1' directional='true' diffuse='0.6 0.7 0.9'/>",
            "    <light pos='150 0 120' dir='-1 0 -0.6' directional='true' diffuse='0.3 0.4 0.7'/>",
            *["    " + line for line in cam_xml],
            *["    " + line for line in world_geoms],
            "  </worldbody>",
            "  <asset>",
            *["    " + line for line in asset_meshes],
            "  </asset>",
            "</mujoco>",
        ]
    )
    return mjcf


def _render_pngs_and_orbit_gif(
    mjcf_xml: str,
    out_dir: Path,
    cameras: list[str],
    orbit_lookat: np.ndarray,
    orbit_dist: float,
    orbit_elev: float,
    orbit_frames: int = 60,
    size_hw: tuple[int, int] = (480, 640),
) -> None:
    _safe_mkdir(out_dir)

    model = mujoco.MjModel.from_xml_string(mjcf_xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    h, w = size_hw
    renderer = mujoco.Renderer(model, h, w)

    for cam in cameras:
        renderer.update_scene(data, camera=cam)
        img = renderer.render()
        imageio.imwrite(out_dir / f"{cam}.png", img)

    # Orbit GIF via free camera
    frames = []
    cam = mujoco.MjvCamera()
    cam.lookat[:] = orbit_lookat
    cam.distance = orbit_dist
    cam.elevation = orbit_elev
    cam.azimuth = 0.0
    for i in range(orbit_frames):
        cam.azimuth = 360.0 * i / orbit_frames
        renderer.update_scene(data, camera=cam)
        frames.append(renderer.render())
    imageio.mimsave(out_dir / "orbit.gif", frames, duration=0.06)

    renderer.close()


def _world_config(world: str) -> dict:
    # Cameras use MuJoCo camera xml `xyaxes` (two 3D vectors: x-axis then y-axis in world frame).
    if world == "herkules_ship_wreck":
        return {
            "world_rel_path": "uuv_gazebo_worlds/worlds/herkules_ship_wreck.world",
            "cameras": [
                {
                    "name": "cam_close",
                    "pos": "-25 25 -55",
                    "xyaxes": "-0.707 0.707 0  -0.2 -0.2 0.96",
                },
                {
                    "name": "cam_side",
                    "pos": "20 -30 -55",
                    "xyaxes": "0.832 0.555 0  -0.05 0.08 0.995",
                },
            ],
            "orbit_lookat": np.array([0.0, 0.0, -60.0], dtype=np.float64),
            "orbit_dist": 55.0,
            "orbit_elev": -10.0,
        }
    if world == "munkholmen_island":
        return {
            "world_rel_path": "uuv_gazebo_worlds/worlds/munkholmen.world",
            "cameras": [
                {
                    "name": "cam_island_wide",
                    "pos": "-300 -300 20",
                    "xyaxes": "0.707 -0.707 0  0.2 0.2 0.96",
                },
                {
                    "name": "cam_island_under",
                    "pos": "-120 -160 -20",
                    "xyaxes": "0.8 0.6 0  -0.2 0.27 0.94",
                },
            ],
            "orbit_lookat": np.array([-103.391, -121.403, 0.0], dtype=np.float64),
            "orbit_dist": 250.0,
            "orbit_elev": 10.0,
        }
    if world == "tutorial_seabed":
        return {
            "world_rel_path": "uuv_tutorials/uuv_tutorial_seabed_world/worlds/example_underwater.world",
            "cameras": [
                {
                    "name": "cam_tutorial",
                    "pos": "12 -18 -8",
                    "xyaxes": "0.832 0.555 0  -0.12 0.18 0.976",
                }
            ],
            "orbit_lookat": np.array([0.0, 0.0, 0.0], dtype=np.float64),
            "orbit_dist": 30.0,
            "orbit_elev": -20.0,
        }
    raise ValueError(f"Unknown world id {world!r}")


def build_geoms_from_world(uuv_root: Path, world_id: str, cache_mesh_dir: Path) -> list[StaticGeom]:
    cfg = _world_config(world_id)
    world_path = uuv_root / cfg["world_rel_path"]
    includes = _parse_world_includes(world_path)

    geoms: list[StaticGeom] = []
    for model_name, inc_pose in includes:
        if model_name in ("sun", "ned_frame"):
            continue
        try:
            model_geoms = _parse_model_static_geoms(uuv_root, model_name, cache_mesh_dir)
        except FileNotFoundError:
            # Skip models that reference Gazebo "Media/" assets not vendored in the repo.
            continue
        inc_q = _rpy_to_quat_wxyz(inc_pose.roll, inc_pose.pitch, inc_pose.yaw)
        inc_pos = np.array([inc_pose.x, inc_pose.y, inc_pose.z], dtype=np.float64)
        for g in model_geoms:
            out_pos = inc_pos + _quat_rotate_vec_wxyz(inc_q, g.pos)
            out_quat = _quat_mul_wxyz(inc_q, g.quat_wxyz)
            geoms.append(
                StaticGeom(
                    name=g.name,
                    kind=g.kind,
                    pos=out_pos,
                    quat_wxyz=out_quat,
                    mesh_obj=g.mesh_obj,
                    box_size=g.box_size,
                    rgba=g.rgba,
                )
            )
    return geoms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--uuv-root",
        type=Path,
        default=Path("/data/private/user2/workspace/ocean/project_mgmt/sync/_external_scene_cache/uuv_simulator_full2"),
        help="Path to a local checkout of https://github.com/uuvsimulator/uuv_simulator",
    )
    parser.add_argument(
        "--world",
        type=str,
        required=True,
        choices=["herkules_ship_wreck", "munkholmen_island", "tutorial_seabed"],
        help="Which third-party UUV world to preview-render",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: runs/external_scene_previews/uuv/<world>)",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("runs/_cache/external_scenes/uuv"),
        help="Local cache dir for converted meshes (kept local; do not commit)",
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()

    if os.environ.get("MUJOCO_GL") != "egl":
        raise RuntimeError("Set MUJOCO_GL=egl for headless rendering")

    uuv_root: Path = args.uuv_root
    out_dir: Path = args.out_dir or Path("runs/external_scene_previews/uuv") / args.world
    cache_dir: Path = args.cache_dir

    cfg = _world_config(args.world)
    cache_mesh_dir = cache_dir / "meshes"
    _safe_mkdir(cache_mesh_dir)

    geoms = build_geoms_from_world(uuv_root, args.world, cache_mesh_dir)
    meshdir = cache_mesh_dir

    cameras_xml = cfg["cameras"]
    mjcf = _build_mjcf_for_geoms(
        geoms=geoms,
        meshdir=meshdir,
        cameras=cameras_xml,
        off_w=args.width,
        off_h=args.height,
    )

    # Save MJCF for inspection/debugging
    _safe_mkdir(out_dir)
    (out_dir / "scene_preview.xml").write_text(mjcf)

    camera_names = [c["name"] for c in cameras_xml]
    _render_pngs_and_orbit_gif(
        mjcf_xml=mjcf,
        out_dir=out_dir,
        cameras=camera_names,
        orbit_lookat=cfg["orbit_lookat"],
        orbit_dist=cfg["orbit_dist"],
        orbit_elev=cfg["orbit_elev"],
        orbit_frames=72,
        size_hw=(args.height, args.width),
    )

    meta = {
        "source": "UUV Simulator (Gazebo/ROS)",
        "license": "Apache-2.0",
        "uuv_root": str(uuv_root),
        "world_id": args.world,
    }
    (out_dir / "source_manifest.json").write_text(
        __import__("json").dumps(meta, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
