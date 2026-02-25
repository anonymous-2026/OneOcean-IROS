#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402
import trimesh  # noqa: E402
from PIL import Image  # noqa: E402


@dataclass(frozen=True)
class Pose:
    t_ns: int
    position: np.ndarray  # (3,)
    quat_wxyz: np.ndarray  # (4,)


def _read_jsonish_yaml(path: Path) -> dict[str, Any]:
    # MIMIR-UW sensor.yaml files are JSON-formatted.
    return json.loads(path.read_text(encoding="utf-8"))


def _load_poses(csv_path: Path) -> list[Pose]:
    poses: list[Pose] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = int(row["timestamp"])
            p = np.array([float(row["t_0_kf_X"]), float(row["t_0_kf_Y"]), float(row["t_0_kf_Z"])], dtype=np.float64)
            q = np.array(
                [float(row["q_0_kf_w"]), float(row["q_0_kf_x"]), float(row["q_0_kf_y"]), float(row["q_0_kf_z"])],
                dtype=np.float64,
            )
            poses.append(Pose(t_ns=t, position=p, quat_wxyz=q))
    poses.sort(key=lambda x: x.t_ns)
    return poses


def _nearest_pose(poses: list[Pose], t_ns: int) -> Pose:
    # poses are sorted
    idx = int(np.searchsorted([p.t_ns for p in poses], t_ns))
    if idx <= 0:
        return poses[0]
    if idx >= len(poses):
        return poses[-1]
    a = poses[idx - 1]
    b = poses[idx]
    return a if (t_ns - a.t_ns) <= (b.t_ns - t_ns) else b


def _quat_to_rot_wxyz(q: np.ndarray) -> np.ndarray:
    w, x, y, z = [float(v) for v in q]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n <= 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _invert_transform(R: np.ndarray, t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    R_inv = R.T
    t_inv = -R_inv @ t
    return R_inv, t_inv


def _load_depth(exr_path: Path) -> np.ndarray:
    depth = cv2.imread(str(exr_path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth EXR: {exr_path}")
    depth = np.asarray(depth)
    if depth.ndim == 3:
        depth = depth[:, :, 0]
    depth = depth.astype(np.float32)
    return depth


def _write_preview_depth(depth: np.ndarray, out_path: Path) -> None:
    valid = np.isfinite(depth) & (depth > 0)
    if not np.any(valid):
        vis = np.zeros(depth.shape, dtype=np.uint8)
    else:
        lo, hi = np.percentile(depth[valid], [1, 99])
        vis = np.clip((depth - lo) / (hi - lo + 1e-6), 0.0, 1.0)
        vis = (vis * 255.0).astype(np.uint8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def _build_mesh_from_frame(
    rgb_path: Path,
    depth_path: Path,
    sensor_yaml_path: Path,
    pose_csv_path: Path,
    out_dir: Path,
    stride: int,
    max_depth_m: float,
    discontinuity_m: float,
    center_on_pose: bool,
) -> dict[str, Any]:
    sensor = _read_jsonish_yaml(sensor_yaml_path)
    intr = np.asarray(sensor["intrinsics"], dtype=np.float64)
    fx = float(intr[0][0])
    fy = float(intr[1][1])
    cx = float(intr[0][2])
    cy = float(intr[1][2])

    T_BS = np.asarray(sensor["T_BS"], dtype=np.float64)
    R_BS = T_BS[:3, :3]
    t_BS = T_BS[:3, 3]
    R_SB, t_SB = _invert_transform(R_BS, t_BS)

    depth = _load_depth(depth_path)
    H, W = depth.shape

    # timestamp from filename (MIMIR uses integer ns filenames)
    try:
        t_ns = int(depth_path.stem)
    except ValueError:
        t_ns = 0

    poses = _load_poses(pose_csv_path)
    pose = _nearest_pose(poses, t_ns) if poses else Pose(t_ns=t_ns, position=np.zeros(3), quat_wxyz=np.array([1, 0, 0, 0]))
    R_WB = _quat_to_rot_wxyz(pose.quat_wxyz)
    t_WB = pose.position

    if stride < 1:
        stride = 1
    ys = np.arange(0, H, stride)
    xs = np.arange(0, W, stride)
    grid_i = -np.ones((len(ys), len(xs)), dtype=np.int32)

    verts: list[list[float]] = []
    uvs: list[list[float]] = []
    dvals = np.full((len(ys), len(xs)), np.nan, dtype=np.float32)

    for yi, v in enumerate(ys):
        for xi, u in enumerate(xs):
            d = float(depth[v, u])
            if not (np.isfinite(d) and d > 0.0 and d <= float(max_depth_m)):
                continue

            # Camera coords: x right, y up, z forward
            x_cam = (float(u) - cx) * d / fx
            y_cam = -(float(v) - cy) * d / fy
            z_cam = d
            p_s = np.array([x_cam, y_cam, z_cam], dtype=np.float64)

            # Sensor -> body -> world
            p_b = R_SB @ p_s + t_SB
            p_w = R_WB @ p_b + t_WB
            if center_on_pose:
                p_w = p_w - t_WB

            grid_i[yi, xi] = len(verts)
            verts.append([float(p_w[0]), float(p_w[1]), float(p_w[2])])
            uvs.append([float(u) / float(max(1, W - 1)), 1.0 - float(v) / float(max(1, H - 1))])
            dvals[yi, xi] = d

    faces: list[list[int]] = []
    # Triangulate quads, skipping discontinuities (depth jumps).
    for yi in range(len(ys) - 1):
        for xi in range(len(xs) - 1):
            i00 = int(grid_i[yi, xi])
            i10 = int(grid_i[yi, xi + 1])
            i01 = int(grid_i[yi + 1, xi])
            i11 = int(grid_i[yi + 1, xi + 1])
            if i00 < 0 or i10 < 0 or i01 < 0 or i11 < 0:
                continue
            depths = [float(dvals[yi, xi]), float(dvals[yi, xi + 1]), float(dvals[yi + 1, xi]), float(dvals[yi + 1, xi + 1])]
            if (max(depths) - min(depths)) > float(discontinuity_m):
                continue
            faces.append([i00, i10, i11])
            faces.append([i00, i11, i01])

    out_dir.mkdir(parents=True, exist_ok=True)
    # Previews
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is not None:
        cv2.imwrite(str(out_dir / "preview_rgb.png"), rgb)
    _write_preview_depth(depth, out_dir / "preview_depth.png")

    if not faces or not verts:
        raise RuntimeError("No mesh faces/vertices created (check depth validity/stride/max_depth)")

    # Textured mesh export (OBJ + MTL + texture PNG emitted by trimesh)
    texture_img = Image.open(rgb_path)
    material = trimesh.visual.texture.SimpleMaterial(image=texture_img)
    visual = trimesh.visual.texture.TextureVisuals(uv=np.asarray(uvs, dtype=np.float64), image=texture_img, material=material)
    mesh = trimesh.Trimesh(vertices=np.asarray(verts, dtype=np.float64), faces=np.asarray(faces, dtype=np.int64), visual=visual, process=False)
    obj_path = out_dir / "stage.obj"
    mesh.export(obj_path)

    bounds = mesh.bounds.astype(float).tolist()
    center = mesh.bounding_box.centroid.astype(float).tolist()
    extents = mesh.bounding_box.extents.astype(float).tolist()
    radius = float(np.linalg.norm(mesh.bounding_box.extents) * 0.5)

    (out_dir / "stage.object_config.json").write_text(
        json.dumps(
            {
                "render_asset": "stage.obj",
                "collision_asset": "stage.obj",
                "up": [0.0, 1.0, 0.0],
                "front": [0.0, 0.0, -1.0],
                "scale": [1.0, 1.0, 1.0],
                "margin": 0.03,
                "friction_coefficient": 0.5,
                "restitution_coefficient": 0.1,
                "units_to_meters": 1.0,
                "force_flat_shading": False,
                "mass": 1.0,
                "COM": [0.0, 0.0, 0.0],
                "use_bounding_box_for_collision": True,
                "join_collision_meshes": True,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    meta = {
        "timestamp_ns": int(t_ns),
        "pose_used_ns": int(pose.t_ns),
        "pose_position": [float(x) for x in pose.position],
        "pose_quat_wxyz": [float(x) for x in pose.quat_wxyz],
        "stride": int(stride),
        "max_depth_m": float(max_depth_m),
        "discontinuity_m": float(discontinuity_m),
        "center_on_pose": bool(center_on_pose),
        "mesh": {
            "num_vertices": int(mesh.vertices.shape[0]),
            "num_faces": int(mesh.faces.shape[0]),
            "bounds": bounds,
            "center": center,
            "extents": extents,
            "radius": radius,
        },
        "inputs": {
            "rgb": str(rgb_path),
            "depth_exr": str(depth_path),
            "sensor_yaml": str(sensor_yaml_path),
            "pose_csv": str(pose_csv_path),
        },
        "outputs": {"stage_obj": str(obj_path), "object_config": str(out_dir / "stage.object_config.json")},
    }
    (out_dir / "scene_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return meta


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rgb", type=str, required=True)
    ap.add_argument("--depth-exr", type=str, required=True)
    ap.add_argument("--sensor-yaml", type=str, required=True)
    ap.add_argument("--pose-csv", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-depth-m", type=float, default=80.0)
    ap.add_argument("--discontinuity-m", type=float, default=1.5)
    ap.add_argument("--center-on-pose", action="store_true", help="Translate mesh so pose position is at origin")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    meta = _build_mesh_from_frame(
        rgb_path=Path(args.rgb),
        depth_path=Path(args.depth_exr),
        sensor_yaml_path=Path(args.sensor_yaml),
        pose_csv_path=Path(args.pose_csv),
        out_dir=Path(args.out_dir),
        stride=int(args.stride),
        max_depth_m=float(args.max_depth_m),
        discontinuity_m=float(args.discontinuity_m),
        center_on_pose=bool(args.center_on_pose),
    )
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()

