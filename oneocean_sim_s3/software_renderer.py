from __future__ import annotations

from dataclasses import dataclass
from math import atan2, cos, sin, tan
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return vec.astype(np.float64)
    return (vec / norm).astype(np.float64)


@dataclass(frozen=True)
class CameraConfig:
    width: int = 640
    height: int = 480
    fovy_deg: float = 60.0
    near: float = 0.2
    far: float = 5000.0


@dataclass(frozen=True)
class CameraPose:
    eye_m: tuple[float, float, float]
    target_m: tuple[float, float, float]
    up: tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass(frozen=True)
class RenderVehicle:
    position_m: tuple[float, float, float]
    yaw_rad: float
    color_bgr: tuple[int, int, int]
    scale_m: float = 1.0
    mesh_path: str | None = None


@dataclass(frozen=True)
class RenderSphere:
    center_m: tuple[float, float, float]
    radius_m: float
    color_bgr: tuple[int, int, int]


_OBJ_CACHE: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def _compute_face_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    n = np.cross(v1 - v0, v2 - v0)
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    norm = np.where(norm > 1e-9, norm, 1.0)
    return (n / norm).astype(np.float64)


def load_obj_mesh(path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load a simple OBJ (v/f only) as triangle mesh with cached face normals."""
    key = str(Path(path).expanduser().resolve())
    cached = _OBJ_CACHE.get(key)
    if cached is not None:
        return cached

    vertices: list[list[float]] = []
    faces: list[list[int]] = []
    with open(key, "r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("v "):
                parts = line.split()
                if len(parts) < 4:
                    continue
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.split()[1:]
                if len(parts) < 3:
                    continue
                idx: list[int] = []
                for p in parts[:3]:
                    s = p.split("/")[0]
                    idx.append(int(s) - 1)
                faces.append(idx)

    if not vertices or not faces:
        raise ValueError(f"OBJ missing vertices/faces: {key}")

    v_arr = np.asarray(vertices, dtype=np.float64)
    f_arr = np.asarray(faces, dtype=np.int32)
    n_arr = _compute_face_normals(v_arr, f_arr)
    _OBJ_CACHE[key] = (v_arr, f_arr, n_arr)
    return v_arr, f_arr, n_arr


def _apply_fog(color_bgr: np.ndarray, fog_color_bgr: np.ndarray, fog_t: float) -> np.ndarray:
    fog_t = float(np.clip(fog_t, 0.0, 1.0))
    return (1.0 - fog_t) * color_bgr + fog_t * fog_color_bgr


def _underwater_fog_t(depth_m: float, fog_start_m: float, fog_end_m: float) -> float:
    if fog_end_m <= fog_start_m:
        return 0.0
    return float(np.clip((float(depth_m) - float(fog_start_m)) / (float(fog_end_m) - float(fog_start_m)), 0.0, 1.0))


def _camera_basis(pose: CameraPose) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    eye = np.asarray(pose.eye_m, dtype=np.float64)
    target = np.asarray(pose.target_m, dtype=np.float64)
    up = np.asarray(pose.up, dtype=np.float64)

    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, up))
    up2 = _normalize(np.cross(right, forward))
    return eye, right, up2, forward


def _project_points(
    points_world: np.ndarray,
    *,
    cam_eye: np.ndarray,
    cam_right: np.ndarray,
    cam_up: np.ndarray,
    cam_forward: np.ndarray,
    cam: CameraConfig,
) -> tuple[np.ndarray, np.ndarray]:
    rel = points_world - cam_eye[None, :]
    x = rel @ cam_right
    y = rel @ cam_up
    z = rel @ cam_forward

    fovy = float(cam.fovy_deg) * np.pi / 180.0
    fy = 0.5 * float(cam.height) / max(1e-6, tan(fovy / 2.0))
    fx = fy
    cx = 0.5 * float(cam.width)
    cy = 0.5 * float(cam.height)

    z_safe = np.where(z > 1e-6, z, 1e-6)
    u = (fx * (x / z_safe) + cx).astype(np.float64)
    v = (-fy * (y / z_safe) + cy).astype(np.float64)
    uv = np.stack([u, v], axis=1)
    return uv, z


def _vehicle_mesh(scale_m: float) -> tuple[np.ndarray, np.ndarray]:
    length = 2.6 * float(scale_m)
    width = 1.1 * float(scale_m)
    height = 0.7 * float(scale_m)

    nose = np.array([+0.5 * length, 0.0, 0.0], dtype=np.float64)
    tail = np.array([-0.5 * length, 0.0, 0.0], dtype=np.float64)
    left = np.array([0.0, +0.5 * width, 0.0], dtype=np.float64)
    right = np.array([0.0, -0.5 * width, 0.0], dtype=np.float64)
    top = np.array([0.0, 0.0, +0.5 * height], dtype=np.float64)
    bottom = np.array([0.0, 0.0, -0.5 * height], dtype=np.float64)

    vertices = np.stack([nose, tail, left, right, top, bottom], axis=0)
    faces = np.array(
        [
            [0, 2, 4],
            [0, 4, 3],
            [0, 3, 5],
            [0, 5, 2],
            [1, 4, 2],
            [1, 3, 4],
            [1, 5, 3],
            [1, 2, 5],
        ],
        dtype=np.int32,
    )
    return vertices, faces


def render_scene(
    *,
    terrain_vertices_m: np.ndarray,
    terrain_faces: np.ndarray,
    terrain_face_normals: np.ndarray,
    obstacles: Iterable[RenderSphere],
    vehicles: Iterable[RenderVehicle],
    camera: CameraConfig,
    pose: CameraPose,
    particles_xyz_m: np.ndarray | None = None,
    time_sec: float = 0.0,
    water_color_bgr: tuple[int, int, int] = (155, 105, 40),
) -> np.ndarray:
    height = int(camera.height)
    width = int(camera.width)
    img = np.zeros((height, width, 3), dtype=np.float64)
    water = np.asarray(water_color_bgr, dtype=np.float64)
    img[:, :, :] = water[None, None, :]

    # Brighter near the top; darker near the bottom (simple underwater depth cue).
    grad = np.linspace(1.10, 0.85, height, dtype=np.float64)[:, None, None]
    img = np.clip(img * grad, 0.0, 255.0).astype(np.uint8)

    cam_eye, cam_right, cam_up, cam_forward = _camera_basis(pose)
    uv, z_cam = _project_points(
        terrain_vertices_m,
        cam_eye=cam_eye,
        cam_right=cam_right,
        cam_up=cam_up,
        cam_forward=cam_forward,
        cam=camera,
    )

    face_uv = uv[terrain_faces]  # (M, 3, 2)
    face_z = z_cam[terrain_faces]  # (M, 3)
    face_depth = face_z.mean(axis=1)

    visible = np.all(face_z > float(camera.near), axis=1)
    visible &= np.all(face_uv[:, :, 0] > -0.3 * camera.width, axis=1)
    visible &= np.all(face_uv[:, :, 0] < 1.3 * camera.width, axis=1)
    visible &= np.all(face_uv[:, :, 1] > -0.3 * camera.height, axis=1)
    visible &= np.all(face_uv[:, :, 1] < 1.3 * camera.height, axis=1)

    idx = np.nonzero(visible)[0]
    if idx.size:
        idx = idx[np.argsort(face_depth[idx])[::-1]]

    light_dir = _normalize(np.array([0.22, 0.18, 1.0], dtype=np.float64))
    sand = np.array([120.0, 175.0, 200.0], dtype=np.float64)
    rock = np.array([70.0, 90.0, 110.0], dtype=np.float64)
    fog_color = np.array([205.0, 145.0, 75.0], dtype=np.float64)

    for fi in idx.tolist():
        poly = np.round(face_uv[fi]).astype(np.int32)
        if np.any(np.isnan(poly)):
            continue
        if poly.shape != (3, 2):
            continue
        normal = terrain_face_normals[fi]
        shade = float(np.clip(0.35 + 0.55 * max(0.0, float(normal @ light_dir)), 0.0, 1.0))
        depth = float(face_depth[fi])
        fog_t = _underwater_fog_t(depth_m=depth, fog_start_m=8.0, fog_end_m=115.0)

        tri = terrain_faces[fi]
        centroid = np.mean(terrain_vertices_m[tri], axis=0)
        nx = float(centroid[0])
        ny = float(centroid[1])

        # Procedural "seafloor material" texture + mild caustics.
        noise = 0.55 * sin(0.75 * nx + 0.22 * ny) + 0.45 * cos(0.68 * ny - 0.18 * nx)
        noise = 0.5 + 0.5 * float(np.clip(noise, -1.0, 1.0))
        slope = float(np.clip(1.0 - float(normal[2]), 0.0, 1.0))
        rock_t = float(np.clip(0.30 * noise + 0.95 * slope, 0.0, 1.0))

        caust = sin(2.4 * nx + 1.8 * ny + 0.8 * float(time_sec)) * sin(
            2.1 * nx - 2.0 * ny + 1.1 * float(time_sec)
        )
        caust = 0.5 + 0.5 * float(caust)
        caust_gain = 0.85 + 0.25 * caust

        base_color = ((1.0 - rock_t) * sand + rock_t * rock) * caust_gain
        color = base_color * shade
        color = _apply_fog(color, fog_color, fog_t)
        cv2.fillConvexPoly(img, poly, color=tuple(int(c) for c in np.clip(color, 0.0, 255.0)))

    for sphere in obstacles:
        center = np.asarray(sphere.center_m, dtype=np.float64)[None, :]
        center_uv, center_z = _project_points(
            center,
            cam_eye=cam_eye,
            cam_right=cam_right,
            cam_up=cam_up,
            cam_forward=cam_forward,
            cam=camera,
        )
        z = float(center_z[0])
        if z <= float(camera.near):
            continue
        fovy = float(camera.fovy_deg) * np.pi / 180.0
        f = 0.5 * float(camera.height) / max(1e-6, tan(fovy / 2.0))
        radius_px = float(f * float(sphere.radius_m) / z)
        if radius_px < 1.0:
            continue
        cx, cy = int(round(center_uv[0, 0])), int(round(center_uv[0, 1]))
        fog_t = _underwater_fog_t(depth_m=z, fog_start_m=6.0, fog_end_m=110.0)
        base_color = np.asarray(sphere.color_bgr, dtype=np.float64)
        color = _apply_fog(base_color, fog_color, fog_t)
        cv2.circle(img, (cx, cy), int(round(radius_px)), tuple(int(c) for c in color), thickness=-1, lineType=cv2.LINE_AA)
        hi = _apply_fog(base_color + np.array([40.0, 40.0, 40.0], dtype=np.float64), fog_color, fog_t)
        cv2.circle(
            img,
            (cx - int(0.25 * radius_px), cy - int(0.25 * radius_px)),
            max(1, int(round(0.35 * radius_px))),
            tuple(int(c) for c in np.clip(hi, 0.0, 255.0)),
            thickness=-1,
            lineType=cv2.LINE_AA,
        )

    for vehicle in vehicles:
        if vehicle.mesh_path:
            v_local, v_faces, v_normals_local = load_obj_mesh(vehicle.mesh_path)
        else:
            v_local, v_faces = _vehicle_mesh(1.0)
            v_normals_local = _compute_face_normals(v_local, v_faces)

        yaw = float(vehicle.yaw_rad)
        rot = np.array(
            [
                [cos(yaw), -sin(yaw), 0.0],
                [sin(yaw), cos(yaw), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        pos = np.asarray(vehicle.position_m, dtype=np.float64)
        verts = (v_local * float(vehicle.scale_m)) @ rot.T + pos[None, :]

        vu, vz = _project_points(
            verts,
            cam_eye=cam_eye,
            cam_right=cam_right,
            cam_up=cam_up,
            cam_forward=cam_forward,
            cam=camera,
        )
        face_uv = vu[v_faces]
        face_z = vz[v_faces]
        face_depth = face_z.mean(axis=1)
        vidx = np.nonzero(np.all(face_z > float(camera.near), axis=1))[0]
        if vidx.size:
            vidx = vidx[np.argsort(face_depth[vidx])[::-1]]
        for fi in vidx.tolist():
            poly = np.round(face_uv[fi]).astype(np.int32)
            depth = float(face_depth[fi])
            fog_t = _underwater_fog_t(depth_m=depth, fog_start_m=4.0, fog_end_m=95.0)
            normal = (rot @ v_normals_local[fi]).astype(np.float64)
            shade = float(np.clip(0.45 + 0.5 * max(0.0, float(normal @ light_dir)), 0.0, 1.0))
            base = np.asarray(vehicle.color_bgr, dtype=np.float64)
            color = _apply_fog(base * shade, fog_color, fog_t)
            cv2.fillConvexPoly(img, poly, tuple(int(c) for c in np.clip(color, 0.0, 255.0)), lineType=cv2.LINE_AA)

    # Particulate matter / suspended sediment (parallax cue).
    if particles_xyz_m is not None and len(particles_xyz_m):
        pts = np.asarray(particles_xyz_m, dtype=np.float64)
        puv, pz = _project_points(
            pts,
            cam_eye=cam_eye,
            cam_right=cam_right,
            cam_up=cam_up,
            cam_forward=cam_forward,
            cam=camera,
        )
        order = np.argsort(pz)[::-1]  # far -> near
        for pi in order.tolist():
            z = float(pz[pi])
            if z <= float(camera.near):
                continue
            u = int(round(float(puv[pi, 0])))
            v = int(round(float(puv[pi, 1])))
            if not (0 <= u < width and 0 <= v < height):
                continue
            fog_t = _underwater_fog_t(depth_m=z, fog_start_m=3.0, fog_end_m=85.0)
            strength = float(0.85 * (1.0 - fog_t) + 0.15)
            color = _apply_fog(np.array([220.0, 235.0, 245.0], dtype=np.float64) * strength, fog_color, fog_t)
            radius = max(1, int(round(0.6 + 2.6 * (1.0 - fog_t))))
            cv2.circle(
                img,
                (u, v),
                radius,
                tuple(int(c) for c in np.clip(color, 0.0, 255.0)),
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    # Soft haze pass (underwater "volumetric-ish" cue).
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=1.2, sigmaY=1.2)
    img = cv2.addWeighted(img, 0.80, blur, 0.20, 0.0)

    return img


def yaw_from_velocity(vx: float, vy: float, default: float = 0.0) -> float:
    if abs(vx) < 1e-6 and abs(vy) < 1e-6:
        return float(default)
    return float(atan2(vy, vx))
