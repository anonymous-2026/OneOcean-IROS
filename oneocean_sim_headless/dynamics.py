from __future__ import annotations

import math

import numpy as np


def wrap_pi(x: float) -> float:
    """Wrap angle to (-pi, pi]."""
    y = float((x + math.pi) % (2.0 * math.pi) - math.pi)
    # Map -pi to +pi for consistency.
    if abs(y + math.pi) < 1e-12:
        return math.pi
    return y


def rotmat_yaw_roll_pitch(*, yaw_y: float, roll_x: float, pitch_z: float) -> np.ndarray:
    """Rotation matrix R (body->world) for our headless convention.

    World axes:
      - x: east (meters)
      - y: depth (positive down, meters)
      - z: north (meters)

    Angles:
      - yaw_y: rotation about +y (down) axis
      - roll_x: rotation about +x axis
      - pitch_z: rotation about +z axis

    Composition (intrinsic body rotations) implemented as:
      R = R_y(yaw_y) @ R_x(roll_x) @ R_z(pitch_z)
    """
    cy = math.cos(float(yaw_y))
    sy = math.sin(float(yaw_y))
    cr = math.cos(float(roll_x))
    sr = math.sin(float(roll_x))
    cp = math.cos(float(pitch_z))
    sp = math.sin(float(pitch_z))

    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float64)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]], dtype=np.float64)
    Rz = np.array([[cp, -sp, 0.0], [sp, cp, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return Ry @ Rx @ Rz


def quat_from_axis_angle(axis_xyz: np.ndarray, angle_rad: float) -> np.ndarray:
    a = np.asarray(axis_xyz, dtype=np.float64).reshape(3)
    n = float(np.linalg.norm(a))
    if not np.isfinite(n) or n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    a = a / n
    ha = 0.5 * float(angle_rad)
    s = float(math.sin(ha))
    c = float(math.cos(ha))
    return np.array([a[0] * s, a[1] * s, a[2] * s, c], dtype=np.float64)


def quat_mul_xyzw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for xyzw quaternions (vector part first, scalar last)."""
    x1, y1, z1, w1 = [float(v) for v in np.asarray(q1, dtype=np.float64).reshape(4)]
    x2, y2, z2, w2 = [float(v) for v in np.asarray(q2, dtype=np.float64).reshape(4)]
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float64,
    )


def quat_normalize_xyzw(q: np.ndarray) -> np.ndarray:
    qq = np.asarray(q, dtype=np.float64).reshape(4)
    n = float(np.linalg.norm(qq))
    if not np.isfinite(n) or n < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return (qq / n).astype(np.float64)


def quat_from_yaw_roll_pitch(*, yaw_y: float, roll_x: float, pitch_z: float) -> np.ndarray:
    """Quaternion (xyzw) consistent with rotmat_yaw_roll_pitch composition."""
    qy = quat_from_axis_angle(np.array([0.0, 1.0, 0.0], dtype=np.float64), float(yaw_y))
    qx = quat_from_axis_angle(np.array([1.0, 0.0, 0.0], dtype=np.float64), float(roll_x))
    qz = quat_from_axis_angle(np.array([0.0, 0.0, 1.0], dtype=np.float64), float(pitch_z))
    return quat_normalize_xyzw(quat_mul_xyzw(quat_mul_xyzw(qy, qx), qz))

