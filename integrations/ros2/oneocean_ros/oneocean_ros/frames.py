from __future__ import annotations

import numpy as np


CORE_TO_ENU = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, -1.0, 0.0],
    ],
    dtype=np.float64,
)


def core_vector_to_enu(vector_xyz: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector_xyz, dtype=np.float64)
    return np.einsum("ij,...j->...i", CORE_TO_ENU, vector)


def enu_vector_to_core(vector_xyz: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector_xyz, dtype=np.float64)
    return np.einsum("ij,...j->...i", CORE_TO_ENU.T, vector)


def _quaternion_to_matrix(quaternion_xyzw: np.ndarray) -> np.ndarray:
    x_value, y_value, z_value, w_value = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm([x_value, y_value, z_value, w_value]))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.eye(3, dtype=np.float64)
    x_value, y_value, z_value, w_value = [value / norm for value in (x_value, y_value, z_value, w_value)]
    return np.array(
        [
            [1.0 - 2.0 * (y_value * y_value + z_value * z_value), 2.0 * (x_value * y_value - z_value * w_value), 2.0 * (x_value * z_value + y_value * w_value)],
            [2.0 * (x_value * y_value + z_value * w_value), 1.0 - 2.0 * (x_value * x_value + z_value * z_value), 2.0 * (y_value * z_value - x_value * w_value)],
            [2.0 * (x_value * z_value - y_value * w_value), 2.0 * (y_value * z_value + x_value * w_value), 1.0 - 2.0 * (x_value * x_value + y_value * y_value)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = 2.0 * np.sqrt(trace + 1.0)
        quaternion = np.array(
            [
                (rotation[2, 1] - rotation[1, 2]) / scale,
                (rotation[0, 2] - rotation[2, 0]) / scale,
                (rotation[1, 0] - rotation[0, 1]) / scale,
                0.25 * scale,
            ],
            dtype=np.float64,
        )
    else:
        diagonal_index = int(np.argmax(np.diag(rotation)))
        if diagonal_index == 0:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]))
            quaternion = np.array([0.25 * scale, (rotation[0, 1] + rotation[1, 0]) / scale, (rotation[0, 2] + rotation[2, 0]) / scale, (rotation[2, 1] - rotation[1, 2]) / scale], dtype=np.float64)
        elif diagonal_index == 1:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]))
            quaternion = np.array([(rotation[0, 1] + rotation[1, 0]) / scale, 0.25 * scale, (rotation[1, 2] + rotation[2, 1]) / scale, (rotation[0, 2] - rotation[2, 0]) / scale], dtype=np.float64)
        else:
            scale = 2.0 * np.sqrt(max(0.0, 1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]))
            quaternion = np.array([(rotation[0, 2] + rotation[2, 0]) / scale, (rotation[1, 2] + rotation[2, 1]) / scale, 0.25 * scale, (rotation[1, 0] - rotation[0, 1]) / scale], dtype=np.float64)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    quaternion = quaternion / norm
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    return quaternion


def core_quaternion_to_enu(quaternion_xyzw: np.ndarray) -> np.ndarray:
    core_rotation = _quaternion_to_matrix(quaternion_xyzw)
    return _matrix_to_quaternion(CORE_TO_ENU @ core_rotation @ CORE_TO_ENU.T)


def enu_quaternion_to_core(quaternion_xyzw: np.ndarray) -> np.ndarray:
    enu_rotation = _quaternion_to_matrix(quaternion_xyzw)
    return _matrix_to_quaternion(CORE_TO_ENU.T @ enu_rotation @ CORE_TO_ENU)
