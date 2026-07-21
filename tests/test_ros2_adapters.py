from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROS_PACKAGE = Path(__file__).resolve().parents[1] / "integrations" / "ros2" / "oneocean_ros"
sys.path.insert(0, str(ROS_PACKAGE))

from oneocean_ros.frames import core_quaternion_to_enu, core_vector_to_enu, enu_quaternion_to_core, enu_vector_to_core
from oneocean_ros.messages import commands_enu_to_core, snapshots_from_observation


def test_ros2_vector_frame_roundtrip() -> None:
    core = np.array([[2.0, 3.0, 5.0], [-1.0, 0.5, 7.0]], dtype=np.float64)
    enu = core_vector_to_enu(core)
    np.testing.assert_array_equal(enu, np.array([[2.0, 5.0, -3.0], [-1.0, 7.0, -0.5]]))
    np.testing.assert_array_equal(enu_vector_to_core(enu), core)


def test_ros2_quaternion_frame_roundtrip() -> None:
    core = np.array([0.2, -0.3, 0.1, 0.9], dtype=np.float64)
    core /= np.linalg.norm(core)
    restored = enu_quaternion_to_core(core_quaternion_to_enu(core))
    if float(np.dot(core, restored)) < 0.0:
        restored = -restored
    np.testing.assert_allclose(restored, core, atol=1e-12)


def test_ros2_observation_and_command_conversion() -> None:
    observation = {
        "positions_xyz": np.array([[1.0, 2.0, 3.0]], dtype=np.float64),
        "quaternions_xyzw": np.array([[0.0, 0.0, 0.0, 1.0]], dtype=np.float64),
        "body_velocity_uvw_pqr": np.array([[0.5, 0.1, 0.2, 0.01, 0.02, 0.03]], dtype=np.float64),
        "currents_xyz": np.array([[0.4, 0.0, -0.2]], dtype=np.float64),
        "pollution_probe": np.array([0.7], dtype=np.float64),
        "elevation_m": np.array([-42.0], dtype=np.float64),
        "land_mask": np.array([0.0], dtype=np.float64),
        "latitude": np.array([42.1], dtype=np.float64),
        "longitude": np.array([-70.2], dtype=np.float64),
    }
    snapshot = snapshots_from_observation(observation)[0]
    np.testing.assert_array_equal(snapshot.position_enu_m, np.array([1.0, 3.0, -2.0]))
    np.testing.assert_array_equal(snapshot.current_enu_mps, np.array([0.4, -0.2, 0.0]))
    assert snapshot.bathymetry_depth_m == 42.0
    command_core = commands_enu_to_core(np.array([[1.0, 2.0, 3.0]], dtype=np.float64), 1)
    np.testing.assert_array_equal(command_core, np.array([[1.0, -3.0, 2.0]]))
