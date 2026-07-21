from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmark_core.controllers import preset_controller
from benchmark_core.env import EnvConfig, HeadlessOceanEnv
from benchmark_core.tasks import preset_task


def _write_zero_current_cache(path: Path) -> Path:
    latitude = np.linspace(42.0, 42.002, 6, dtype=np.float64)
    longitude = np.linspace(-71.0, -70.998, 7, dtype=np.float64)
    shape = (latitude.size, longitude.size)
    np.savez_compressed(
        path,
        latitude=latitude,
        longitude=longitude,
        u=np.zeros(shape, dtype=np.float64),
        v=np.zeros(shape, dtype=np.float64),
    )
    return path


def _make_environment(cache: Path, output: Path) -> HeadlessOceanEnv:
    config = EnvConfig(
        drift_cache_npz=str(cache),
        pollution_model="gaussian",
        dynamics_model="kinematic",
        constraint_mode="off",
        bathy_mode="off",
    )
    environment = HeadlessOceanEnv(config, out_dir=output, seed=17, n_agents=2)
    environment.reset(
        task=preset_task(kind="station_keeping", difficulty="easy"),
        controller=preset_controller(kind="station_keep", max_speed_mps=config.max_speed_mps),
    )
    return environment


def test_none_action_preserves_controller_behavior(tmp_path: Path) -> None:
    cache = _write_zero_current_cache(tmp_path / "drift.npz")
    first = _make_environment(cache, tmp_path / "first")
    second = _make_environment(cache, tmp_path / "second")
    try:
        done_first, info_first = first.step()
        done_second, info_second = second.step(None)
        np.testing.assert_array_equal(first.positions_xyz, second.positions_xyz)
        assert done_first == done_second
        assert info_first == info_second
        assert info_first["action_source"] == "controller"
    finally:
        first.close()
        second.close()


def test_external_action_uses_shared_clipping_and_observation(tmp_path: Path) -> None:
    cache = _write_zero_current_cache(tmp_path / "drift.npz")
    environment = _make_environment(cache, tmp_path / "external")
    try:
        initial = environment.positions_xyz
        direction = 1.0 if initial[0, 0] <= 0.5 * environment.tile_size_x_m else -1.0
        actions = np.zeros((2, 3), dtype=np.float64)
        actions[0, 0] = 100.0 * direction
        _, info = environment.step(actions)
        displacement = environment.positions_xyz[0] - initial[0]
        assert info["action_source"] == "external"
        assert displacement[0] == pytest.approx(direction * environment.cfg.max_speed_mps * environment.cfg.dt_s)
        assert displacement[1] == pytest.approx(0.0)
        assert displacement[2] == pytest.approx(0.0)
        observation = environment.observe()
        assert observation["positions_xyz"].shape == (2, 3)
        assert observation["currents_xyz"].shape == (2, 3)
        assert observation["quaternions_xyzw"].shape == (2, 4)
        observation["positions_xyz"][0, 0] += 1000.0
        assert observation["positions_xyz"][0, 0] != environment.positions_xyz[0, 0]
    finally:
        environment.close()


def test_external_action_rejects_invalid_payload(tmp_path: Path) -> None:
    cache = _write_zero_current_cache(tmp_path / "drift.npz")
    environment = _make_environment(cache, tmp_path / "invalid")
    try:
        with pytest.raises(ValueError, match="shape"):
            environment.step(np.zeros((1, 3), dtype=np.float64))
        invalid = np.zeros((2, 3), dtype=np.float64)
        invalid[0, 0] = np.nan
        with pytest.raises(ValueError, match="finite"):
            environment.step(invalid)
    finally:
        environment.close()
