from __future__ import annotations

from pathlib import Path

import numpy as np

from oneocean_sim_headless.controllers import preset_controller
from oneocean_sim_headless.env import EnvConfig, HeadlessOceanEnv
from oneocean_sim_headless.tasks import preset_task


def _write_drift_cache_with_constant_current(tmp: Path, *, u_x: float) -> Path:
    lat = np.linspace(42.0, 42.002, 6, dtype=np.float64)
    lon = np.linspace(-71.0, -70.998, 7, dtype=np.float64)
    u = np.full((lat.size, lon.size), float(u_x), dtype=np.float64)
    v = np.zeros((lat.size, lon.size), dtype=np.float64)
    out = tmp / "drift_const.npz"
    np.savez_compressed(out, latitude=lat, longitude=lon, u=u, v=v)
    return out


def test_6dof_stationkeeping_bounded_under_current(tmp_path: Path) -> None:
    drift = _write_drift_cache_with_constant_current(tmp_path, u_x=0.25)
    out_dir = tmp_path / "run"

    cfg = EnvConfig(
        drift_cache_npz=str(drift),
        pollution_model="gaussian",
        dt_s=1.0,
        dynamics_model="6dof",
        constraint_mode="off",
        bathy_mode="off",
    )
    env = HeadlessOceanEnv(cfg, out_dir=out_dir, seed=0, n_agents=2)
    task = preset_task(kind="station_keeping", difficulty="easy")
    ctrl = preset_controller(kind="station_keep", max_speed_mps=cfg.max_speed_mps)
    env.reset(task=task, controller=ctrl)

    # Make station-keeping explicit: hold each agent at its initial position.
    init = env.positions_xyz
    env.task_state.goal_xyz = init[0].copy()  # broadcast goal (good enough for boundedness check)

    max_d = 0.0
    for _ in range(40):
        done, _info = env.step()
        pos = env.positions_xyz
        d = float(np.linalg.norm(pos[0] - init[0]))
        max_d = max(max_d, d)
        if done:
            break
    env.close()

    # Under a mild current and bounded-speed controller, drift should stay bounded.
    assert np.isfinite(max_d)
    assert max_d < 30.0

