from __future__ import annotations

from pathlib import Path


def test_h3_current_series_npz_schema(tmp_path: Path) -> None:
    import numpy as np

    npz = tmp_path / "currents.npz"
    time_ns = np.array([0, 1, 2], dtype="int64")
    depth_m = np.array([0.5, 5.0], dtype="float32")
    uo = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype="float32")
    vo = np.array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]], dtype="float32")
    np.savez_compressed(
        npz,
        time_ns=time_ns,
        depth_m=depth_m,
        uo=uo,
        vo=vo,
        latitude=32.5,
        longitude=-66.0,
        source_dataset="dummy.nc",
    )

    from tracks.h3_oceangym.run_task_suite import _CurrentSeries

    s = _CurrentSeries(str(npz), depth_m=0.4)
    assert s.t_len == 3
    assert s.d_len == 2
    assert abs(s.depth_selected_m - 0.5) < 1e-6
    u, v = s.uv_at(time_idx=0)
    assert abs(u - 1.0) < 1e-6
    assert abs(v - 0.1) < 1e-6


def test_h3_difficulty_presets() -> None:
    from tracks.h3_oceangym.run_task_suite import SuiteCfg, _cfg_with_difficulty

    base = SuiteCfg()
    easy = _cfg_with_difficulty(base, "easy")
    hard = _cfg_with_difficulty(base, "hard")
    assert easy.nav_goal_dist_m <= base.nav_goal_dist_m
    assert hard.nav_goal_dist_m >= base.nav_goal_dist_m
    assert easy.plume_sigma_m >= base.plume_sigma_m
    assert hard.plume_sigma_m <= base.plume_sigma_m

