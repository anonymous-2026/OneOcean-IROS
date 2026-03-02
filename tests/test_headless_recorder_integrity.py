from __future__ import annotations

from pathlib import Path

import numpy as np

from oneocean_sim_headless.cli.run import main as run_main
from oneocean_sim_headless.validators import validate_run_dir


def _write_fake_drift_cache(tmp: Path) -> Path:
    lat = np.linspace(42.0, 42.001, 4, dtype=np.float64)
    lon = np.linspace(-71.0, -70.999, 5, dtype=np.float64)
    u = np.zeros((lat.size, lon.size), dtype=np.float64)
    v = np.zeros((lat.size, lon.size), dtype=np.float64)
    out = tmp / "drift.npz"
    np.savez_compressed(out, latitude=lat, longitude=lon, u=u, v=v)
    return out


def test_headless_recording_validates(tmp_path: Path, monkeypatch) -> None:
    drift = _write_fake_drift_cache(tmp_path)
    out_dir = tmp_path / "run"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prog",
            "--drift-npz",
            str(drift),
            "--task",
            "go_to_goal_current",
            "--controller",
            "go_to_goal",
            "--pollution-model",
            "gaussian",
            "--n-agents",
            "2",
            "--seed",
            "0",
            "--dt",
            "1.0",
            "--dynamics-model",
            "6dof",
            "--constraint-mode",
            "off",
            "--max-steps",
            "8",
            "--out-dir",
            str(out_dir),
            "--validate",
        ],
    )
    code = run_main()
    assert code == 0
    res = validate_run_dir(out_dir)
    assert res.ok, res.reason
