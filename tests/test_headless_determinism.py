from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from oneocean_sim_headless.cli.run import main as run_main


def _write_fake_drift_cache(tmp: Path) -> Path:
    lat = np.linspace(42.0, 42.001, 4, dtype=np.float64)
    lon = np.linspace(-71.0, -70.999, 5, dtype=np.float64)
    u = np.zeros((lat.size, lon.size), dtype=np.float64)
    v = np.zeros((lat.size, lon.size), dtype=np.float64)
    out = tmp / "drift.npz"
    np.savez_compressed(out, latitude=lat, longitude=lon, u=u, v=v)
    return out


def _read_pose(run_dir: Path, agent: int) -> list[list[str]]:
    path = run_dir / "agents" / f"agent_{agent:03d}" / "pose_groundtruth" / "data.csv"
    with path.open("r", encoding="utf-8", newline="") as fp:
        r = csv.reader(fp)
        header = next(r)
        assert header[:4] == ["t", "x", "y", "z"]
        return [row for row in r]


def test_headless_determinism_same_seed(tmp_path: Path, monkeypatch) -> None:
    drift = _write_fake_drift_cache(tmp_path)
    out1 = tmp_path / "r1"
    out2 = tmp_path / "r2"

    for out in (out1, out2):
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--drift-npz",
                str(drift),
                "--task",
                "station_keeping",
                "--controller",
                "station_keep",
                "--pollution-model",
                "gaussian",
                "--n-agents",
                "2",
                "--seed",
                "123",
                "--dt",
                "1.0",
                "--max-steps",
                "10",
                "--out-dir",
                str(out),
            ],
        )
        assert run_main() == 0

    p1 = _read_pose(out1, agent=0)
    p2 = _read_pose(out2, agent=0)
    assert p1[:6] == p2[:6]

