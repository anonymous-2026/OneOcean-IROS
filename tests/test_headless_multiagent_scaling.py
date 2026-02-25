from __future__ import annotations

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


def test_multiagent_smoke(tmp_path: Path, monkeypatch) -> None:
    drift = _write_fake_drift_cache(tmp_path)
    for n in (2, 4, 8):
        out_dir = tmp_path / f"run_n{n}"
        monkeypatch.setattr(
            "sys.argv",
            [
                "prog",
                "--drift-npz",
                str(drift),
                "--task",
                "pollution_containment_multiagent",
                "--controller",
                "containment_ring",
                "--pollution-model",
                "gaussian",
                "--n-agents",
                str(n),
                "--seed",
                "0",
                "--dt",
                "1.0",
                "--max-steps",
                "6",
                "--out-dir",
                str(out_dir),
            ],
        )
        assert run_main() == 0

