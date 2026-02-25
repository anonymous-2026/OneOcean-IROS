from __future__ import annotations

from pathlib import Path

import numpy as np

from oneocean_sim_headless.drift_cache import load_drift_cache


def test_drift_cache_schema_missing_key(tmp_path: Path) -> None:
    lat = np.linspace(0.0, 1.0, 3)
    lon = np.linspace(0.0, 1.0, 4)
    v = np.zeros((lat.size, lon.size))
    bad = tmp_path / "bad.npz"
    # Missing key: 'u'
    np.savez_compressed(bad, latitude=lat, longitude=lon, v=v)

    try:
        load_drift_cache(bad)
        assert False, "expected load_drift_cache to fail"
    except KeyError as e:
        msg = str(e)
        assert "Expected npz keys" in msg
        assert "latitude" in msg and "longitude" in msg and "u" in msg and "v" in msg

