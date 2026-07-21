from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from benchmark_core.environment_contract import validate_environment_product


xr = pytest.importorskip("xarray")


def _write_product(
    path: Path,
    *,
    current_units: str = "m s-1",
    include_mask: bool = True,
    transpose_northward: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    latitude = np.linspace(32.0, 32.2, 3, dtype=np.float64)
    longitude = np.linspace(-66.2, -66.0, 4, dtype=np.float64)
    time = np.array([0.0, 24.0], dtype=np.float64)
    depth = np.array([0.5, 10.0], dtype=np.float64)
    shape = (time.size, depth.size, latitude.size, longitude.size)
    u = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100.0
    v = -u
    elevation = -50.0 * np.ones((latitude.size, longitude.size), dtype=np.float32)
    northward = (
        (("time", "depth", "longitude", "latitude"), v.transpose(0, 1, 3, 2), {"units": current_units})
        if transpose_northward
        else (("time", "depth", "latitude", "longitude"), v, {"units": current_units})
    )
    variables = {
        "uo": (("time", "depth", "latitude", "longitude"), u, {"units": current_units}),
        "vo": northward,
        "elevation": (("latitude", "longitude"), elevation, {"units": "m", "positive": "up"}),
    }
    if include_mask:
        variables["land_mask"] = (
            ("latitude", "longitude"),
            np.zeros_like(elevation, dtype=np.uint8),
            {"units": "1", "flag_meanings": "valid_water invalid_terrain_or_nodata"},
        )
    dataset = xr.Dataset(
        data_vars=variables,
        coords={
            "time": ("time", time, {"units": "hours since 2025-01-01"}),
            "depth": ("depth", depth, {"units": "m", "positive": "down"}),
            "latitude": ("latitude", latitude, {"units": "degrees_north"}),
            "longitude": ("longitude", longitude, {"units": "degrees_east"}),
        },
        attrs={
            "source": "test fixture",
            "generated_at_utc": "2026-07-21 00:00:00Z",
            "interpolation_method": "none",
        },
    )
    dataset.to_netcdf(path)
    dataset.close()
    return u, v


def test_environment_product_contract_passes(tmp_path: Path) -> None:
    product = tmp_path / "environment.nc"
    _write_product(product)
    report = validate_environment_product(product)
    assert report.ok, report.to_dict()
    assert report.status == "PASS"
    assert report.canonical_mapping["eastward_current"] == "uo"
    assert report.canonical_mapping["land_mask"] == "land_mask"


def test_environment_product_missing_mask_fails(tmp_path: Path) -> None:
    product = tmp_path / "missing_mask.nc"
    _write_product(product, include_mask=False)
    report = validate_environment_product(product)
    assert not report.ok
    assert any(issue.code == "missing_variable" and "land_mask" in issue.message for issue in report.issues)


def test_environment_product_strict_units_fail(tmp_path: Path) -> None:
    product = tmp_path / "bad_units.nc"
    _write_product(product, current_units="knots")
    report = validate_environment_product(product, strict_units=True)
    assert not report.ok
    assert sum(issue.code == "unexpected_current_units" for issue in report.issues) == 2


def test_environment_product_current_shape_mismatch_fails(tmp_path: Path) -> None:
    product = tmp_path / "shape_mismatch.nc"
    _write_product(product, transpose_northward=True)
    report = validate_environment_product(product)
    assert not report.ok
    assert any(issue.code == "current_shape_mismatch" for issue in report.issues)


def test_environment_product_drift_roundtrip(tmp_path: Path) -> None:
    product = tmp_path / "environment.nc"
    u, v = _write_product(product)
    latitude = np.linspace(32.0, 32.2, 3, dtype=np.float64)
    longitude = np.linspace(-66.2, -66.0, 4, dtype=np.float64)
    cache = tmp_path / "drift.npz"
    np.savez_compressed(cache, latitude=latitude, longitude=longitude, u=u[1, 1], v=v[1, 1])
    report = validate_environment_product(product, drift_npz=cache, time_index=1, depth_index=1)
    assert report.ok, report.to_dict()
    assert report.roundtrip is not None
    assert report.roundtrip["coordinate_match"]
    assert report.roundtrip["u_max_abs_error"] == 0.0
