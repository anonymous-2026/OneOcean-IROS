from __future__ import annotations

from pathlib import Path

import numpy as np

from benchmark_core.pollution import GaussianPlumeConfig, GaussianPlumeField, OCPNetConfig, OCPNetPollutionField


def test_gaussian_agent_sink_decreases_mass() -> None:
    field = GaussianPlumeField(GaussianPlumeConfig())
    bounds = (np.zeros(3, dtype=np.float64), np.full(3, 100.0, dtype=np.float64))
    field.reset(np.random.default_rng(4), bounds)
    before = field.mass_fraction()
    field.apply_agent_sink(field.center_xyz[None, :], radius_m=1.0, strength_per_s=0.2, dt_s=1.5)
    after = field.mass_fraction()
    assert 0.0 <= after < before


def test_ocpnet_agent_sink_does_not_increase_concentration(tmp_path: Path) -> None:
    latitude = np.linspace(42.0, 42.01, 4, dtype=np.float64)
    longitude = np.linspace(-71.0, -70.99, 5, dtype=np.float64)
    shape = (latitude.size, longitude.size)
    field = OCPNetPollutionField(
        OCPNetConfig(grid_resolution=(6, 6, 4), sink_radius_m=4.0, sink_strength_per_s=0.15),
        domain_size_m=(30.0, 30.0, 12.0),
        drift_u_latlon=np.zeros(shape, dtype=np.float64),
        drift_v_latlon=np.zeros(shape, dtype=np.float64),
        latitude=latitude,
        longitude=longitude,
        output_dir=tmp_path,
    )
    field.reset(
        np.random.default_rng(3),
        (np.zeros(3, dtype=np.float64), np.array([30.0, 12.0, 30.0], dtype=np.float64)),
    )
    concentration = np.ones((6, 6, 4), dtype=np.float64)
    field.model.pollutant_fields[field.pollutant].set_concentration(field.pollutant, concentration)
    before = float(np.sum(field.model.pollutant_fields[field.pollutant].get_concentration(field.pollutant)))
    field.apply_agent_sink(np.array([[15.0, 6.0, 15.0]], dtype=np.float64))
    after = float(np.sum(field.model.pollutant_fields[field.pollutant].get_concentration(field.pollutant)))
    assert 0.0 <= after < before
