import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from OCPNet.PollutionModel3D.src.model import PollutionModel3D


def _create_velocity_field(grid_shape: Tuple[int, int, int]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    nx, ny, nz = grid_shape
    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    z = np.linspace(0.0, 1.0, nz)
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")

    u = 0.06 * np.sin(2.0 * np.pi * xx) * np.cos(2.0 * np.pi * yy)
    v = -0.06 * np.cos(2.0 * np.pi * xx) * np.sin(2.0 * np.pi * yy)
    w = 0.004 * np.sin(np.pi * zz)
    return u, v, w


def _create_environment_fields(grid_shape: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    nx, ny, nz = grid_shape
    z = np.linspace(0.0, 1.0, nz)
    zz = np.tile(z, (nx, ny, 1))
    return {
        "temperature": 292.0 + 3.0 * np.exp(-1.7 * zz),
        "pH": 7.7 - 0.4 * zz,
        "DO": 7.8 - 1.8 * zz,
        "light_intensity": 900.0 * np.exp(-2.6 * zz),
        "wave_velocity": 0.08 * np.exp(-2.1 * zz),
        "salinity": 34.2 + 0.5 * zz,
    }


def run_synthetic_diffusion_case(
    output_dir: Path,
    nx: int = 24,
    ny: int = 24,
    nz: int = 12,
    steps: int = 40,
    time_step: float = 20.0,
) -> Dict[str, float]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = PollutionModel3D(
        domain_size=(240.0, 240.0, 60.0),
        grid_resolution=(nx, ny, nz),
        time_step=time_step,
        output_dir=output_dir,
    )

    u, v, w = _create_velocity_field((nx, ny, nz))
    model.set_velocity_field(u, v, w)
    for field_name, field_value in _create_environment_fields((nx, ny, nz)).items():
        model.set_environmental_field(field_name, field_value)

    pollutant_name = "microplastic"
    model.add_pollutant(
        name=pollutant_name,
        initial_concentration=0.005,
        molecular_weight=1.0,
        decay_rate=8e-7,
        diffusion_coefficient=8e-8,
    )

    model.add_source(
        type="point",
        pollutant=pollutant_name,
        position=(120.0, 120.0, 6.0),
        emission_rate=0.015,
        time_function=lambda t: 1.0 + 0.2 * np.sin(2 * np.pi * t / 1800.0),
    )

    model.set_output_parameters(
        output_fields=[pollutant_name],
        output_interval=max(1.0, steps * time_step / 5.0),
        visualization_fields=[pollutant_name],
        visualization_interval=max(1.0, steps * time_step / 5.0),
        statistics_fields=[pollutant_name],
        statistics_interval=max(1.0, steps * time_step / 5.0),
    )

    end_time = steps * time_step
    model.run(end_time=end_time, progress_interval=max(1.0, end_time))

    final_field = model.get_field(pollutant_name)
    summary = {
        "pollutant": pollutant_name,
        "time_step": float(time_step),
        "steps": int(steps),
        "end_time": float(end_time),
        "grid_resolution": [int(nx), int(ny), int(nz)],
        "final_min": float(np.min(final_field)),
        "final_max": float(np.max(final_field)),
        "final_mean": float(np.mean(final_field)),
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    return summary

