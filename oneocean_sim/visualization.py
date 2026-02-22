from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .data import CombinedDataset


def render_trajectory_with_currents(
    ds: CombinedDataset,
    trajectory: list[dict[str, float]],
    time_index: int,
    depth_index: int,
    include_tides: bool,
    output_path: Path,
    grid_stride: int = 12,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    longitudes = np.asarray([row["longitude"] for row in trajectory], dtype=float)
    latitudes = np.asarray([row["latitude"] for row in trajectory], dtype=float)

    u_grid, v_grid = ds.uv_grid(
        time_index=time_index,
        depth_index=depth_index,
        include_tides=include_tides,
    )
    lon_grid = ds.longitude_values
    lat_grid = ds.latitude_values

    plt.figure(figsize=(10, 8))
    plt.quiver(
        lon_grid[::grid_stride],
        lat_grid[::grid_stride],
        u_grid[::grid_stride, ::grid_stride],
        v_grid[::grid_stride, ::grid_stride],
        alpha=0.45,
        color="#1f77b4",
    )
    plt.plot(longitudes, latitudes, linewidth=2.0, color="#d62728", label="Trajectory")
    plt.scatter([longitudes[0]], [latitudes[0]], color="#2ca02c", s=55, label="Start")
    plt.scatter([longitudes[-1]], [latitudes[-1]], color="#9467bd", s=55, label="End")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.title("OneOcean S1 Trajectory with Current Field")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path
