"""Backward-compatible wrappers for pollution visualization utilities."""

from OCPNet.pollution.viz import (
    analyze_nc_file,
    generate_synthetic_diffusion_series,
    plot_3d_currents,
    plot_pollutant_diffusion,
)

__all__ = [
    "analyze_nc_file",
    "generate_synthetic_diffusion_series",
    "plot_3d_currents",
    "plot_pollutant_diffusion",
]

