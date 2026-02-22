"""Utilities and CLI entrypoints for pollution simulation and visualization."""

from .runner import run_synthetic_diffusion_case
from .viz import analyze_nc_file, plot_3d_currents, plot_pollutant_diffusion, simulate_diffusion_from_dataset

__all__ = [
    "run_synthetic_diffusion_case",
    "analyze_nc_file",
    "plot_3d_currents",
    "plot_pollutant_diffusion",
    "simulate_diffusion_from_dataset",
]
