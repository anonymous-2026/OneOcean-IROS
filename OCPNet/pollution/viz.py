import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


def _as_path(path_like: Union[str, Path]) -> Path:
    path = Path(path_like)
    path.mkdir(parents=True, exist_ok=True)
    return path


def analyze_nc_file(nc_file_path: Union[str, Path]):
    import xarray as xr

    nc_file = Path(nc_file_path)
    if not nc_file.exists():
        raise FileNotFoundError(f"File not found: {nc_file}")

    with xr.open_dataset(nc_file) as dataset:
        stats_dict: Dict[str, Dict[str, float]] = {}
        for var in dataset.variables.keys():
            if var in {"time", "depth", "latitude", "longitude"}:
                continue
            values = dataset[var].values.flatten()
            values = values[~np.isnan(values)]
            if values.size == 0:
                continue
            stats_dict[var] = {
                "Min": float(np.min(values)),
                "Max": float(np.max(values)),
                "Mean": float(np.mean(values)),
                "Median": float(np.median(values)),
                "Std": float(np.std(values)),
            }

        time_info = None
        if "time" in dataset.variables and dataset["time"].size > 0:
            time_values = dataset["time"].values
            time_info = (time_values[0], time_values[-1], time_values[-1] - time_values[0])
    return stats_dict, time_info


def _open_dataset(dataset_or_path):
    import xarray as xr

    if isinstance(dataset_or_path, (str, Path)):
        return xr.open_dataset(dataset_or_path), True
    return dataset_or_path, False


def plot_3d_currents(
    dataset_or_path,
    output_dir: Union[str, Path],
    skip: int = 8,
    arrow_size: float = 0.06,
    arrow_height_offset: float = 5.0,
    arrow_alpha: float = 0.55,
    arrow_head_size: int = 10,
    time_index: int = 0,
    depth_index: int = 0,
) -> Dict[str, str]:
    output_path = _as_path(output_dir)
    dataset, should_close = _open_dataset(dataset_or_path)
    try:
        elevation = dataset["elevation"].values
        lats = dataset["latitude"].values
        lons = dataset["longitude"].values
        uo = dataset["uo"].isel(time=time_index, depth=depth_index).values
        vo = dataset["vo"].isel(time=time_index, depth=depth_index).values
        utotal = dataset["utotal"].isel(time=time_index, depth=depth_index).values
        vtotal = dataset["vtotal"].isel(time=time_index, depth=depth_index).values
    finally:
        if should_close:
            dataset.close()

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    def _render(speed_u, speed_v, color_speed, out_name: str, title: str):
        fig, axis = plt.subplots(figsize=(12, 10), subplot_kw={"projection": "3d"})
        cmap = plt.cm.viridis
        norm = mcolors.Normalize(vmin=float(np.nanmin(color_speed)), vmax=float(np.nanmax(color_speed)))
        axis.plot_surface(
            lon_grid,
            lat_grid,
            elevation,
            facecolors=cmap(norm(color_speed)),
            edgecolor="none",
            alpha=1.0,
            linewidth=0.0,
        )

        lon_sample = lon_grid[::skip, ::skip]
        lat_sample = lat_grid[::skip, ::skip]
        u_sample = speed_u[::skip, ::skip]
        v_sample = speed_v[::skip, ::skip]
        speed = np.sqrt(u_sample ** 2 + v_sample ** 2)
        safe = speed > 1e-8
        arrow_height = float(np.nanmax(elevation)) + arrow_height_offset
        axis.quiver(
            lon_sample[safe],
            lat_sample[safe],
            np.full(np.count_nonzero(safe), arrow_height),
            (u_sample[safe] / speed[safe]),
            (v_sample[safe] / speed[safe]),
            np.zeros(np.count_nonzero(safe)),
            color="black",
            alpha=arrow_alpha,
            length=arrow_size,
            normalize=True,
            arrow_length_ratio=0.3,
        )

        scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        scalar.set_array([])
        fig.colorbar(scalar, ax=axis, shrink=0.55, aspect=14, label="Speed (m/s)")
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_zlabel("Elevation (m)")
        axis.set_title(title)
        out_file = output_path / out_name
        fig.savefig(out_file, dpi=240, bbox_inches="tight")
        plt.close(fig)
        return str(out_file)

    base_speed = np.sqrt(uo ** 2 + vo ** 2)
    total_speed = np.sqrt(utotal ** 2 + vtotal ** 2)
    base_file = _render(uo, vo, base_speed, "base_current_3d.png", "Base Current (uo/vo) over Bathymetry")
    total_file = _render(utotal, vtotal, total_speed, "total_current_3d.png", "Total Current (utotal/vtotal) over Bathymetry")
    return {"base_current_3d": base_file, "total_current_3d": total_file}


def generate_synthetic_diffusion_series(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    days: int = 80,
    seed: int = 7,
) -> Sequence[np.ndarray]:
    rng = np.random.default_rng(seed)
    fields = []
    for day in range(1, days + 1):
        center_lon = float(np.mean(lon_grid) + 1.2 * np.sin(day / 9.0))
        center_lat = float(np.mean(lat_grid) + 0.9 * np.cos(day / 11.0))
        spread = 0.9 + 0.6 * np.sin(day / 13.0)
        dist_lon = lon_grid - center_lon
        dist_lat = lat_grid - center_lat
        field = np.exp(-((dist_lon ** 2 + dist_lat ** 2) / (2 * spread ** 2)))
        field += 0.03 * rng.normal(size=field.shape)
        field = np.clip(field, 0.0, None)
        max_v = np.max(field)
        fields.append(field / max_v if max_v > 0 else field)
    return fields


def plot_pollutant_diffusion(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    pollutant_data: Sequence[np.ndarray],
    days: Sequence[int],
    pollutant_name: str,
    pollutant_data_all_days: Optional[Sequence[np.ndarray]] = None,
    output_dir: Union[str, Path] = ".",
    prefix: str = "pollutant_diffusion",
    concentration_floor: float = 0.08,
) -> Dict[str, str]:
    output_path = _as_path(output_dir)
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "pollution_scale",
        [(0.0, "#0b2e57"), (0.25, "#2a71b8"), (0.5, "#46b2c9"), (0.75, "#f0b24b"), (1.0, "#d9483b")],
    )
    levels = np.linspace(concentration_floor, 1.0, 11)

    n = len(days)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4.4 * nrows))
    axes = np.array(axes).reshape(-1)
    contour = None
    for index, day in enumerate(days):
        axis = axes[index]
        pollutant = np.array(pollutant_data[index], dtype=float)
        pollutant[pollutant < concentration_floor] = np.nan
        contour = axis.contourf(lon_grid, lat_grid, pollutant, levels=levels, cmap=cmap, extend="max")
        axis.contour(lon_grid, lat_grid, pollutant, levels=levels, colors="white", linewidths=0.3, alpha=0.35)
        axis.set_title(f"Day {day}")
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_aspect("equal")
        axis.grid(alpha=0.25, linestyle="--")
    for index in range(n, len(axes)):
        fig.delaxes(axes[index])
    if contour is not None:
        colorbar = fig.colorbar(contour, ax=axes[:n], shrink=0.9, pad=0.02)
        colorbar.set_label("Relative concentration")
    fig.suptitle(pollutant_name)
    fig.subplots_adjust(left=0.06, right=0.93, bottom=0.08, top=0.9, wspace=0.22, hspace=0.28)
    panel_file = output_path / f"{prefix}_panel.png"
    fig.savefig(panel_file, dpi=260)
    plt.close(fig)

    outputs = {"panel_png": str(panel_file)}
    if pollutant_data_all_days:
        fig_anim, ax_anim = plt.subplots(figsize=(8, 6))
        base = np.array(pollutant_data_all_days[0], dtype=float)
        base[base < concentration_floor] = np.nan
        img = ax_anim.imshow(
            base,
            origin="lower",
            cmap=cmap,
            vmin=concentration_floor,
            vmax=1.0,
            extent=[np.min(lon_grid), np.max(lon_grid), np.min(lat_grid), np.max(lat_grid)],
            interpolation="nearest",
        )
        ax_anim.set_title("Day 1")
        ax_anim.set_xlabel("Longitude")
        ax_anim.set_ylabel("Latitude")
        ax_anim.grid(alpha=0.25, linestyle="--")
        fig_anim.colorbar(img, ax=ax_anim, label="Relative concentration")

        def update(frame: int):
            field = np.array(pollutant_data_all_days[frame], dtype=float)
            field[field < concentration_floor] = np.nan
            img.set_data(field)
            ax_anim.set_title(f"Day {frame + 1}")
            return [img]

        ani = animation.FuncAnimation(fig_anim, update, frames=len(pollutant_data_all_days), interval=120, blit=True)
        gif_file = output_path / f"{prefix}.gif"
        ani.save(gif_file, writer=animation.PillowWriter(fps=8))
        plt.close(fig_anim)
        outputs["gif"] = str(gif_file)

    return outputs


def simulate_diffusion_from_dataset(
    nc_path: Union[str, Path],
    output_dir: Union[str, Path],
    depth_index: int = 0,
    time_start: int = 0,
    time_count: int = 24,
    spatial_stride: int = 2,
    diffusion_coeff: float = 18.0,
    frame_seconds: float = 1800.0,
    substeps: int = 3,
    prefix: str = "dataset_diffusion",
) -> Dict[str, Union[str, float, int]]:
    import xarray as xr

    output_path = _as_path(output_dir)
    nc_file = Path(nc_path)
    if not nc_file.exists():
        raise FileNotFoundError(f"File not found: {nc_file}")

    with xr.open_dataset(nc_file) as ds:
        u_name = "utotal" if "utotal" in ds.variables else "uo"
        v_name = "vtotal" if "vtotal" in ds.variables else "vo"

        u_da = ds[u_name]
        v_da = ds[v_name]

        if "depth" in u_da.dims:
            u_da = u_da.isel(depth=depth_index)
        if "depth" in v_da.dims:
            v_da = v_da.isel(depth=depth_index)

        u_da = u_da.isel(time=slice(time_start, time_start + time_count))
        v_da = v_da.isel(time=slice(time_start, time_start + time_count))

        if "latitude" in u_da.dims and "longitude" in u_da.dims:
            u_da = u_da.isel(latitude=slice(None, None, spatial_stride), longitude=slice(None, None, spatial_stride))
            v_da = v_da.isel(latitude=slice(None, None, spatial_stride), longitude=slice(None, None, spatial_stride))
        else:
            raise ValueError("Expected latitude/longitude dimensions in current fields.")

        lats = u_da["latitude"].values
        lons = u_da["longitude"].values
        u_series = np.asarray(u_da.values, dtype=float)
        v_series = np.asarray(v_da.values, dtype=float)

    n_frames = min(u_series.shape[0], v_series.shape[0])
    if n_frames < 2:
        raise ValueError("Not enough time frames for dataset-driven diffusion simulation.")

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_span = max(float(np.max(lons) - np.min(lons)), 1e-6)
    lat_span = max(float(np.max(lats) - np.min(lats)), 1e-6)

    center_a = (float(np.min(lons) + 0.36 * lon_span), float(np.min(lats) + 0.45 * lat_span))
    center_b = (float(np.min(lons) + 0.68 * lon_span), float(np.min(lats) + 0.62 * lat_span))
    sigma_lon = max(lon_span * 0.08, 1e-6)
    sigma_lat = max(lat_span * 0.08, 1e-6)

    concentration = np.exp(
        -(((lon_grid - center_a[0]) ** 2) / (2.0 * sigma_lon**2) + ((lat_grid - center_a[1]) ** 2) / (2.0 * sigma_lat**2))
    )
    concentration += 0.75 * np.exp(
        -(((lon_grid - center_b[0]) ** 2) / (2.0 * sigma_lon**2) + ((lat_grid - center_b[1]) ** 2) / (2.0 * sigma_lat**2))
    )
    concentration = concentration / np.max(concentration)

    mean_lat = float(np.mean(lats))
    meters_per_deg_lat = 110_540.0
    meters_per_deg_lon = max(111_320.0 * np.cos(np.deg2rad(mean_lat)), 1e-6)
    dx_m = max(float(np.mean(np.diff(lons))) * meters_per_deg_lon, 1e-6)
    dy_m = max(float(np.mean(np.diff(lats))) * meters_per_deg_lat, 1e-6)
    dt = frame_seconds / max(1, substeps)

    frames = []
    for frame_idx in range(n_frames):
        u_frame = u_series[frame_idx]
        v_frame = v_series[frame_idx]
        u_frame = np.nan_to_num(u_frame, nan=0.0)
        v_frame = np.nan_to_num(v_frame, nan=0.0)

        for _ in range(max(1, substeps)):
            dc_dx = np.gradient(concentration, dx_m, axis=1)
            dc_dy = np.gradient(concentration, dy_m, axis=0)
            lap_x = np.gradient(np.gradient(concentration, dx_m, axis=1), dx_m, axis=1)
            lap_y = np.gradient(np.gradient(concentration, dy_m, axis=0), dy_m, axis=0)
            lap = lap_x + lap_y

            tendency = (-u_frame * dc_dx) + (-v_frame * dc_dy) + (diffusion_coeff * lap)
            concentration = concentration + dt * tendency
            concentration = np.clip(concentration, 0.0, None)
            max_value = np.max(concentration)
            if max_value > 0:
                concentration = concentration / max_value

        frames.append(concentration.copy())

    sample_indices = np.linspace(0, n_frames - 1, num=min(6, n_frames), dtype=int)
    sampled_days = [int(index + 1) for index in sample_indices]
    sampled_fields = [frames[index] for index in sample_indices]

    media = plot_pollutant_diffusion(
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        pollutant_data=sampled_fields,
        days=sampled_days,
        pollutant_name="Dataset-driven Microplastic Diffusion Proxy",
        pollutant_data_all_days=frames,
        output_dir=output_path,
        prefix=prefix,
    )
    media["frame_count"] = int(n_frames)
    media["u_variable"] = u_name
    media["v_variable"] = v_name
    media["spatial_stride"] = int(spatial_stride)
    media["depth_index"] = int(depth_index)
    media["diffusion_coeff"] = float(diffusion_coeff)
    return media
