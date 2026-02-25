import math
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import colorsys


_LEGACY_LAND_COLORS = [(0.0, "#2e4536"), (0.5, "#4e5e3c"), (1.0, "#a69176")]
_LEGACY_POLLUTANT_COLORS = [
    (0.0, "#b3e6ff"),
    (0.2, "#9ad1f0"),
    (0.4, "#80bccc"),
    (0.6, "#f0e68c"),
    (0.8, "#ff9999"),
    (1.0, "#ff0000"),
]


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

    def _desaturate(hex_color: str, sat_scale: float = 0.75, light_scale: float = 1.0) -> Tuple[float, float, float]:
        hex_color = hex_color.lstrip("#")
        r = int(hex_color[0:2], 16) / 255.0
        g = int(hex_color[2:4], 16) / 255.0
        b = int(hex_color[4:6], 16) / 255.0
        h, l, s = colorsys.rgb_to_hls(r, g, b)
        s = max(0.0, min(1.0, s * sat_scale))
        l = max(0.0, min(1.0, l * light_scale))
        r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
        return r2, g2, b2

    plume_like_cmap = mcolors.LinearSegmentedColormap.from_list(
        "plume_like_current",
        [
            (0.0, _desaturate("#0b2e57", 0.8, 0.95)),
            (0.25, _desaturate("#2a71b8", 0.75, 0.98)),
            (0.5, _desaturate("#46b2c9", 0.7, 1.0)),
            (0.75, _desaturate("#f0b24b", 0.7, 1.0)),
            (1.0, _desaturate("#d9483b", 0.75, 1.0)),
        ],
    )

    views = [(25, 45), (32, 125), (25, 215)]

    def _render(speed_u, speed_v, color_speed, out_name: str, title: str):
        fig, axis = plt.subplots(figsize=(12, 10), subplot_kw={"projection": "3d"})
        norm = mcolors.Normalize(vmin=float(np.nanmin(color_speed)), vmax=float(np.nanmax(color_speed)))
        cmap = plt.cm.viridis
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
            arrow_length_ratio=0.55,
            linewidth=1.1,
        )

        scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        scalar.set_array([])
        fig.subplots_adjust(right=0.86)
        cax = fig.add_axes([0.88, 0.18, 0.02, 0.62])
        fig.colorbar(scalar, cax=cax, label="Speed (m/s)")
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_zlabel("Elevation (m)")
        out_file = output_path / out_name
        fig.savefig(out_file, dpi=240, bbox_inches="tight")
        eps_file = out_file.with_suffix(".eps")
        fig.savefig(eps_file, format="eps", bbox_inches="tight")
        plt.close(fig)
        return str(out_file), str(eps_file)

    def _render_multi_view(speed_u, speed_v, color_speed, stem: str, cmap_style: str):
        outputs = {}
        for elev, azim in views:
            fig, axis = plt.subplots(figsize=(12, 10), subplot_kw={"projection": "3d"})
            axis.view_init(elev=elev, azim=azim)
            norm = mcolors.Normalize(vmin=float(np.nanmin(color_speed)), vmax=float(np.nanmax(color_speed)))
            cmap = plume_like_cmap if cmap_style == "plume" else plt.cm.viridis
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
                arrow_length_ratio=0.7,
                linewidth=1.2,
            )

            scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            scalar.set_array([])
            fig.subplots_adjust(right=0.86)
            cax = fig.add_axes([0.88, 0.18, 0.02, 0.62])
            fig.colorbar(scalar, cax=cax, label="Speed (m/s)")
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            axis.set_zlabel("Elevation (m)")

            out_name = f"{stem}_{cmap_style}_el{elev:02d}_az{azim:03d}.png"
            out_file = output_path / out_name
            fig.savefig(out_file, dpi=240, bbox_inches="tight")
            eps_file = out_file.with_suffix(".eps")
            fig.savefig(eps_file, format="eps", bbox_inches="tight")
            plt.close(fig)
            outputs[out_name] = str(out_file)
            outputs[out_name.replace(".png", ".eps")] = str(eps_file)
        return outputs

    base_speed = np.sqrt(uo ** 2 + vo ** 2)
    total_speed = np.sqrt(utotal ** 2 + vtotal ** 2)
    outputs = {}
    outputs.update(_render_multi_view(uo, vo, base_speed, "base_current_3d", "viridis"))
    outputs.update(_render_multi_view(uo, vo, base_speed, "base_current_3d", "plume"))
    outputs.update(_render_multi_view(utotal, vtotal, total_speed, "total_current_3d", "viridis"))
    outputs.update(_render_multi_view(utotal, vtotal, total_speed, "total_current_3d", "plume"))
    return outputs


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
    concentration_floor: float = 0.2,
    basemap_elevation: Optional[np.ndarray] = None,
    basemap_land_mask: Optional[np.ndarray] = None,
) -> Dict[str, str]:
    output_path = _as_path(output_dir)
    land_cmap = mcolors.LinearSegmentedColormap.from_list("legacy_land", _LEGACY_LAND_COLORS)
    pollutant_cmap = mcolors.LinearSegmentedColormap.from_list("legacy_pollutant", _LEGACY_POLLUTANT_COLORS)
    pollutant_cmap.set_bad((0, 0, 0, 0))
    levels = np.linspace(concentration_floor, 1.0, 9)

    def _format_axes(axis):
        # Cartopy is not available in the current runtime, so we use explicit degree ticks.
        x_min, x_max = float(np.min(lon_grid)), float(np.max(lon_grid))
        y_min, y_max = float(np.min(lat_grid)), float(np.max(lat_grid))
        x_ticks = np.linspace(x_min, x_max, num=7)
        y_ticks = np.linspace(y_min, y_max, num=7)
        axis.set_xticks(x_ticks)
        axis.set_yticks(y_ticks)

        def _fmt_lon(value, _pos=None):
            suffix = "E" if value >= 0 else "W"
            return f"{abs(value):.0f}°{suffix}"

        def _fmt_lat(value, _pos=None):
            suffix = "N" if value >= 0 else "S"
            return f"{abs(value):.0f}°{suffix}"

        axis.set_xticklabels([_fmt_lon(v) for v in x_ticks], rotation=0)
        axis.set_yticklabels([_fmt_lat(v) for v in y_ticks], rotation=0)
        axis.grid(alpha=0.35, linestyle="--", linewidth=0.6)
        axis.set_xlim(x_min, x_max)
        axis.set_ylim(y_min, y_max)

    def _draw_basemap(axis):
        axis.set_facecolor("#000c3f")
        if basemap_elevation is None:
            return

        elevation = np.asarray(basemap_elevation, dtype=float)
        if basemap_land_mask is not None:
            land = np.asarray(basemap_land_mask, dtype=float) > 0.5
        else:
            land = elevation >= 0

        land_height = np.where(land, elevation, np.nan)
        ocean_depth = np.where(~land, elevation, np.nan)

        if np.any(np.isfinite(ocean_depth)):
            ocean_cmap = mcolors.LinearSegmentedColormap.from_list(
                "ocean_bathy",
                [(0.0, "#00122b"), (0.35, "#00315f"), (0.7, "#0a4b7e"), (1.0, "#1b6fa6")],
            )
            axis.pcolormesh(
                lon_grid,
                lat_grid,
                ocean_depth,
                cmap=ocean_cmap,
                shading="auto",
                alpha=0.9,
            )

        if np.any(np.isfinite(land_height)):
            axis.pcolormesh(
                lon_grid,
                lat_grid,
                land_height,
                cmap=land_cmap,
                shading="auto",
                alpha=0.95,
            )

    n = len(days)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 4.4 * nrows))
    axes = np.array(axes).reshape(-1)
    contour = None
    for index, day in enumerate(days):
        axis = axes[index]
        _draw_basemap(axis)
        pollutant = np.array(pollutant_data[index], dtype=float)
        pollutant[pollutant < concentration_floor] = np.nan
        if basemap_land_mask is not None:
            pollutant[np.asarray(basemap_land_mask, dtype=float) > 0.5] = np.nan
        contour = axis.contourf(lon_grid, lat_grid, pollutant, levels=levels, cmap=pollutant_cmap, extend="max")
        axis.contour(lon_grid, lat_grid, pollutant, levels=levels, colors="#aa66f5", linewidths=0.5, alpha=0.6)
        axis.set_xlabel("Longitude")
        axis.set_ylabel("Latitude")
        axis.set_aspect("equal")
        _format_axes(axis)
    for index in range(n, len(axes)):
        fig.delaxes(axes[index])
    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.08, top=0.95, wspace=0.22, hspace=0.28)
    if contour is not None:
        cax = fig.add_axes([0.88, 0.18, 0.02, 0.62])
        colorbar = fig.colorbar(contour, cax=cax)
        colorbar.set_label("Relative concentration")
    panel_file = output_path / f"{prefix}_panel.png"
    fig.savefig(panel_file, dpi=260)
    panel_eps = panel_file.with_suffix(".eps")
    fig.savefig(panel_eps, format="eps")
    plt.close(fig)

    outputs = {"panel_png": str(panel_file), "panel_eps": str(panel_eps)}
    if pollutant_data_all_days:
        fig_anim, ax_anim = plt.subplots(figsize=(8, 6))
        _draw_basemap(ax_anim)
        base = np.array(pollutant_data_all_days[0], dtype=float)
        base[base < concentration_floor] = np.nan
        if basemap_land_mask is not None:
            base[np.asarray(basemap_land_mask, dtype=float) > 0.5] = np.nan
        extent = [float(np.min(lon_grid)), float(np.max(lon_grid)), float(np.min(lat_grid)), float(np.max(lat_grid))]
        img = ax_anim.imshow(
            np.ma.masked_invalid(base),
            origin="lower",
            cmap=pollutant_cmap,
            vmin=concentration_floor,
            vmax=1.0,
            extent=extent,
            interpolation="nearest",
        )
        ax_anim.set_xlabel("Longitude")
        ax_anim.set_ylabel("Latitude")
        _format_axes(ax_anim)
        fig_anim.subplots_adjust(right=0.86)
        cax = fig_anim.add_axes([0.88, 0.18, 0.02, 0.62])
        fig_anim.colorbar(img, cax=cax, label="Relative concentration")

        def update(frame: int):
            field = np.array(pollutant_data_all_days[frame], dtype=float)
            field[field < concentration_floor] = np.nan
            if basemap_land_mask is not None:
                field[np.asarray(basemap_land_mask, dtype=float) > 0.5] = np.nan
            img.set_data(np.ma.masked_invalid(field))
            return [img]

        ani = animation.FuncAnimation(fig_anim, update, frames=len(pollutant_data_all_days), interval=140, blit=True)
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
    auto_coast: bool = True,
    coast_halfspan_deg: float = 3.0,
) -> Dict[str, Union[str, float, int]]:
    import xarray as xr

    output_path = _as_path(output_dir)
    nc_file = Path(nc_path)
    if not nc_file.exists():
        raise FileNotFoundError(f"File not found: {nc_file}")

    with xr.open_dataset(nc_file) as ds:
        u_name = "utotal" if "utotal" in ds.variables else "uo"
        v_name = "vtotal" if "vtotal" in ds.variables else "vo"

        ds_use = ds
        if auto_coast and "land_mask" in ds.variables:
            lm = ds["land_mask"]
            search_stride = max(4, spatial_stride)
            lm_s = lm.isel(latitude=slice(None, None, search_stride), longitude=slice(None, None, search_stride))
            lm_sv = np.asarray(lm_s.values, dtype=float)
            land = lm_sv > 0.5
            if np.any(land) and np.any(~land):
                # Find an ocean cell adjacent to land (coastline proxy).
                ocean = ~land
                coast_ocean = ocean.copy()
                coast_ocean[:-1, :] &= land[1:, :]
                coast_ocean[1:, :] |= (ocean[1:, :] & land[:-1, :])
                coast_ocean[:, :-1] |= (ocean[:, :-1] & land[:, 1:])
                coast_ocean[:, 1:] |= (ocean[:, 1:] & land[:, :-1])
                candidates = np.argwhere(coast_ocean)
                if candidates.size > 0:
                    ci, cj = candidates[len(candidates) // 2]
                    lat0 = float(lm_s["latitude"].values[ci])
                    lon0 = float(lm_s["longitude"].values[cj])

                    lat_min = lat0 - coast_halfspan_deg
                    lat_max = lat0 + coast_halfspan_deg
                    lon_min = lon0 - coast_halfspan_deg
                    lon_max = lon0 + coast_halfspan_deg

                    lats = ds["latitude"].values
                    lons = ds["longitude"].values
                    lat_inc = bool(lats[0] < lats[-1])
                    lon_inc = bool(lons[0] < lons[-1])
                    lat_slice = slice(lat_min, lat_max) if lat_inc else slice(lat_max, lat_min)
                    lon_slice = slice(lon_min, lon_max) if lon_inc else slice(lon_max, lon_min)
                    ds_use = ds.sel(latitude=lat_slice, longitude=lon_slice)

        u_da = ds_use[u_name]
        v_da = ds_use[v_name]

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

        basemap_elevation = None
        basemap_land_mask = None
        if "elevation" in ds_use.variables:
            elev = ds_use["elevation"]
            elev = elev.isel(latitude=slice(None, None, spatial_stride), longitude=slice(None, None, spatial_stride))
            basemap_elevation = np.asarray(elev.values, dtype=float)
        if "land_mask" in ds_use.variables:
            lm = ds_use["land_mask"]
            lm = lm.isel(latitude=slice(None, None, spatial_stride), longitude=slice(None, None, spatial_stride))
            basemap_land_mask = np.asarray(lm.values, dtype=float)

    n_frames = min(u_series.shape[0], v_series.shape[0])
    if n_frames < 2:
        raise ValueError("Not enough time frames for dataset-driven diffusion simulation.")

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_span = max(float(np.max(lons) - np.min(lons)), 1e-6)
    lat_span = max(float(np.max(lats) - np.min(lats)), 1e-6)

    center_a = (float(np.min(lons) + 0.36 * lon_span), float(np.min(lats) + 0.45 * lat_span))
    center_b = (float(np.min(lons) + 0.68 * lon_span), float(np.min(lats) + 0.62 * lat_span))
    sigma_lon = max(lon_span * 0.12, 1e-6)
    sigma_lat = max(lat_span * 0.12, 1e-6)

    concentration = np.exp(
        -(((lon_grid - center_a[0]) ** 2) / (2.0 * sigma_lon**2) + ((lat_grid - center_a[1]) ** 2) / (2.0 * sigma_lat**2))
    )
    concentration += 0.75 * np.exp(
        -(((lon_grid - center_b[0]) ** 2) / (2.0 * sigma_lon**2) + ((lat_grid - center_b[1]) ** 2) / (2.0 * sigma_lat**2))
    )
    concentration = concentration / np.max(concentration)
    if basemap_land_mask is not None:
        concentration[np.asarray(basemap_land_mask, dtype=float) > 0.5] = 0.0

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
            if basemap_land_mask is not None:
                concentration[np.asarray(basemap_land_mask, dtype=float) > 0.5] = 0.0
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
        basemap_elevation=basemap_elevation,
        basemap_land_mask=basemap_land_mask,
    )
    media["frame_count"] = int(n_frames)
    media["u_variable"] = u_name
    media["v_variable"] = v_name
    media["spatial_stride"] = int(spatial_stride)
    media["depth_index"] = int(depth_index)
    media["diffusion_coeff"] = float(diffusion_coeff)
    return media
