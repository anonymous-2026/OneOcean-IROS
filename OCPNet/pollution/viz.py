import json
import math
import colorsys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np


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


def _open_dataset(dataset_or_path):
    import xarray as xr

    if isinstance(dataset_or_path, (str, Path)):
        return xr.open_dataset(dataset_or_path), True
    return dataset_or_path, False


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


def _desaturate_hex(hex_color: str, sat_scale: float = 0.75, light_scale: float = 1.0) -> Tuple[float, float, float]:
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16) / 255.0
    g = int(hex_color[2:4], 16) / 255.0
    b = int(hex_color[4:6], 16) / 255.0
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    s = max(0.0, min(1.0, s * sat_scale))
    l = max(0.0, min(1.0, l * light_scale))
    r2, g2, b2 = colorsys.hls_to_rgb(h, l, s)
    return r2, g2, b2


def _direction_rgb(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    angle = np.arctan2(v, u)
    hue = (angle + math.pi) / (2.0 * math.pi)
    sat = np.full_like(hue, 0.85, dtype=float)
    val = np.full_like(hue, 0.95, dtype=float)
    rgb = np.empty((hue.size, 3), dtype=float)
    for i, (h, s, va) in enumerate(zip(hue.flat, sat.flat, val.flat)):
        rgb[i, :] = colorsys.hsv_to_rgb(float(h), float(s), float(va))
    return rgb


def plot_3d_currents(
    dataset_or_path,
    output_dir: Union[str, Path],
    skip: int = 15,
    arrow_size: float = 0.06,
    arrow_height_offset: float = 5.0,
    arrow_alpha: float = 0.65,
    time_index: int = 0,
    depth_index: int = 0,
    styles: Sequence[str] = ("diffusion",),
    arrow_color_mode: str = "direction",
    views: Sequence[Tuple[int, int]] = ((25, 45), (32, 125), (25, 215)),
) -> Dict[str, str]:
    output_path = _as_path(output_dir)
    dataset, should_close = _open_dataset(dataset_or_path)
    try:
        elevation = dataset["elevation"].values
        lats = dataset["latitude"].values
        lons = dataset["longitude"].values
        uo = dataset["uo"].isel(time=time_index, depth=depth_index).values
        vo = dataset["vo"].isel(time=time_index, depth=depth_index).values
        utotal = dataset["utotal"].isel(time=time_index, depth=depth_index).values if "utotal" in dataset.variables else None
        vtotal = dataset["vtotal"].isel(time=time_index, depth=depth_index).values if "vtotal" in dataset.variables else None
    finally:
        if should_close:
            dataset.close()

    lon_grid, lat_grid = np.meshgrid(lons, lats)

    plume_like_cmap = mcolors.LinearSegmentedColormap.from_list(
        "plume_like_current",
        [
            (0.0, _desaturate_hex("#0b2e57", 0.8, 0.95)),
            (0.25, _desaturate_hex("#2a71b8", 0.75, 0.98)),
            (0.5, _desaturate_hex("#46b2c9", 0.7, 1.0)),
            (0.75, _desaturate_hex("#f0b24b", 0.7, 1.0)),
            (1.0, _desaturate_hex("#d9483b", 0.75, 1.0)),
        ],
    )
    diffusion_like_cmap = mcolors.LinearSegmentedColormap.from_list(
        "diffusion_like_current",
        [
            (0.0, _desaturate_hex("#b3e6ff", 0.55, 0.75)),
            (0.2, _desaturate_hex("#9ad1f0", 0.55, 0.8)),
            (0.4, _desaturate_hex("#80bccc", 0.55, 0.9)),
            (0.6, _desaturate_hex("#f0e68c", 0.6, 0.95)),
            (0.8, _desaturate_hex("#ff9999", 0.6, 0.95)),
            (1.0, _desaturate_hex("#ff0000", 0.65, 0.95)),
        ],
    )

    def _cmap(style: str):
        s = str(style).strip().lower()
        if s == "plume":
            return plume_like_cmap
        if s == "diffusion":
            return diffusion_like_cmap
        return plt.cm.viridis

    def _render_multi_view(speed_u: np.ndarray, speed_v: np.ndarray, color_speed: np.ndarray, stem: str, style: str):
        outputs: Dict[str, str] = {}
        cmap = _cmap(style)
        norm = mcolors.Normalize(vmin=float(np.nanmin(color_speed)), vmax=float(np.nanmax(color_speed)))

        for elev, azim in views:
            fig, axis = plt.subplots(figsize=(12, 10), subplot_kw={"projection": "3d"})
            axis.view_init(elev=elev, azim=azim)
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
            speed = np.sqrt(u_sample**2 + v_sample**2)
            safe = speed > 1e-8
            arrow_height = float(np.nanmax(elevation)) + arrow_height_offset
            arrow_len = float(arrow_size) * 2.0
            ratio = 0.65

            x = lon_sample[safe]
            y = lat_sample[safe]
            z = np.full(np.count_nonzero(safe), arrow_height)
            u = (u_sample[safe] / speed[safe]).astype(float)
            v = (v_sample[safe] / speed[safe]).astype(float)
            w = np.zeros(np.count_nonzero(safe), dtype=float)

            if str(arrow_color_mode).strip().lower() == "direction":
                rgb = _direction_rgb(u, v)
                core_colors = [tuple(c) for c in rgb]
            else:
                core_colors = "white"

            axis.quiver(
                x,
                y,
                z,
                u,
                v,
                w,
                color="black",
                alpha=min(1.0, float(arrow_alpha) + 0.25),
                length=arrow_len,
                normalize=True,
                arrow_length_ratio=ratio,
                linewidth=2.4,
            )
            axis.quiver(
                x,
                y,
                z,
                u,
                v,
                w,
                color=core_colors,
                alpha=min(1.0, float(arrow_alpha) + 0.35),
                length=arrow_len,
                normalize=True,
                arrow_length_ratio=ratio,
                linewidth=1.3,
            )
            tip_x = x + u * arrow_len
            tip_y = y + v * arrow_len
            axis.scatter(
                tip_x,
                tip_y,
                z,
                s=14.0,
                marker="D",
                c=core_colors,
                alpha=min(1.0, float(arrow_alpha) + 0.4),
                depthshade=False,
            )

            scalar = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            scalar.set_array([])
            fig.subplots_adjust(right=0.86)
            cax = fig.add_axes([0.88, 0.18, 0.02, 0.62])
            fig.colorbar(scalar, cax=cax, label="Speed (m/s)")
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            axis.set_zlabel("Elevation (m)")

            out_name = f"{stem}_{style}_el{elev:02d}_az{azim:03d}.png"
            out_file = output_path / out_name
            fig.savefig(out_file, dpi=240, bbox_inches="tight")
            eps_file = out_file.with_suffix(".eps")
            fig.savefig(eps_file, format="eps", bbox_inches="tight")
            plt.close(fig)
            outputs[out_name] = str(out_file)
            outputs[out_name.replace(".png", ".eps")] = str(eps_file)
        return outputs

    outputs: Dict[str, str] = {}
    base_speed = np.sqrt(uo**2 + vo**2)
    for style in styles:
        outputs.update(_render_multi_view(uo, vo, base_speed, "base_current_3d", str(style).strip().lower()))

    if utotal is not None and vtotal is not None:
        total_speed = np.sqrt(utotal**2 + vtotal**2)
        for style in styles:
            outputs.update(_render_multi_view(utotal, vtotal, total_speed, "total_current_3d", str(style).strip().lower()))

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
        field = np.exp(-((dist_lon**2 + dist_lat**2) / (2 * spread**2)))
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
    panel_labels: Optional[Sequence[str]] = None,
    frame_labels: Optional[Sequence[str]] = None,
    output_dir: Union[str, Path] = ".",
    prefix: str = "pollutant_diffusion",
    concentration_floor: float = 0.2,
    basemap_elevation: Optional[np.ndarray] = None,
    basemap_land_mask: Optional[np.ndarray] = None,
    use_cartopy: bool = True,
    basemap_style: str = "stock",
    smooth_passes: int = 2,
    upsample: int = 3,
    wspace: float = 0.18,
    hspace: float = 0.12,
) -> Dict[str, str]:
    output_path = _as_path(output_dir)
    land_cmap = mcolors.LinearSegmentedColormap.from_list("legacy_land", _LEGACY_LAND_COLORS)
    pollutant_cmap = mcolors.LinearSegmentedColormap.from_list("legacy_pollutant", _LEGACY_POLLUTANT_COLORS)
    pollutant_cmap.set_bad((0, 0, 0, 0))
    levels = np.linspace(concentration_floor, 1.0, 9)

    if panel_labels is None:
        panel_labels = [f"Day {day}" for day in days]

    cartopy_ctx = None
    if use_cartopy:
        try:
            import cartopy.crs as ccrs  # type: ignore
            import cartopy.feature as cfeature  # type: ignore

            cartopy_ctx = (ccrs, cfeature)
        except Exception:
            cartopy_ctx = None

    def _smooth_nan_field(field: np.ndarray, invalid: np.ndarray, passes: int = 1) -> np.ndarray:
        if passes <= 0:
            return field
        kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=float)
        kernel = kernel / float(np.sum(kernel))
        out = np.array(field, dtype=float)
        valid = ~invalid
        out[invalid] = 0.0
        weight = valid.astype(float)

        def conv2(a: np.ndarray) -> np.ndarray:
            pad = np.pad(a, ((1, 1), (1, 1)), mode="edge")
            return (
                kernel[0, 0] * pad[:-2, :-2]
                + kernel[0, 1] * pad[:-2, 1:-1]
                + kernel[0, 2] * pad[:-2, 2:]
                + kernel[1, 0] * pad[1:-1, :-2]
                + kernel[1, 1] * pad[1:-1, 1:-1]
                + kernel[1, 2] * pad[1:-1, 2:]
                + kernel[2, 0] * pad[2:, :-2]
                + kernel[2, 1] * pad[2:, 1:-1]
                + kernel[2, 2] * pad[2:, 2:]
            )

        for _ in range(passes):
            num = conv2(out)
            den = conv2(weight)
            out = np.divide(num, np.clip(den, 1e-8, None))
        out[invalid] = np.nan
        return out

    def _upsample_grid_and_field(
        field: np.ndarray,
        lon_grid_in: np.ndarray,
        lat_grid_in: np.ndarray,
        factor: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if factor <= 1:
            return lon_grid_in, lat_grid_in, field
        lons_1d = lon_grid_in[0, :]
        lats_1d = lat_grid_in[:, 0]
        lon_hi = np.linspace(float(lons_1d.min()), float(lons_1d.max()), num=len(lons_1d) * factor)
        lat_hi = np.linspace(float(lats_1d.min()), float(lats_1d.max()), num=len(lats_1d) * factor)
        lon_hi_grid, lat_hi_grid = np.meshgrid(lon_hi, lat_hi)

        field_in = np.array(field, dtype=float)
        nan_mask = ~np.isfinite(field_in)
        if np.any(nan_mask):
            filled = field_in.copy()
            filled[nan_mask] = 0.0
            field_in = filled

        tmp = np.empty((field_in.shape[0], lon_hi.shape[0]), dtype=float)
        for i in range(field_in.shape[0]):
            tmp[i, :] = np.interp(lon_hi, lons_1d, field_in[i, :])
        out = np.empty((lat_hi.shape[0], lon_hi.shape[0]), dtype=float)
        for j in range(tmp.shape[1]):
            out[:, j] = np.interp(lat_hi, lats_1d, tmp[:, j])

        if np.any(nan_mask):
            out[out <= 0.0] = np.nan
        return lon_hi_grid, lat_hi_grid, out

    def _format_axes(axis):
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
            axis.pcolormesh(lon_grid, lat_grid, ocean_depth, cmap=ocean_cmap, shading="auto", alpha=0.9)

        if np.any(np.isfinite(land_height)):
            axis.pcolormesh(lon_grid, lat_grid, land_height, cmap=land_cmap, shading="auto", alpha=0.95)

    def _draw_cartopy_basemap(axis):
        if cartopy_ctx is None:
            return
        ccrs, cfeature = cartopy_ctx
        x_min, x_max = float(np.min(lon_grid)), float(np.max(lon_grid))
        y_min, y_max = float(np.min(lat_grid)), float(np.max(lat_grid))
        axis.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.PlateCarree())
        style = str(basemap_style or "").strip().lower()
        if style in {"stock", "stock_img", "stockimg"}:
            try:
                axis.stock_img()
            except Exception:
                axis.set_facecolor("#000c3f")
                axis.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#000c3f", zorder=0)
                axis.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#4e5e3c", zorder=1)
        else:
            axis.set_facecolor("#000c3f")
            axis.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#000c3f", zorder=0)
            axis.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#4e5e3c", zorder=1)
        axis.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.55, edgecolor="black", zorder=10)
        gl = axis.gridlines(draw_labels=True, linewidth=0.5, color="white", alpha=0.28, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

    def _land_mask_fill(axis):
        if basemap_land_mask is None:
            return
        land = (np.asarray(basemap_land_mask, dtype=float) > 0.5).astype(float)
        land = np.where(land > 0.5, 1.0, np.nan)
        fill_cmap = mcolors.ListedColormap(["#4e5e3c"])
        if cartopy_ctx is None:
            axis.pcolormesh(lon_grid, lat_grid, land, cmap=fill_cmap, shading="auto", alpha=0.35)
        else:
            ccrs, _cfeature = cartopy_ctx
            axis.pcolormesh(
                lon_grid,
                lat_grid,
                land,
                cmap=fill_cmap,
                shading="auto",
                alpha=0.35,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

    def _land_mask_boundary(axis, zorder: int = 29):
        if basemap_land_mask is None:
            return
        land = (np.asarray(basemap_land_mask, dtype=float) > 0.5).astype(float)
        if cartopy_ctx is None:
            axis.contour(lon_grid, lat_grid, land, levels=[0.5], colors="black", linewidths=0.75, alpha=0.9)
        else:
            ccrs, _cfeature = cartopy_ctx
            axis.contour(
                lon_grid,
                lat_grid,
                land,
                levels=[0.5],
                colors="black",
                linewidths=0.75,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                zorder=zorder,
            )

    def _stamp_time(axis, label: str):
        axis.text(
            0.02,
            0.98,
            label,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            color="white",
            zorder=30,
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
        )

    n = len(days)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    if cartopy_ctx is None:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.0 * ncols, 4.4 * nrows))
        axes = np.array(axes).reshape(-1)
    else:
        ccrs, _cfeature = cartopy_ctx
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.2 * ncols, 4.6 * nrows),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        axes = np.array(axes).reshape(-1)

    contour = None
    for index, day in enumerate(days):
        axis = axes[index]
        if cartopy_ctx is None:
            _draw_basemap(axis)
        else:
            _draw_cartopy_basemap(axis)
        _land_mask_fill(axis)

        pollutant = np.array(pollutant_data[index], dtype=float)
        pollutant[pollutant < concentration_floor] = np.nan
        if basemap_land_mask is not None:
            pollutant[np.asarray(basemap_land_mask, dtype=float) > 0.5] = np.nan
        invalid = ~np.isfinite(pollutant)
        pollutant = _smooth_nan_field(pollutant, invalid=invalid, passes=max(0, int(smooth_passes)))
        lon_plot, lat_plot, field_plot = _upsample_grid_and_field(pollutant, lon_grid, lat_grid, int(upsample))

        if cartopy_ctx is None:
            contour = axis.contourf(lon_plot, lat_plot, field_plot, levels=levels, cmap=pollutant_cmap, extend="max")
            axis.contour(lon_plot, lat_plot, field_plot, levels=levels, colors="#aa66f5", linewidths=0.45, alpha=0.6)
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            axis.set_aspect("equal")
            _format_axes(axis)
        else:
            ccrs, _cfeature = cartopy_ctx
            transform = ccrs.PlateCarree()
            contour = axis.contourf(
                lon_plot,
                lat_plot,
                field_plot,
                levels=levels,
                cmap=pollutant_cmap,
                extend="max",
                transform=transform,
                zorder=20,
            )
            axis.contour(
                lon_plot,
                lat_plot,
                field_plot,
                levels=levels,
                colors="#aa66f5",
                linewidths=0.45,
                alpha=0.6,
                transform=transform,
                zorder=21,
            )

        _land_mask_boundary(axis, zorder=29)
        _stamp_time(axis, panel_labels[index] if index < len(panel_labels) else f"Day {day}")

    for index in range(n, len(axes)):
        fig.delaxes(axes[index])
    fig.subplots_adjust(left=0.06, right=0.86, bottom=0.08, top=0.95, wspace=float(wspace), hspace=float(hspace))
    if contour is not None:
        cax = fig.add_axes([0.88, 0.18, 0.02, 0.62])
        colorbar = fig.colorbar(contour, cax=cax)
        colorbar.set_label("Relative concentration")
    panel_file = output_path / f"{prefix}_panel.png"
    fig.savefig(panel_file, dpi=260, bbox_inches="tight")
    panel_eps = panel_file.with_suffix(".eps")
    fig.savefig(panel_eps, format="eps", bbox_inches="tight")
    plt.close(fig)

    outputs: Dict[str, str] = {"panel_png": str(panel_file), "panel_eps": str(panel_eps)}

    if pollutant_data_all_days:
        if frame_labels is None:
            frame_labels = [f"Day {i + 1}" for i in range(len(pollutant_data_all_days))]
        if cartopy_ctx is None:
            fig_anim, ax_anim = plt.subplots(figsize=(8.2, 6.0))
            _draw_basemap(ax_anim)
        else:
            ccrs, _cfeature = cartopy_ctx
            fig_anim = plt.figure(figsize=(8.5, 6.0))
            ax_anim = fig_anim.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
            _draw_cartopy_basemap(ax_anim)
        _land_mask_fill(ax_anim)
        _land_mask_boundary(ax_anim, zorder=29)

        base = np.array(pollutant_data_all_days[0], dtype=float)
        base[base < concentration_floor] = np.nan
        if basemap_land_mask is not None:
            base[np.asarray(basemap_land_mask, dtype=float) > 0.5] = np.nan
        invalid = ~np.isfinite(base)
        base = _smooth_nan_field(base, invalid=invalid, passes=max(1, int(smooth_passes)))
        lon_plot, lat_plot, base_plot = _upsample_grid_and_field(base, lon_grid, lat_grid, int(upsample))
        extent = [float(np.min(lon_plot)), float(np.max(lon_plot)), float(np.min(lat_plot)), float(np.max(lat_plot))]
        img_transform = None
        zorder = None
        if cartopy_ctx is not None:
            ccrs, _cfeature = cartopy_ctx
            img_transform = ccrs.PlateCarree()
            zorder = 20

        img = ax_anim.imshow(
            np.ma.masked_invalid(base_plot),
            origin="lower",
            cmap=pollutant_cmap,
            vmin=concentration_floor,
            vmax=1.0,
            extent=extent,
            interpolation="bilinear",
            transform=img_transform,
            zorder=zorder,
        )
        stamp = ax_anim.text(
            0.02,
            0.98,
            frame_labels[0] if frame_labels else "Day 1",
            transform=ax_anim.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            color="white",
            zorder=30,
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
        )
        if cartopy_ctx is None:
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
            invalid = ~np.isfinite(field)
            field = _smooth_nan_field(field, invalid=invalid, passes=max(1, int(smooth_passes)))
            _lon_plot, _lat_plot, field_plot = _upsample_grid_and_field(field, lon_grid, lat_grid, int(upsample))
            img.set_data(np.ma.masked_invalid(field_plot))
            if frame_labels and frame < len(frame_labels):
                stamp.set_text(frame_labels[frame])
            return [img, stamp]

        ani = animation.FuncAnimation(fig_anim, update, frames=len(pollutant_data_all_days), interval=140, blit=False)
        gif_file = output_path / f"{prefix}.gif"
        ani.save(gif_file, writer=animation.PillowWriter(fps=8), dpi=220)
        plt.close(fig_anim)
        outputs["gif"] = str(gif_file)

    return outputs


def plot_multi_pollutant_overlay(
    lon_grid: np.ndarray,
    lat_grid: np.ndarray,
    pollutants: Dict[str, Sequence[np.ndarray]],
    reaction_frames: Optional[Union[Sequence[np.ndarray], Dict[str, Sequence[np.ndarray]]]],
    days: Sequence[int],
    panel_labels: Sequence[str],
    frame_labels: Optional[Sequence[str]],
    output_dir: Union[str, Path],
    prefix: str,
    basemap_elevation: Optional[np.ndarray],
    basemap_land_mask: Optional[np.ndarray],
    use_cartopy: bool = True,
    basemap_style: str = "stock",
    smooth_passes: int = 2,
    upsample: int = 3,
    pollutants_all_days: Optional[Dict[str, Sequence[np.ndarray]]] = None,
    reaction_frames_all_days: Optional[Union[Sequence[np.ndarray], Dict[str, Sequence[np.ndarray]]]] = None,
) -> Dict[str, str]:
    output_path = _as_path(output_dir)

    cartopy_ctx = None
    if use_cartopy:
        try:
            import cartopy.crs as ccrs  # type: ignore
            import cartopy.feature as cfeature  # type: ignore

            cartopy_ctx = (ccrs, cfeature)
        except Exception:
            cartopy_ctx = None

    def _stamp_time(axis, label: str):
        axis.text(
            0.02,
            0.98,
            label,
            transform=axis.transAxes,
            va="top",
            ha="left",
            fontsize=12,
            color="white",
            zorder=30,
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
        )

    def _smooth_nan_field(field: np.ndarray, invalid: np.ndarray, passes: int = 1) -> np.ndarray:
        if passes <= 0:
            return field
        kernel = np.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]], dtype=float)
        kernel = kernel / float(np.sum(kernel))
        out = np.array(field, dtype=float)
        valid = ~invalid
        out[invalid] = 0.0
        weight = valid.astype(float)

        def conv2(a: np.ndarray) -> np.ndarray:
            pad = np.pad(a, ((1, 1), (1, 1)), mode="edge")
            return (
                kernel[0, 0] * pad[:-2, :-2]
                + kernel[0, 1] * pad[:-2, 1:-1]
                + kernel[0, 2] * pad[:-2, 2:]
                + kernel[1, 0] * pad[1:-1, :-2]
                + kernel[1, 1] * pad[1:-1, 1:-1]
                + kernel[1, 2] * pad[1:-1, 2:]
                + kernel[2, 0] * pad[2:, :-2]
                + kernel[2, 1] * pad[2:, 1:-1]
                + kernel[2, 2] * pad[2:, 2:]
            )

        for _ in range(passes):
            num = conv2(out)
            den = conv2(weight)
            out = np.divide(num, np.clip(den, 1e-8, None))
        out[invalid] = np.nan
        return out

    def _upsample_grid_and_field(field: np.ndarray, factor: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if factor <= 1:
            return lon_grid, lat_grid, field
        lons_1d = lon_grid[0, :]
        lats_1d = lat_grid[:, 0]
        lon_hi = np.linspace(float(lons_1d.min()), float(lons_1d.max()), num=len(lons_1d) * factor)
        lat_hi = np.linspace(float(lats_1d.min()), float(lats_1d.max()), num=len(lats_1d) * factor)
        lon_hi_grid, lat_hi_grid = np.meshgrid(lon_hi, lat_hi)

        field_in = np.array(field, dtype=float)
        nan_mask = ~np.isfinite(field_in)
        if np.any(nan_mask):
            filled = field_in.copy()
            filled[nan_mask] = 0.0
            field_in = filled

        tmp = np.empty((field_in.shape[0], lon_hi.shape[0]), dtype=float)
        for i in range(field_in.shape[0]):
            tmp[i, :] = np.interp(lon_hi, lons_1d, field_in[i, :])
        out = np.empty((lat_hi.shape[0], lon_hi.shape[0]), dtype=float)
        for j in range(tmp.shape[1]):
            out[:, j] = np.interp(lat_hi, lats_1d, tmp[:, j])

        if np.any(nan_mask):
            out[out <= 0.0] = np.nan
        return lon_hi_grid, lat_hi_grid, out

    def _draw_cartopy_basemap(axis):
        if cartopy_ctx is None:
            return
        ccrs, cfeature = cartopy_ctx
        x_min, x_max = float(np.min(lon_grid)), float(np.max(lon_grid))
        y_min, y_max = float(np.min(lat_grid)), float(np.max(lat_grid))
        axis.set_extent([x_min, x_max, y_min, y_max], crs=ccrs.PlateCarree())
        style = str(basemap_style or "").strip().lower()
        if style in {"stock", "stock_img", "stockimg"}:
            try:
                axis.stock_img()
            except Exception:
                axis.set_facecolor("#000c3f")
                axis.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#000c3f", zorder=0)
                axis.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#4e5e3c", zorder=1)
        else:
            axis.set_facecolor("#000c3f")
            axis.add_feature(cfeature.OCEAN.with_scale("10m"), facecolor="#000c3f", zorder=0)
            axis.add_feature(cfeature.LAND.with_scale("10m"), facecolor="#4e5e3c", zorder=1)
        axis.add_feature(cfeature.COASTLINE.with_scale("10m"), linewidth=0.55, edgecolor="black", zorder=10)
        gl = axis.gridlines(draw_labels=True, linewidth=0.5, color="white", alpha=0.28, linestyle="--")
        gl.top_labels = False
        gl.right_labels = False

    def _land_mask_fill(axis):
        if basemap_land_mask is None:
            return
        land = (np.asarray(basemap_land_mask, dtype=float) > 0.5).astype(float)
        land = np.where(land > 0.5, 1.0, np.nan)
        fill_cmap = mcolors.ListedColormap(["#4e5e3c"])
        if cartopy_ctx is None:
            axis.pcolormesh(lon_grid, lat_grid, land, cmap=fill_cmap, shading="auto", alpha=0.35)
        else:
            ccrs, _cfeature = cartopy_ctx
            axis.pcolormesh(
                lon_grid,
                lat_grid,
                land,
                cmap=fill_cmap,
                shading="auto",
                alpha=0.35,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )

    def _land_mask_boundary(axis, zorder: int = 29):
        if basemap_land_mask is None:
            return
        land = (np.asarray(basemap_land_mask, dtype=float) > 0.5).astype(float)
        if cartopy_ctx is None:
            axis.contour(lon_grid, lat_grid, land, levels=[0.5], colors="black", linewidths=0.75, alpha=0.9)
        else:
            ccrs, _cfeature = cartopy_ctx
            axis.contour(
                lon_grid,
                lat_grid,
                land,
                levels=[0.5],
                colors="black",
                linewidths=0.75,
                alpha=0.9,
                transform=ccrs.PlateCarree(),
                zorder=zorder,
            )

    names = list(pollutants.keys())
    # user requirement: pollutant/reaction overlays must NOT use blue tones.
    safe_cmaps = [plt.cm.YlOrRd, plt.cm.Oranges, plt.cm.Greens, plt.cm.magma]
    cmap_map = {name: safe_cmaps[i % len(safe_cmaps)] for i, name in enumerate(names)}
    alpha_defaults = [0.58, 0.52, 0.46, 0.44]
    alpha_map = {name: alpha_defaults[i % len(alpha_defaults)] for i, name in enumerate(names)}

    reactions_panel: Dict[str, Sequence[np.ndarray]] = {}
    if reaction_frames is not None:
        if isinstance(reaction_frames, dict):
            reactions_panel = {str(k): list(v) for k, v in reaction_frames.items()}
        else:
            reactions_panel = {"Reaction rate": list(reaction_frames)}

    reactions_all: Dict[str, Sequence[np.ndarray]] = {}
    if reaction_frames_all_days is not None:
        if isinstance(reaction_frames_all_days, dict):
            reactions_all = {str(k): list(v) for k, v in reaction_frames_all_days.items()}
        else:
            reactions_all = {"Reaction rate": list(reaction_frames_all_days)}
    elif reactions_panel:
        reactions_all = reactions_panel

    n = len(days)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))
    if cartopy_ctx is None:
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5.0 * ncols, 4.4 * nrows))
        axes = np.array(axes).reshape(-1)
    else:
        ccrs, _cfeature = cartopy_ctx
        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(5.2 * ncols, 4.6 * nrows),
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
        axes = np.array(axes).reshape(-1)

    transform = None
    if cartopy_ctx is not None:
        ccrs, _cfeature = cartopy_ctx
        transform = ccrs.PlateCarree()

    overlay_levels = np.array([0.08, 0.12, 0.16, 0.20, 0.28, 0.36, 0.44, 0.52, 0.60, 0.70, 0.80, 0.90, 0.97, 1.00], dtype=float)
    reaction_levels = np.array([0.15, 0.25, 0.35, 0.50, 0.65, 0.80, 0.92, 1.00], dtype=float)
    reaction_cmaps = [plt.cm.autumn, plt.cm.spring, plt.cm.copper, plt.cm.RdPu]
    reaction_alphas = [0.22, 0.20, 0.18, 0.18]

    for idx, _day in enumerate(days):
        axis = axes[idx]
        if cartopy_ctx is None:
            axis.set_facecolor("#000c3f")
        else:
            _draw_cartopy_basemap(axis)
        _land_mask_fill(axis)

        for name in names:
            field = np.array(pollutants[name][idx], dtype=float)
            if basemap_land_mask is not None:
                field[np.asarray(basemap_land_mask, dtype=float) > 0.5] = np.nan
            invalid = ~np.isfinite(field)
            field = _smooth_nan_field(field, invalid=invalid, passes=max(0, int(smooth_passes)))
            lon_plot, lat_plot, field_plot = _upsample_grid_and_field(field, int(upsample))
            if cartopy_ctx is None:
                axis.contourf(lon_plot, lat_plot, field_plot, levels=overlay_levels, cmap=cmap_map[name], alpha=alpha_map[name], extend="max")
            else:
                axis.contourf(
                    lon_plot,
                    lat_plot,
                    field_plot,
                    levels=overlay_levels,
                    cmap=cmap_map[name],
                    alpha=alpha_map[name],
                    extend="max",
                    transform=transform,
                    zorder=20,
                )

        if reactions_panel:
            for ridx, (rname, rframes) in enumerate(reactions_panel.items()):
                rxn = np.array(rframes[idx], dtype=float)
                if basemap_land_mask is not None:
                    rxn[np.asarray(basemap_land_mask, dtype=float) > 0.5] = np.nan
                invalid = ~np.isfinite(rxn)
                rxn = _smooth_nan_field(rxn, invalid=invalid, passes=max(0, int(smooth_passes)))
                lon_plot, lat_plot, rxn_plot = _upsample_grid_and_field(rxn, int(upsample))
                finite = np.isfinite(rxn_plot)
                mx = float(np.nanmax(rxn_plot)) if np.any(finite) else 0.0
                if mx <= 0:
                    continue
                rxn_norm = rxn_plot / mx
                rxn_norm[~finite] = np.nan
                cmap = reaction_cmaps[ridx % len(reaction_cmaps)]
                alpha = reaction_alphas[ridx % len(reaction_alphas)]
                if cartopy_ctx is None:
                    axis.contourf(lon_plot, lat_plot, rxn_norm, levels=reaction_levels, cmap=cmap, alpha=alpha, extend="max")
                    axis.contour(lon_plot, lat_plot, rxn_norm, levels=[0.55], colors=[cmap(0.85)], linewidths=0.7, alpha=min(1.0, alpha + 0.25))
                else:
                    axis.contourf(
                        lon_plot,
                        lat_plot,
                        rxn_norm,
                        levels=reaction_levels,
                        cmap=cmap,
                        alpha=alpha,
                        extend="max",
                        transform=transform,
                        zorder=23,
                    )
                    axis.contour(
                        lon_plot,
                        lat_plot,
                        rxn_norm,
                        levels=[0.55],
                        colors=[cmap(0.85)],
                        linewidths=0.7,
                        alpha=min(1.0, alpha + 0.25),
                        transform=transform,
                        zorder=24,
                    )

        _land_mask_boundary(axis, zorder=29)
        _stamp_time(axis, panel_labels[idx])

    for idx in range(n, len(axes)):
        fig.delaxes(axes[idx])

    fig.subplots_adjust(left=0.06, right=0.985, bottom=0.08, top=0.90, wspace=0.18, hspace=0.12)

    legend_handles: List = []
    for name in names:
        legend_handles.append(Patch(facecolor=cmap_map[name](0.75), edgecolor="none", label=name, alpha=alpha_map[name]))
    if reactions_panel:
        for ridx, rname in enumerate(reactions_panel.keys()):
            cmap = reaction_cmaps[ridx % len(reaction_cmaps)]
            legend_handles.append(Patch(facecolor=cmap(0.80), edgecolor="none", label=rname, alpha=reaction_alphas[ridx % len(reaction_alphas)]))
    # Legend: top of figure, single row (user requirement).
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=max(1, len(legend_handles)),
        frameon=False,
        columnspacing=1.2,
        handlelength=1.6,
    )

    panel_file = output_path / f"{prefix}_panel.png"
    fig.savefig(panel_file, dpi=260, bbox_inches="tight")
    panel_eps = panel_file.with_suffix(".eps")
    fig.savefig(panel_eps, format="eps", bbox_inches="tight")
    plt.close(fig)

    outputs: Dict[str, str] = {"panel_png": str(panel_file), "panel_eps": str(panel_eps)}

    # Optional GIF (single-panel): higher DPI and same time stamp labels.
    if pollutants_all_days:
        if frame_labels is None:
            frame_labels = [f"Step {i + 1}" for i in range(len(next(iter(pollutants_all_days.values()))))]
        n_anim = min(len(frame_labels), *(len(seq) for seq in pollutants_all_days.values()))
        if reactions_all:
            n_anim = min(n_anim, *(len(seq) for seq in reactions_all.values()))
        if n_anim > 0:
            if cartopy_ctx is None:
                fig_anim, ax_anim = plt.subplots(figsize=(8.6, 6.2))
                ax_anim.set_facecolor("#000c3f")
            else:
                ccrs, _cfeature = cartopy_ctx
                fig_anim = plt.figure(figsize=(8.9, 6.4))
                ax_anim = fig_anim.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
                _draw_cartopy_basemap(ax_anim)

            _land_mask_fill(ax_anim)
            _land_mask_boundary(ax_anim, zorder=29)

            first_name = next(iter(pollutants_all_days.keys()))
            base0 = np.array(pollutants_all_days[first_name][0], dtype=float)
            if basemap_land_mask is not None:
                base0[np.asarray(basemap_land_mask, dtype=float) > 0.5] = np.nan
            invalid0 = ~np.isfinite(base0)
            base0 = _smooth_nan_field(base0, invalid=invalid0, passes=max(1, int(smooth_passes)))
            lon_hi, lat_hi, base_hi = _upsample_grid_and_field(base0, int(upsample))
            extent = [float(np.min(lon_hi)), float(np.max(lon_hi)), float(np.min(lat_hi)), float(np.max(lat_hi))]

            img_transform = None
            zorder = None
            if cartopy_ctx is not None:
                ccrs, _cfeature = cartopy_ctx
                img_transform = ccrs.PlateCarree()
                zorder = 20

            norm = mcolors.BoundaryNorm(overlay_levels, ncolors=256, clip=True)
            pollutant_layers = []
            for name in names:
                img = ax_anim.imshow(
                    np.ma.masked_invalid(base_hi) * 0.0,
                    origin="lower",
                    cmap=cmap_map[name],
                    norm=norm,
                    extent=extent,
                    interpolation="bilinear",
                    alpha=alpha_map[name],
                    transform=img_transform,
                    zorder=zorder,
                )
                pollutant_layers.append((name, img))

            reaction_layers = []
            if reactions_all:
                rnorm = mcolors.BoundaryNorm(reaction_levels, ncolors=256, clip=True)
                for ridx, (rname, _rframes) in enumerate(reactions_all.items()):
                    cmap = reaction_cmaps[ridx % len(reaction_cmaps)]
                    alpha = reaction_alphas[ridx % len(reaction_alphas)]
                    img = ax_anim.imshow(
                        np.ma.masked_invalid(base_hi) * 0.0,
                        origin="lower",
                        cmap=cmap,
                        norm=rnorm,
                        extent=extent,
                        interpolation="bilinear",
                        alpha=alpha,
                        transform=img_transform,
                        zorder=(zorder + 3) if zorder is not None else None,
                    )
                    reaction_layers.append((rname, img))

            stamp = ax_anim.text(
                0.02,
                0.98,
                frame_labels[0] if frame_labels else "Step 1",
                transform=ax_anim.transAxes,
                va="top",
                ha="left",
                fontsize=12,
                color="white",
                zorder=30,
                bbox={"facecolor": "black", "alpha": 0.35, "pad": 2, "edgecolor": "none"},
            )
            if cartopy_ctx is None:
                ax_anim.set_xlabel("Longitude")
                ax_anim.set_ylabel("Latitude")
                x_min, x_max = float(np.min(lon_grid)), float(np.max(lon_grid))
                y_min, y_max = float(np.min(lat_grid)), float(np.max(lat_grid))
                ax_anim.grid(alpha=0.35, linestyle="--", linewidth=0.6)
                ax_anim.set_xlim(x_min, x_max)
                ax_anim.set_ylim(y_min, y_max)

            def update(frame: int):
                artists = []
                for name, img in pollutant_layers:
                    field = np.array(pollutants_all_days[name][frame], dtype=float)
                    if basemap_land_mask is not None:
                        field[np.asarray(basemap_land_mask, dtype=float) > 0.5] = np.nan
                    invalid = ~np.isfinite(field)
                    field = _smooth_nan_field(field, invalid=invalid, passes=max(1, int(smooth_passes)))
                    _lon_plot, _lat_plot, field_plot = _upsample_grid_and_field(field, int(upsample))
                    field_plot = np.where(field_plot >= overlay_levels[0], field_plot, np.nan)
                    img.set_data(np.ma.masked_invalid(field_plot))
                    artists.append(img)

                for rname, img in reaction_layers:
                    rxn = np.array(reactions_all[rname][frame], dtype=float)
                    if basemap_land_mask is not None:
                        rxn[np.asarray(basemap_land_mask, dtype=float) > 0.5] = np.nan
                    invalid = ~np.isfinite(rxn)
                    rxn = _smooth_nan_field(rxn, invalid=invalid, passes=max(1, int(smooth_passes)))
                    _lon_plot, _lat_plot, rxn_plot = _upsample_grid_and_field(rxn, int(upsample))
                    finite = np.isfinite(rxn_plot)
                    mx = float(np.nanmax(rxn_plot)) if np.any(finite) else 0.0
                    if mx > 0:
                        rxn_norm = rxn_plot / mx
                    else:
                        rxn_norm = rxn_plot
                    rxn_norm = np.where(rxn_norm >= reaction_levels[0], rxn_norm, np.nan)
                    img.set_data(np.ma.masked_invalid(rxn_norm))
                    artists.append(img)

                if frame_labels and frame < len(frame_labels):
                    stamp.set_text(frame_labels[frame])
                artists.append(stamp)
                return artists

            ani = animation.FuncAnimation(fig_anim, update, frames=int(n_anim), interval=140, blit=False)
            gif_file = output_path / f"{prefix}.gif"
            ani.save(gif_file, writer=animation.PillowWriter(fps=8), dpi=220)
            plt.close(fig_anim)
            outputs["gif"] = str(gif_file)

    return outputs


def _land_fraction_window(land: np.ndarray, ci: int, cj: int, half_i: int, half_j: int) -> float:
    i0 = max(0, ci - half_i)
    i1 = min(land.shape[0], ci + half_i + 1)
    j0 = max(0, cj - half_j)
    j1 = min(land.shape[1], cj + half_j + 1)
    window = land[i0:i1, j0:j1]
    return float(np.mean(window)) if window.size else 0.0


def _coastal_ocean_candidates(land: np.ndarray) -> np.ndarray:
    land = np.asarray(land, dtype=bool)
    ocean = ~land
    neigh = np.zeros_like(land, dtype=int)
    neigh[1:, :] += land[:-1, :].astype(int)
    neigh[:-1, :] += land[1:, :].astype(int)
    neigh[:, 1:] += land[:, :-1].astype(int)
    neigh[:, :-1] += land[:, 1:].astype(int)
    return ocean & (neigh > 0)


def _select_coast_seeds(
    land_mask: np.ndarray,
    lon_vals: np.ndarray,
    lat_vals: np.ndarray,
    k: int,
    halfspan_deg: float,
) -> List[Tuple[float, float, float]]:
    land = np.asarray(land_mask, dtype=float) > 0.5
    coast_ocean = _coastal_ocean_candidates(land)
    candidates = np.argwhere(coast_ocean)
    if candidates.size == 0:
        return []

    deg_per_cell_lat = max(abs(float(lat_vals[1] - lat_vals[0])), 1e-6) if lat_vals.size > 1 else 1e-3
    deg_per_cell_lon = max(abs(float(lon_vals[1] - lon_vals[0])), 1e-6) if lon_vals.size > 1 else 1e-3
    half_i = max(1, int(round(halfspan_deg / deg_per_cell_lat)))
    half_j = max(1, int(round(halfspan_deg / deg_per_cell_lon)))

    scored: List[Tuple[float, int, int, float, float]] = []
    for ci, cj in candidates:
        neighbors = 0
        if ci > 0:
            neighbors += int(land[ci - 1, cj])
        if ci + 1 < land.shape[0]:
            neighbors += int(land[ci + 1, cj])
        if cj > 0:
            neighbors += int(land[ci, cj - 1])
        if cj + 1 < land.shape[1]:
            neighbors += int(land[ci, cj + 1])
        frac = _land_fraction_window(land, int(ci), int(cj), half_i, half_j)
        frac_penalty = 0.0
        # Prefer windows with visible coastline + some land (for coastal spill scenarios).
        if frac < 0.06:
            frac_penalty += (0.06 - frac) * 10.0
        if frac > 0.6:
            frac_penalty += (frac - 0.6) * 3.0
        lon_bias = float(lon_vals[int(cj)])
        score = frac_penalty - 0.6 * float(neighbors) + 0.002 * float(lon_bias)
        scored.append((score, int(ci), int(cj), float(lon_bias), float(lat_vals[int(ci)])))

    scored.sort(key=lambda x: x[0])
    picked: List[Tuple[float, float, float]] = []
    picked.append((scored[0][3], scored[0][4], _land_fraction_window(land, scored[0][1], scored[0][2], half_i, half_j)))

    if k <= 1:
        return picked

    remaining = scored[1:]
    while len(picked) < k and remaining:
        best_idx = None
        best_val = None
        for idx, (score, _ci, _cj, lon0, lat0) in enumerate(remaining):
            min_dist = min((lon0 - p[0]) ** 2 + (lat0 - p[1]) ** 2 for p in picked)
            val = min_dist - 0.04 * score
            if best_val is None or val > best_val:
                best_val = val
                best_idx = idx
        if best_idx is None:
            break
        _score, ci, cj, lon0, lat0 = remaining.pop(best_idx)
        picked.append((float(lon0), float(lat0), _land_fraction_window(land, ci, cj, half_i, half_j)))
    return picked


def simulate_diffusion_from_dataset(
    nc_path: Union[str, Path],
    output_dir: Union[str, Path],
    depth_index: int = 0,
    time_start: int = 0,
    time_count: int = 48,
    spatial_stride: int = 1,
    diffusion_coeff: float = 18.0,
    frame_seconds: float = 86400.0,
    substeps: int = 4,
    prefix: str = "dataset_diffusion",
    auto_coast: bool = True,
    coast_halfspan_deg: float = 3.0,
    seed_override: Optional[Tuple[float, float]] = None,
    force_crop: bool = True,
    basemap_style: str = "stock",
) -> Dict[str, Union[str, float, int, list]]:
    output_path = _as_path(output_dir)

    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        ds_use = ds
        u_name = "uo" if "uo" in ds.variables else "utotal"
        v_name = "vo" if "vo" in ds.variables else "vtotal"
        if u_name not in ds.variables or v_name not in ds.variables:
            raise ValueError("Dataset must contain `uo/vo` or `utotal/vtotal`.")

        coast_seed = None
        coast_window_land_frac = None
        if seed_override is not None:
            coast_seed = (float(seed_override[0]), float(seed_override[1]))
        elif auto_coast and "land_mask" in ds.variables:
            lm = ds["land_mask"]
            if "time" in lm.dims:
                lm = lm.isel(time=0)
            if "depth" in lm.dims:
                lm = lm.isel(depth=0)
            seeds = _select_coast_seeds(
                lm.values,
                ds["longitude"].values,
                ds["latitude"].values,
                k=1,
                halfspan_deg=float(coast_halfspan_deg),
            )
            if seeds:
                coast_seed = (seeds[0][0], seeds[0][1])
                coast_window_land_frac = float(seeds[0][2])

        if coast_seed is not None and "land_mask" in ds.variables:
            lons = ds["longitude"].values
            lats = ds["latitude"].values
            lat_min = coast_seed[1] - coast_halfspan_deg
            lat_max = coast_seed[1] + coast_halfspan_deg
            lon_min = coast_seed[0] - coast_halfspan_deg
            lon_max = coast_seed[0] + coast_halfspan_deg

            lat_inc = bool(lats[0] < lats[-1])
            lon_inc = bool(lons[0] < lons[-1])
            lat_slice = slice(lat_min, lat_max) if lat_inc else slice(lat_max, lat_min)
            lon_slice = slice(lon_min, lon_max) if lon_inc else slice(lon_max, lon_min)

            if force_crop or (coast_window_land_frac is not None and coast_window_land_frac >= 0.005):
                ds_use = ds.sel(latitude=lat_slice, longitude=lon_slice)

        u_da = ds_use[u_name]
        v_da = ds_use[v_name]

        if "depth" in u_da.dims:
            u_da = u_da.isel(depth=depth_index)
        if "depth" in v_da.dims:
            v_da = v_da.isel(depth=depth_index)

        u_da = u_da.isel(time=slice(time_start, time_start + time_count))
        v_da = v_da.isel(time=slice(time_start, time_start + time_count))

        if "latitude" not in u_da.dims or "longitude" not in u_da.dims:
            raise ValueError("Expected latitude/longitude dimensions in current fields.")

        u_da = u_da.isel(latitude=slice(None, None, spatial_stride), longitude=slice(None, None, spatial_stride))
        v_da = v_da.isel(latitude=slice(None, None, spatial_stride), longitude=slice(None, None, spatial_stride))

        lats = u_da["latitude"].values
        lons = u_da["longitude"].values
        times = u_da["time"].values
        u_series = np.asarray(u_da.values, dtype=float)
        v_series = np.asarray(v_da.values, dtype=float)

        basemap_elevation = None
        basemap_land_mask = None
        if "elevation" in ds_use.variables:
            elev = ds_use["elevation"].isel(latitude=slice(None, None, spatial_stride), longitude=slice(None, None, spatial_stride))
            basemap_elevation = np.asarray(elev.values, dtype=float)
        if "land_mask" in ds_use.variables:
            lm = ds_use["land_mask"]
            if "time" in lm.dims:
                lm = lm.isel(time=0)
            if "depth" in lm.dims:
                lm = lm.isel(depth=0)
            lm = lm.isel(latitude=slice(None, None, spatial_stride), longitude=slice(None, None, spatial_stride))
            basemap_land_mask = np.asarray(lm.values, dtype=float)

    n_frames = min(u_series.shape[0], v_series.shape[0])
    if n_frames < 2:
        raise ValueError("Not enough time frames for dataset-driven diffusion simulation.")

    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_span = max(float(np.max(lons) - np.min(lons)), 1e-6)
    lat_span = max(float(np.max(lats) - np.min(lats)), 1e-6)

    if coast_seed is not None:
        center_a = (float(coast_seed[0]), float(coast_seed[1]))
        center_b = (float(center_a[0] + 0.22 * lon_span), float(center_a[1] + 0.12 * lat_span))
    else:
        center_a = (float(np.min(lons) + 0.36 * lon_span), float(np.min(lats) + 0.45 * lat_span))
        center_b = (float(np.min(lons) + 0.68 * lon_span), float(np.min(lats) + 0.62 * lat_span))

    sigma_lon = max(lon_span * 0.12, 1e-6)
    sigma_lat = max(lat_span * 0.12, 1e-6)

    source_sigma_lon = max(sigma_lon * 0.28, 1e-6)
    source_sigma_lat = max(sigma_lat * 0.28, 1e-6)
    source_field = np.exp(
        -(((lon_grid - center_a[0]) ** 2) / (2.0 * source_sigma_lon**2) + ((lat_grid - center_a[1]) ** 2) / (2.0 * source_sigma_lat**2))
    )
    source_field = source_field / max(1e-8, float(np.max(source_field)))
    source_strength = 0.018

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
        u_frame = np.nan_to_num(u_series[frame_idx], nan=0.0)
        v_frame = np.nan_to_num(v_series[frame_idx], nan=0.0)

        for _ in range(max(1, substeps)):
            dc_dx = np.gradient(concentration, dx_m, axis=1)
            dc_dy = np.gradient(concentration, dy_m, axis=0)
            lap_x = np.gradient(np.gradient(concentration, dx_m, axis=1), dx_m, axis=1)
            lap_y = np.gradient(np.gradient(concentration, dy_m, axis=0), dy_m, axis=0)
            lap = lap_x + lap_y

            tendency = (-u_frame * dc_dx) + (-v_frame * dc_dy) + (diffusion_coeff * lap)
            concentration = concentration + dt * tendency
            concentration = concentration + source_strength * source_field
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

    delta_days = None
    try:
        if len(times) >= 2:
            delta = (np.datetime64(times[1]) - np.datetime64(times[0])).astype("timedelta64[s]").astype(int)
            delta_days = abs(delta) / 86400.0
    except Exception:
        delta_days = None
    label_prefix = "Day" if (delta_days is not None and delta_days <= 1.5) else "Step"
    try:
        all_labels = [f"{label_prefix} {i + 1} | {str(np.datetime_as_string(t, unit='D'))}" for i, t in enumerate(times[:n_frames])]
    except Exception:
        all_labels = [f"{label_prefix} {i + 1}" for i in range(n_frames)]
    sampled_labels = [all_labels[index] for index in sample_indices]

    media = plot_pollutant_diffusion(
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        pollutant_data=sampled_fields,
        days=sampled_days,
        pollutant_name="Dataset-driven Microplastic Diffusion Proxy",
        pollutant_data_all_days=frames,
        panel_labels=sampled_labels,
        frame_labels=all_labels,
        output_dir=output_path,
        prefix=prefix,
        basemap_elevation=basemap_elevation,
        basemap_land_mask=basemap_land_mask,
        use_cartopy=True,
        basemap_style=basemap_style,
        smooth_passes=2,
        upsample=4,
    )

    media["coast_seed"] = list(coast_seed) if coast_seed is not None else None
    media["coast_window_land_frac"] = float(coast_window_land_frac) if coast_window_land_frac is not None else None
    media["domain_lon_min"] = float(np.min(lons))
    media["domain_lon_max"] = float(np.max(lons))
    media["domain_lat_min"] = float(np.min(lats))
    media["domain_lat_max"] = float(np.max(lats))
    media["frame_count"] = int(n_frames)
    media["u_variable"] = u_name
    media["v_variable"] = v_name
    media["spatial_stride"] = int(spatial_stride)
    media["depth_index"] = int(depth_index)
    media["diffusion_coeff"] = float(diffusion_coeff)
    return media


def simulate_diffusion_suite_from_dataset(
    nc_path: Union[str, Path],
    output_dir: Union[str, Path],
    seed_count: int = 6,
    depth_index: int = 0,
    time_start: int = 0,
    time_count: int = 48,
    spatial_stride: int = 1,
    diffusion_coeff: float = 18.0,
    frame_seconds: float = 86400.0,
    substeps: int = 4,
    prefix: str = "dataset_suite",
    coast_halfspan_deg: float = 3.0,
    basemap_style: str = "stock",
) -> Dict[str, Union[str, int, list, dict]]:
    output_path = _as_path(output_dir)

    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        if "land_mask" not in ds.variables:
            raise ValueError("Suite mode requires `land_mask` to pick multiple coastal seeds.")
        lm = ds["land_mask"]
        if "time" in lm.dims:
            lm = lm.isel(time=0)
        if "depth" in lm.dims:
            lm = lm.isel(depth=0)
        seeds = _select_coast_seeds(
            lm.values,
            ds["longitude"].values,
            ds["latitude"].values,
            k=int(seed_count),
            halfspan_deg=float(coast_halfspan_deg),
        )

    # Keep only the latest artifacts in this folder: remove prior prefix-matching outputs.
    for path in output_path.glob(f"{prefix}*.*"):
        try:
            if path.is_file():
                path.unlink()
        except Exception:
            pass

    runs = []
    for idx, (lon0, lat0, frac) in enumerate(seeds):
        out = simulate_diffusion_from_dataset(
            nc_path=nc_path,
            output_dir=output_path,
            depth_index=depth_index,
            time_start=time_start,
            time_count=time_count,
            spatial_stride=spatial_stride,
            diffusion_coeff=diffusion_coeff,
            frame_seconds=frame_seconds,
            substeps=substeps,
            prefix=f"{prefix}_seed{idx:02d}",
            auto_coast=False,
            coast_halfspan_deg=coast_halfspan_deg,
            seed_override=(lon0, lat0),
            force_crop=True,
            basemap_style=basemap_style,
        )
        out["seed_index"] = idx
        out["seed_lonlat"] = [float(lon0), float(lat0)]
        out["seed_land_frac"] = float(frac)
        runs.append(out)

    manifest = {
        "output_dir": str(output_path),
        "seed_count": int(len(runs)),
        "prefix": prefix,
        "runs": runs,
    }
    (output_path / f"{prefix}_suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def simulate_multi_pollutant_from_dataset(
    nc_path: Union[str, Path],
    output_dir: Union[str, Path],
    depth_index: int = 0,
    time_start: int = 0,
    time_count: int = 48,
    spatial_stride: int = 1,
    diffusion_coeff: float = 18.0,
    frame_seconds: float = 86400.0,
    substeps: int = 4,
    prefix: str = "dataset_multispecies",
    auto_coast: bool = True,
    coast_halfspan_deg: float = 3.0,
    seed_override: Optional[Tuple[float, float]] = None,
    force_crop: bool = True,
    basemap_style: str = "stock",
    reaction_rate: float = 0.18,
    weathering_rate: float = 0.05,
) -> Dict[str, Union[str, float, int, list]]:
    output_path = _as_path(output_dir)

    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        ds_use = ds
        u_name = "uo" if "uo" in ds.variables else "utotal"
        v_name = "vo" if "vo" in ds.variables else "vtotal"
        if u_name not in ds.variables or v_name not in ds.variables:
            raise ValueError("Dataset must contain `uo/vo` or `utotal/vtotal`.")

        coast_seed = None
        coast_window_land_frac = None
        if seed_override is not None:
            coast_seed = (float(seed_override[0]), float(seed_override[1]))
        elif auto_coast and "land_mask" in ds.variables:
            lm = ds["land_mask"]
            if "time" in lm.dims:
                lm = lm.isel(time=0)
            if "depth" in lm.dims:
                lm = lm.isel(depth=0)
            seeds = _select_coast_seeds(
                lm.values,
                ds["longitude"].values,
                ds["latitude"].values,
                k=1,
                halfspan_deg=float(coast_halfspan_deg),
            )
            if seeds:
                coast_seed = (seeds[0][0], seeds[0][1])
                coast_window_land_frac = float(seeds[0][2])

        if coast_seed is not None and "land_mask" in ds.variables:
            lons = ds["longitude"].values
            lats = ds["latitude"].values
            lat_min = coast_seed[1] - coast_halfspan_deg
            lat_max = coast_seed[1] + coast_halfspan_deg
            lon_min = coast_seed[0] - coast_halfspan_deg
            lon_max = coast_seed[0] + coast_halfspan_deg
            lat_inc = bool(lats[0] < lats[-1])
            lon_inc = bool(lons[0] < lons[-1])
            lat_slice = slice(lat_min, lat_max) if lat_inc else slice(lat_max, lat_min)
            lon_slice = slice(lon_min, lon_max) if lon_inc else slice(lon_max, lon_min)
            if force_crop or (coast_window_land_frac is not None and coast_window_land_frac >= 0.005):
                ds_use = ds.sel(latitude=lat_slice, longitude=lon_slice)

        u_da = ds_use[u_name]
        v_da = ds_use[v_name]
        if "depth" in u_da.dims:
            u_da = u_da.isel(depth=depth_index)
        if "depth" in v_da.dims:
            v_da = v_da.isel(depth=depth_index)
        u_da = u_da.isel(time=slice(time_start, time_start + time_count))
        v_da = v_da.isel(time=slice(time_start, time_start + time_count))
        u_da = u_da.isel(latitude=slice(None, None, spatial_stride), longitude=slice(None, None, spatial_stride))
        v_da = v_da.isel(latitude=slice(None, None, spatial_stride), longitude=slice(None, None, spatial_stride))

        lats = u_da["latitude"].values
        lons = u_da["longitude"].values
        times = u_da["time"].values
        u_series = np.asarray(u_da.values, dtype=float)
        v_series = np.asarray(v_da.values, dtype=float)

        basemap_elevation = None
        basemap_land_mask = None
        if "elevation" in ds_use.variables:
            elev = ds_use["elevation"].isel(latitude=slice(None, None, spatial_stride), longitude=slice(None, None, spatial_stride))
            basemap_elevation = np.asarray(elev.values, dtype=float)
        if "land_mask" in ds_use.variables:
            lm = ds_use["land_mask"]
            if "time" in lm.dims:
                lm = lm.isel(time=0)
            if "depth" in lm.dims:
                lm = lm.isel(depth=0)
            lm = lm.isel(latitude=slice(None, None, spatial_stride), longitude=slice(None, None, spatial_stride))
            basemap_land_mask = np.asarray(lm.values, dtype=float)

    n_frames = min(u_series.shape[0], v_series.shape[0])
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    lon_span = max(float(np.max(lons) - np.min(lons)), 1e-6)
    lat_span = max(float(np.max(lats) - np.min(lats)), 1e-6)
    center = (float(coast_seed[0]) if coast_seed else float(np.mean(lons)), float(coast_seed[1]) if coast_seed else float(np.mean(lats)))
    center_b = (float(center[0] + 0.18 * lon_span), float(center[1] + 0.10 * lat_span))

    sigma_lon = max(lon_span * 0.10, 1e-6)
    sigma_lat = max(lat_span * 0.10, 1e-6)
    source_sigma_lon = max(sigma_lon * 0.25, 1e-6)
    source_sigma_lat = max(sigma_lat * 0.25, 1e-6)

    src_a = np.exp(-(((lon_grid - center[0]) ** 2) / (2.0 * source_sigma_lon**2) + ((lat_grid - center[1]) ** 2) / (2.0 * source_sigma_lat**2)))
    src_b = np.exp(-(((lon_grid - center_b[0]) ** 2) / (2.0 * source_sigma_lon**2) + ((lat_grid - center_b[1]) ** 2) / (2.0 * source_sigma_lat**2)))
    src_a = src_a / max(1e-8, float(np.max(src_a)))
    src_b = src_b / max(1e-8, float(np.max(src_b)))

    A = np.exp(-(((lon_grid - center[0]) ** 2) / (2.0 * sigma_lon**2) + ((lat_grid - center[1]) ** 2) / (2.0 * sigma_lat**2)))
    B = np.exp(-(((lon_grid - center_b[0]) ** 2) / (2.0 * sigma_lon**2) + ((lat_grid - center_b[1]) ** 2) / (2.0 * sigma_lat**2)))
    C = np.zeros_like(A)
    A = A / max(1e-8, float(np.max(A)))
    B = B / max(1e-8, float(np.max(B)))

    if basemap_land_mask is not None:
        land = np.asarray(basemap_land_mask, dtype=float) > 0.5
        A[land] = 0.0
        B[land] = 0.0
        C[land] = 0.0

    mean_lat = float(np.mean(lats))
    meters_per_deg_lat = 110_540.0
    meters_per_deg_lon = max(111_320.0 * np.cos(np.deg2rad(mean_lat)), 1e-6)
    dx_m = max(float(np.mean(np.diff(lons))) * meters_per_deg_lon, 1e-6)
    dy_m = max(float(np.mean(np.diff(lats))) * meters_per_deg_lat, 1e-6)
    dt = frame_seconds / max(1, substeps)

    frames_A: List[np.ndarray] = []
    frames_B: List[np.ndarray] = []
    frames_C: List[np.ndarray] = []
    frames_R_agg: List[np.ndarray] = []
    frames_R_weather: List[np.ndarray] = []
    for frame_idx in range(n_frames):
        u_frame = np.nan_to_num(u_series[frame_idx], nan=0.0)
        v_frame = np.nan_to_num(v_series[frame_idx], nan=0.0)
        for _ in range(max(1, substeps)):
            def step(field: np.ndarray) -> np.ndarray:
                dc_dx = np.gradient(field, dx_m, axis=1)
                dc_dy = np.gradient(field, dy_m, axis=0)
                lap_x = np.gradient(np.gradient(field, dx_m, axis=1), dx_m, axis=1)
                lap_y = np.gradient(np.gradient(field, dy_m, axis=0), dy_m, axis=0)
                lap = lap_x + lap_y
                return field + dt * ((-u_frame * dc_dx) + (-v_frame * dc_dy) + (diffusion_coeff * lap))

            A = step(A)
            B = step(B)
            C = step(C)

            R_agg = float(reaction_rate) * np.clip(A, 0.0, None) * np.clip(B, 0.0, None)
            R_weather = float(weathering_rate) * np.clip(B, 0.0, None)
            A = np.clip(A - dt * R_agg, 0.0, None)
            B = np.clip(B - dt * (R_agg + R_weather), 0.0, None)
            C = np.clip(C + dt * R_agg, 0.0, None)

            A = A + 0.010 * src_a
            B = B + 0.008 * src_b

            if basemap_land_mask is not None:
                land = np.asarray(basemap_land_mask, dtype=float) > 0.5
                A[land] = 0.0
                B[land] = 0.0
                C[land] = 0.0

            for field in (A, B, C):
                mx = float(np.max(field))
                if mx > 0:
                    field /= mx

        frames_A.append(A.copy())
        frames_B.append(B.copy())
        frames_C.append(C.copy())
        frames_R_agg.append(R_agg.copy())
        frames_R_weather.append(R_weather.copy())

    # Sample 6 panels.
    sample_indices = np.linspace(0, n_frames - 1, num=min(6, n_frames), dtype=int)
    sampled_days = [int(index + 1) for index in sample_indices]

    delta_days = None
    try:
        if len(times) >= 2:
            delta = (np.datetime64(times[1]) - np.datetime64(times[0])).astype("timedelta64[s]").astype(int)
            delta_days = abs(delta) / 86400.0
    except Exception:
        delta_days = None
    label_prefix = "Day" if (delta_days is not None and delta_days <= 1.5) else "Step"
    try:
        all_labels = [f"{label_prefix} {i + 1} | {str(np.datetime_as_string(t, unit='D'))}" for i, t in enumerate(times[:n_frames])]
    except Exception:
        all_labels = [f"{label_prefix} {i + 1}" for i in range(n_frames)]
    panel_labels = [all_labels[index] for index in sample_indices]

    pollutants_panel = {
        "Microplastics": [frames_A[index] for index in sample_indices],
        "Crude oil": [frames_B[index] for index in sample_indices],
        "Aggregates (MP×Oil)": [frames_C[index] for index in sample_indices],
    }
    pollutants_all = {
        "Microplastics": frames_A,
        "Crude oil": frames_B,
        "Aggregates (MP×Oil)": frames_C,
    }
    reactions_panel = {
        "Aggregation rate (MP×Oil)": [frames_R_agg[index] for index in sample_indices],
        "Oil weathering rate": [frames_R_weather[index] for index in sample_indices],
    }
    reactions_all = {
        "Aggregation rate (MP×Oil)": frames_R_agg,
        "Oil weathering rate": frames_R_weather,
    }

    media = plot_multi_pollutant_overlay(
        lon_grid=lon_grid,
        lat_grid=lat_grid,
        pollutants=pollutants_panel,
        reaction_frames=reactions_panel,
        days=sampled_days,
        panel_labels=panel_labels,
        frame_labels=all_labels,
        output_dir=output_path,
        prefix=prefix,
        basemap_elevation=basemap_elevation,
        basemap_land_mask=basemap_land_mask,
        use_cartopy=True,
        basemap_style=basemap_style,
        smooth_passes=2,
        upsample=4,
        pollutants_all_days=pollutants_all,
        reaction_frames_all_days=reactions_all,
    )

    media["coast_seed"] = list(coast_seed) if coast_seed is not None else None
    media["coast_window_land_frac"] = float(coast_window_land_frac) if coast_window_land_frac is not None else None
    media["frame_count"] = int(n_frames)
    media["reaction_rate"] = float(reaction_rate)
    media["weathering_rate"] = float(weathering_rate)
    (output_path / f"{prefix}_manifest.json").write_text(json.dumps(media, indent=2), encoding="utf-8")
    return media


def simulate_multi_pollutant_suite_from_dataset(
    nc_path: Union[str, Path],
    output_dir: Union[str, Path],
    seed_count: int = 6,
    depth_index: int = 0,
    time_start: int = 0,
    time_count: int = 48,
    spatial_stride: int = 1,
    diffusion_coeff: float = 18.0,
    frame_seconds: float = 86400.0,
    substeps: int = 4,
    prefix: str = "dataset_multispecies_suite",
    coast_halfspan_deg: float = 3.0,
    basemap_style: str = "stock",
    reaction_rate: float = 0.18,
    weathering_rate: float = 0.05,
) -> Dict[str, Union[str, int, list, dict]]:
    output_path = _as_path(output_dir)

    import xarray as xr

    with xr.open_dataset(nc_path) as ds:
        if "land_mask" not in ds.variables:
            raise ValueError("Suite mode requires `land_mask` to pick multiple coastal seeds.")
        lm = ds["land_mask"]
        if "time" in lm.dims:
            lm = lm.isel(time=0)
        if "depth" in lm.dims:
            lm = lm.isel(depth=0)
        seeds = _select_coast_seeds(
            lm.values,
            ds["longitude"].values,
            ds["latitude"].values,
            k=int(seed_count),
            halfspan_deg=float(coast_halfspan_deg),
        )

    # Keep only the latest artifacts in this folder: remove prior prefix-matching outputs.
    for path in output_path.glob(f"{prefix}*.*"):
        try:
            if path.is_file():
                path.unlink()
        except Exception:
            pass

    runs = []
    for idx, (lon0, lat0, frac) in enumerate(seeds):
        out = simulate_multi_pollutant_from_dataset(
            nc_path=nc_path,
            output_dir=output_path,
            depth_index=depth_index,
            time_start=time_start,
            time_count=time_count,
            spatial_stride=spatial_stride,
            diffusion_coeff=diffusion_coeff,
            frame_seconds=frame_seconds,
            substeps=substeps,
            prefix=f"{prefix}_seed{idx:02d}",
            auto_coast=False,
            coast_halfspan_deg=coast_halfspan_deg,
            seed_override=(lon0, lat0),
            force_crop=True,
            basemap_style=basemap_style,
            reaction_rate=reaction_rate,
            weathering_rate=weathering_rate,
        )
        out["seed_index"] = idx
        out["seed_lonlat"] = [float(lon0), float(lat0)]
        out["seed_land_frac"] = float(frac)
        runs.append(out)

    manifest = {
        "output_dir": str(output_path),
        "seed_count": int(len(runs)),
        "prefix": prefix,
        "runs": runs,
    }
    (output_path / f"{prefix}_suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
