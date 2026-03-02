import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import rasterio
import xarray as xr
from scipy.interpolate import Rbf, griddata


def _read_terrain_grid(terrain_file: str):
    with rasterio.open(terrain_file) as src:
        elevation = src.read(1)
        bounds = src.bounds
        lons = np.linspace(bounds.left, bounds.right, src.width, dtype=np.float64)
        lats_desc = np.linspace(bounds.top, bounds.bottom, src.height, dtype=np.float64)

    # Use ascending latitude for xarray/scipy interpolation; flip elevation accordingly.
    if lats_desc[0] > lats_desc[-1]:
        lats = lats_desc[::-1]
        elevation = elevation[::-1, :]
    else:
        lats = lats_desc

    return elevation, lats, lons, bounds


def _read_terrain_grid_resampled(terrain_file: str, target_res_deg: float):
    if target_res_deg <= 0:
        raise ValueError(f"target_res_deg must be > 0, got {target_res_deg}")

    from rasterio.enums import Resampling

    with rasterio.open(terrain_file) as src:
        bounds = src.bounds
        width = max(2, int(np.ceil((bounds.right - bounds.left) / target_res_deg)) + 1)
        height = max(2, int(np.ceil((bounds.top - bounds.bottom) / target_res_deg)) + 1)
        elevation = src.read(1, out_shape=(height, width), resampling=Resampling.bilinear)
        lons = np.linspace(bounds.left, bounds.right, width, dtype=np.float64)
        lats_desc = np.linspace(bounds.top, bounds.bottom, height, dtype=np.float64)

    if lats_desc[0] > lats_desc[-1]:
        lats = lats_desc[::-1]
        elevation = elevation[::-1, :]
    else:
        lats = lats_desc

    return elevation, lats, lons, bounds


def interpolate_and_merge_rbf(terrain_file, water_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # Read terrain file
    with rasterio.open(terrain_file) as src:
        elevation = src.read(1)
        bounds = src.bounds
        lons = np.linspace(bounds.left, bounds.right, src.width)
        lats = np.linspace(bounds.top, bounds.bottom, src.height)

    # Read water data
    water_data = xr.open_dataset(water_file, engine="h5netcdf")

    # Create meshgrid for terrain
    terrain_lon_grid, terrain_lat_grid = np.meshgrid(lons, lats)

    combined_ds = xr.Dataset()

    for var in water_data.data_vars:
        print(f"Interpolating variable: {var}")

        # Extract water variable data
        data_var = water_data[var]

        if "time" in data_var.dims and "depth" in data_var.dims:
            # Loop over time and depth
            interpolated_data = []
            for time in data_var.time.values:
                for depth in data_var.depth.values:
                    water_slice = data_var.sel(time=time, depth=depth).values

                    # Flatten coordinates and data for interpolation
                    water_lon_grid, water_lat_grid = np.meshgrid(water_data.longitude.values, water_data.latitude.values)
                    points = np.array([water_lon_grid.flatten(), water_lat_grid.flatten()]).T
                    values = water_slice.flatten()

                    # Perform RBF interpolation
                    rbf = Rbf(points[:, 0], points[:, 1], values, function="linear")
                    interpolated_slice = rbf(terrain_lon_grid, terrain_lat_grid)

                    interpolated_data.append(interpolated_slice)

            interpolated_data = np.array(interpolated_data).reshape(
                len(data_var.time), len(data_var.depth), terrain_lon_grid.shape[0], terrain_lon_grid.shape[1]
            )

            combined_ds[var] = (("time", "depth", "latitude", "longitude"), interpolated_data)
        else:
            # Interpolate 2D data (no time/depth)
            water_slice = data_var.values

            water_lon_grid, water_lat_grid = np.meshgrid(water_data.longitude.values, water_data.latitude.values)
            points = np.array([water_lon_grid.flatten(), water_lat_grid.flatten()]).T
            values = water_slice.flatten()

            rbf = Rbf(points[:, 0], points[:, 1], values, function="linear")
            interpolated_data = rbf(terrain_lon_grid, terrain_lat_grid)

            combined_ds[var] = (("latitude", "longitude"), interpolated_data)

    # Add terrain data as elevation variable
    combined_ds["elevation"] = (("latitude", "longitude"), elevation)

    combined_ds = combined_ds.assign_coords(
        latitude=("latitude", lats),
        longitude=("longitude", lons),
    )

    # Save the combined dataset
    combined_output_path = os.path.join(output_dir, "combined_environment.nc")
    combined_ds.to_netcdf(combined_output_path, engine="h5netcdf")

    print('===================================')
    print(f"All interpolated and combined data have been saved to {output_dir}")
    print("Combined dataset structure:")
    print(combined_ds)
    print('===================================')


# Backward-compatible alias for older scripts/docs
interpolate_and_merge = interpolate_and_merge_rbf


def interpolate_and_merge_fast(
    terrain_file: str,
    water_file: str,
    output_dir: str,
    *,
    method: str = "linear",
    chunks: dict | None = None,
    dtype: str = "float32",
    compression_level: int = 4,
    strict_coordinate_check: bool = True,
    target_res_deg: float | None = None,
    allow_extrapolation: bool = False,
):
    """
    Faster/safer alternative to the RBF-based merge:
    - Uses xarray's vectorized lat/lon interpolation (optionally with dask chunks).
    - Writes float32 + compression to keep file size manageable.
    """
    os.makedirs(output_dir, exist_ok=True)

    if target_res_deg is None:
        elevation, lats, lons, bounds = _read_terrain_grid(terrain_file)
    else:
        elevation, lats, lons, bounds = _read_terrain_grid_resampled(terrain_file, float(target_res_deg))

    # GeoTIFF preprocessing may set land/out-of-range pixels to NaN. Keep a mask and fill NaNs in elevation
    # so downstream consumers always get a dense grid.
    land_mask = ~np.isfinite(elevation)
    elevation = np.where(np.isfinite(elevation), elevation, 0.0).astype(np.float32, copy=False)

    if chunks is None:
        # Small input water file, but large output grid. Chunk by time to limit peak memory.
        chunks = {"time": 24}

    ds = xr.open_dataset(water_file, chunks=chunks, engine="h5netcdf")

    if "latitude" not in ds.coords or "longitude" not in ds.coords:
        raise ValueError("Expected water dataset coords: latitude, longitude")

    did_swap_latlon = False

    def _ranges_overlap(a_min: float, a_max: float, b_min: float, b_max: float, tol: float = 1e-6) -> bool:
        return (a_min <= b_max + tol) and (b_min <= a_max + tol)

    def _fix_swapped_latlon(ds_in: xr.Dataset) -> xr.Dataset:
        nonlocal did_swap_latlon
        lat0 = ds_in["latitude"].values
        lon0 = ds_in["longitude"].values
        t_lat_min, t_lat_max = float(np.min(lats)), float(np.max(lats))
        t_lon_min, t_lon_max = float(np.min(lons)), float(np.max(lons))
        d_lat_min, d_lat_max = float(np.min(lat0)), float(np.max(lat0))
        d_lon_min, d_lon_max = float(np.min(lon0)), float(np.max(lon0))

        ok_direct = _ranges_overlap(d_lat_min, d_lat_max, t_lat_min, t_lat_max) and _ranges_overlap(
            d_lon_min, d_lon_max, t_lon_min, t_lon_max
        )
        ok_swapped = _ranges_overlap(d_lon_min, d_lon_max, t_lat_min, t_lat_max) and _ranges_overlap(
            d_lat_min, d_lat_max, t_lon_min, t_lon_max
        )

        if ok_direct:
            return ds_in
        if ok_swapped:
            did_swap_latlon = True
            if strict_coordinate_check:
                raise ValueError(
                    "Water dataset latitude/longitude look swapped relative to terrain bounds. "
                    "This usually means the CMEMS subset bbox was provided with lon/lat reversed. "
                    "Regenerate the water file with correct bbox (minimum_longitude/maximum_longitude and "
                    "minimum_latitude/maximum_latitude)."
                )
            print(
                "Warning: water dataset latitude/longitude look swapped; proceeding by swapping dims/coords "
                "(NOT recommended for scientific correctness)."
            )
            return ds_in.rename({"latitude": "longitude", "longitude": "latitude"})

        print(
            "Warning: water dataset lat/lon ranges do not overlap terrain bounds (direct or swapped). "
            "Interpolation may yield NaNs."
        )
        if strict_coordinate_check:
            raise ValueError(
                "Water dataset lat/lon ranges do not overlap terrain bounds. "
                "Fix bbox or regenerate terrain/water with consistent region."
            )
        return ds_in

    ds = _fix_swapped_latlon(ds)

    # Ensure coords are monotonic increasing for interpolation.
    if ds["latitude"].values[0] > ds["latitude"].values[-1]:
        ds = ds.sortby("latitude")
    if ds["longitude"].values[0] > ds["longitude"].values[-1]:
        ds = ds.sortby("longitude")

    variables_to_interpolate = [
        v
        for v in ["so", "thetao", "uo", "vo", "zos", "utide", "utotal", "vtide", "vtotal"]
        if v in ds.data_vars
    ]
    missing = sorted(
        set(["so", "thetao", "uo", "vo", "zos", "utide", "utotal", "vtide", "vtotal"]) - set(variables_to_interpolate)
    )
    if missing:
        print(f"Warning: missing variables in water dataset: {missing}")

    interp_kwargs = {"fill_value": "extrapolate"} if allow_extrapolation else None
    interp_ds = ds[variables_to_interpolate].interp(
        latitude=lats,
        longitude=lons,
        method=method,
        kwargs=interp_kwargs,
    )
    interp_ds = interp_ds.astype(dtype)

    # Standardize dimension order for downstream consumers.
    desired = [d for d in ["time", "depth", "latitude", "longitude"] if d in interp_ds.dims]
    remaining = [d for d in interp_ds.dims if d not in desired]
    interp_ds = interp_ds.transpose(*(desired + remaining))

    # Quick NaN check (avoid full reduction on dask arrays).
    sample = interp_ds[variables_to_interpolate[0]].isel(time=0)
    if bool(sample.isnull().any()):
        print(
            "Warning: NaNs detected after interpolation. "
            "This usually indicates bbox/coordinate mismatch between terrain and water datasets."
        )

    combined_ds = interp_ds.assign(
        elevation=(("latitude", "longitude"), elevation),
        land_mask=(("latitude", "longitude"), land_mask.astype(np.uint8)),
    )

    combined_ds.attrs.update(
        {
            "generated_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ"),
            "terrain_file": terrain_file,
            "water_file": water_file,
            "water_latlon_swapped_autocorrected": str(bool(did_swap_latlon)),
            "terrain_bounds_left": float(bounds.left),
            "terrain_bounds_right": float(bounds.right),
            "terrain_bounds_bottom": float(bounds.bottom),
            "terrain_bounds_top": float(bounds.top),
            "interpolation_method": method,
        }
    )

    out_path = os.path.join(output_dir, "combined_environment.nc")

    encoding: dict[str, dict] = {}
    for v in combined_ds.data_vars:
        if v == "elevation":
            encoding[v] = {
                "compression": "gzip",
                "compression_opts": compression_level,
                "dtype": "float32",
            }
            continue
        if v == "land_mask":
            encoding[v] = {
                "compression": "gzip",
                "compression_opts": compression_level,
                "dtype": "uint8",
            }
            continue

        dims = combined_ds[v].dims
        if dims == ("time", "latitude", "longitude"):
            chunksizes = (1, min(240, len(lats)), min(240, len(lons)))
        elif dims == ("time", "depth", "latitude", "longitude"):
            chunksizes = (1, 1, min(240, len(lats)), min(240, len(lons)))
        else:
            chunksizes = None

        enc: dict = {"compression": "gzip", "compression_opts": compression_level, "dtype": dtype}
        if chunksizes:
            enc["chunksizes"] = chunksizes
        encoding[v] = enc

    combined_ds.to_netcdf(out_path, engine="h5netcdf", encoding=encoding)
    print('===================================')
    print(f"Fast combined dataset written to {out_path}")
    print(combined_ds)
    print('===================================')


def visualize_combined_data(combined_file, selected_time=None, selected_depth=None, time_index=None, depth_index=None):
    combined_ds = xr.open_dataset(combined_file, engine="h5netcdf")

    if time_index is not None:
        selected_time = combined_ds['time'].values[time_index] if 'time' in combined_ds.dims else None
    if depth_index is not None:
        selected_depth = combined_ds['depth'].values[depth_index] if 'depth' in combined_ds.dims else None

    for var in combined_ds.data_vars:
        data = combined_ds[var]
        print(f"Data characteristics for variable {var}:")
        print(f"  Dimensions: {data.dims}")
        print(f"  Mean: {data.mean().item():.2f}")
        print(f"  Std Dev: {data.std().item():.2f}")
        print(f"  Min: {data.min().item():.2f}")
        print(f"  Max: {data.max().item():.2f}")
        print("---")

    colormaps = ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'cool', 'hot', 'spring', 'summer']
    for i, var in enumerate(combined_ds.data_vars):
        plt.figure(figsize=(10, 6))
        data = combined_ds[var]
        
        if 'time' in data.dims and 'depth' in data.dims:
            if selected_time is not None and selected_depth is not None:
                data.sel(time=selected_time, depth=selected_depth).plot(cmap=colormaps[i % len(colormaps)])
            else:
                data.isel(time=0, depth=0).plot(cmap=colormaps[i % len(colormaps)])
        elif 'time' in data.dims:
            if selected_time is not None:
                data.sel(time=selected_time).plot(cmap=colormaps[i % len(colormaps)])
            else:
                data.isel(time=0).plot(cmap=colormaps[i % len(colormaps)])
        else:
            data.plot(cmap=colormaps[i % len(colormaps)])

        plt.title(f"{var} at time {selected_time} depth {selected_depth}")
        plt.savefig(os.path.join(os.path.dirname(combined_file), f"{var}_visualization.png"))
        plt.close()


def interpolate_geotiff(terrain_data, terrain_lons, terrain_lats, target_lons, target_lats, method='linear'):
    grid_lon, grid_lat = np.meshgrid(terrain_lons, terrain_lats)
    points = np.array([grid_lon.flatten(), grid_lat.flatten()]).T
    values = terrain_data.flatten()
    target_grid_lon, target_grid_lat = np.meshgrid(target_lons, target_lats)
    interpolated = griddata(points, values, (target_grid_lon, target_grid_lat), method=method)
    return interpolated

