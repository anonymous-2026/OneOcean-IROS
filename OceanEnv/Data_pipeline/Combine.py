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
    ds = xr.open_dataset(water_file)
    original_lats = ds['latitude'].values
    original_lons = ds['longitude'].values
    original_depths = ds['depth'].values
    original_times = ds['time'].values

    print("Original data dimensions:")
    print(f"Times: {len(original_times)}")
    print(f"Depths: {len(original_depths)}")
    print(f"Latitudes: {len(original_lats)}")
    print(f"Longitudes: {len(original_lons)}")

    lon_grid, lat_grid = np.meshgrid(original_lons, original_lats)
    target_lon_grid, target_lat_grid = np.meshgrid(lons, lats)

    variables_to_interpolate = ['so', 'thetao', 'uo', 'vo', 'zos', 'utide', 'utotal', 'vtide', 'vtotal']
    interpolated_variables = {var: [] for var in variables_to_interpolate}

    for var in variables_to_interpolate:
        print(f"Interpolating variable: {var}")
        time_depth_data = []

        for t_idx, time in enumerate(original_times):
            depth_data = []
            for d_idx, depth in enumerate(original_depths):
                print(f"  Time: {time}, Depth: {depth}")
                data = ds[var].isel(time=t_idx, depth=d_idx).values

                valid_mask = np.isfinite(data)
                if not valid_mask.any():
                    print(f"  Warning: All values are NaN or Inf for variable {var} at time {time}, depth {depth}. Using zeros.")
                    interpolated_data = np.zeros((len(lats), len(lons)))
                else:
                    valid_lon = lon_grid[valid_mask]
                    valid_lat = lat_grid[valid_mask]
                    valid_data = data[valid_mask]

                    rbf_interpolator = Rbf(valid_lon, valid_lat, valid_data, function='linear')
                    interpolated_data = rbf_interpolator(target_lon_grid, target_lat_grid)
                
                depth_data.append(interpolated_data)
            time_depth_data.append(depth_data)
        
        # Reshape the data to (time, depth, lat, lon)
        stacked_data = np.array(time_depth_data)
        interpolated_variables[var] = (['time', 'depth', 'latitude', 'longitude'], stacked_data)

    # Add elevation data (no time/depth dimensions)
    combined_variables = interpolated_variables.copy()
    combined_variables["elevation"] = (["latitude", "longitude"], elevation)

    # Create the dataset with all coordinates
    combined_ds = xr.Dataset(
        combined_variables,
        coords={
            'time': original_times,
            'depth': original_depths,
            'latitude': lats,
            'longitude': lons
        }
    )

    # Save the combined dataset
    combined_output_path = os.path.join(output_dir, "combined_environment.nc")
    combined_ds.to_netcdf(combined_output_path)

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

    ds = xr.open_dataset(water_file, chunks=chunks)

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
                "Check that the water file bbox matches the terrain crop bbox."
            )
        return ds_in

    def _normalize_lon(ds_in: xr.Dataset) -> xr.Dataset:
        lon = ds_in["longitude"].values
        lon_min, lon_max = float(np.min(lon)), float(np.max(lon))
        target_min, target_max = float(np.min(lons)), float(np.max(lons))

        # Detect 0..360 vs -180..180 convention mismatch.
        if lon_min >= 0.0 and target_min < 0.0:
            ds_in = ds_in.assign_coords(longitude=((ds_in["longitude"] + 180.0) % 360.0) - 180.0)
        elif lon_max <= 180.0 and target_max > 180.0:
            ds_in = ds_in.assign_coords(longitude=(ds_in["longitude"] % 360.0))
        return ds_in

    ds = _fix_swapped_latlon(ds)
    ds = _normalize_lon(ds)

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
    missing = sorted(set(["so", "thetao", "uo", "vo", "zos", "utide", "utotal", "vtide", "vtotal"]) - set(variables_to_interpolate))
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

    encoding = {}
    for v in combined_ds.data_vars:
        if v == "elevation":
            encoding[v] = {"zlib": True, "complevel": compression_level, "dtype": "float32"}
            continue
        if v == "land_mask":
            encoding[v] = {"zlib": True, "complevel": compression_level, "dtype": "uint8"}
            continue

        dims = combined_ds[v].dims
        if dims == ("time", "latitude", "longitude"):
            chunksizes = (1, min(240, len(lats)), min(240, len(lons)))
        elif dims == ("time", "depth", "latitude", "longitude"):
            chunksizes = (1, 1, min(240, len(lats)), min(240, len(lons)))
        else:
            chunksizes = None

        enc = {"zlib": True, "complevel": compression_level, "dtype": dtype}
        if chunksizes:
            enc["chunksizes"] = chunksizes
        encoding[v] = enc

    combined_ds.to_netcdf(out_path, engine="netcdf4", encoding=encoding)
    print('===================================')
    print(f"Fast combined dataset written to {out_path}")
    print(combined_ds)
    print('===================================')


def visualize_combined_data(combined_file, selected_time=None, selected_depth=None, time_index=None, depth_index=None):
    combined_ds = xr.open_dataset(combined_file)

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
                plt.title(f"{var} (Time: {selected_time}, Depth: {selected_depth})")
        elif 'time' in data.dims:
            if selected_time is not None:
                data.sel(time=selected_time).plot(cmap=colormaps[i % len(colormaps)])
                plt.title(f"{var} (Time: {selected_time})")
        elif 'depth' in data.dims:
            if selected_depth is not None:
                data.sel(depth=selected_depth).plot(cmap=colormaps[i % len(colormaps)])
                plt.title(f"{var} (Depth: {selected_depth})")
        else:
            data.plot(cmap=colormaps[i % len(colormaps)])
            plt.title(f"{var}")
            
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()


def print_structure(combined_file):
    combined_ds = xr.open_dataset(combined_file)
    print("Combined dataset structure:")
    print(combined_ds)


def interpolate_geotiff(file_path, new_resolution, save_path):
    try:
        with rasterio.open(file_path) as dataset:

            data = dataset.read(1)
            transform = dataset.transform
            crs = dataset.crs
            profile = dataset.profile

            rows, cols = data.shape
            x = np.linspace(transform[2], transform[2] + transform[0] * cols, cols)
            y = np.linspace(transform[5], transform[5] + transform[4] * rows, rows)
            xv, yv = np.meshgrid(x, y)

            valid_mask = ~np.isnan(data)
            valid_points = np.array([xv[valid_mask], yv[valid_mask]]).T
            valid_values = data[valid_mask]

            new_x = np.linspace(x.min(), x.max(), new_resolution[1])
            new_y = np.linspace(y.min(), y.max(), new_resolution[0])
            new_xv, new_yv = np.meshgrid(new_x, new_y)

            interpolated_data = griddata(valid_points, valid_values, (new_xv, new_yv), method='linear')

            new_transform = rasterio.transform.from_bounds(new_x.min(), new_y.min(), new_x.max(), new_y.max(),
                                                           new_resolution[1], new_resolution[0])
            profile.update({
                'height': new_resolution[0],
                'width': new_resolution[1],
                'transform': new_transform,
                'dtype': 'float32'
            })

            with rasterio.open(save_path, 'w', **profile) as dst:
                dst.write(interpolated_data.astype(rasterio.float32), 1)

            print(f"* tif saved as: {save_path}")
            return interpolated_data

    except Exception as e:
        print(f"!!! error: {e}")
        return None
