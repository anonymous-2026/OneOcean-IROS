import copernicusmarine
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import os
from PIL import Image
from datetime import datetime, timezone
from pathlib import Path

'''
Parameters:
    - so: 海水盐度 (Sea water salinity)
    - thetao: 海水潜在温度 (Sea water potential temperature)
    - uo: 东向海水流速 (Eastward sea water velocity)
    - vo: 北向海水流速 (Northward sea water velocity)
    - zos: 海面高度 (Sea surface height above geoid)
    - utide: 东向潮流速度 (Eastward tidal velocity)
    - utotal: 总东向海水流速 (Total eastward sea water velocity)
    #- vsdx: 海浪斯托克斯漂移速度 - x 方向 (Sea surface wave stokes drift x velocity)
    #- vsdy: 海浪斯托克斯漂移速度 - y 方向 (Sea surface wave stokes drift y velocity)
    - vtide: 北向潮流速度 (Northward tidal velocity)
    - vtotal: 总北向海水流速 (Total northward sea water velocity)
'''

def _env_or_value(value: str | None, env_key: str) -> str | None:
    if value:
        return value
    v = os.environ.get(env_key)
    return v if v else None


def _open_depth_count(path: str) -> int:
    ds = xr.open_dataset(path)
    if "depth" not in ds.dims:
        return 0
    return int(ds.sizes.get("depth", 0))

def _normalize_lon_values(lon: np.ndarray) -> np.ndarray:
    lon = np.asarray(lon, dtype=np.float64)
    # Normalize to [-180, 180) if values look like 0..360.
    if np.nanmin(lon) >= 0.0 and np.nanmax(lon) > 180.0:
        lon = ((lon + 180.0) % 360.0) - 180.0
    return lon


def _ranges_overlap(a_min: float, a_max: float, b_min: float, b_max: float, tol: float = 1e-6) -> bool:
    return (a_min <= b_max + tol) and (b_min <= a_max + tol)


def _validate_subset_bbox(ds: xr.Dataset, *, min_lon: float, max_lon: float, min_lat: float, max_lat: float) -> None:
    lat = np.asarray(ds["latitude"].values, dtype=np.float64)
    lon = _normalize_lon_values(ds["longitude"].values)

    d_lat_min, d_lat_max = float(np.nanmin(lat)), float(np.nanmax(lat))
    d_lon_min, d_lon_max = float(np.nanmin(lon)), float(np.nanmax(lon))

    ok_direct = _ranges_overlap(d_lat_min, d_lat_max, min_lat, max_lat) and _ranges_overlap(d_lon_min, d_lon_max, min_lon, max_lon)
    ok_swapped = _ranges_overlap(d_lon_min, d_lon_max, min_lat, max_lat) and _ranges_overlap(d_lat_min, d_lat_max, min_lon, max_lon)

    if ok_direct:
        return
    if ok_swapped:
        raise ValueError(
            "Subset bbox likely provided with lon/lat swapped. "
            f"Requested lon=[{min_lon},{max_lon}] lat=[{min_lat},{max_lat}] but got "
            f"lon≈[{d_lon_min:.3f},{d_lon_max:.3f}] lat≈[{d_lat_min:.3f},{d_lat_max:.3f}]."
        )
    raise ValueError(
        "Subset bbox does not match requested region. "
        f"Requested lon=[{min_lon},{max_lon}] lat=[{min_lat},{max_lat}] but got "
        f"lon≈[{d_lon_min:.3f},{d_lon_max:.3f}] lat≈[{d_lat_min:.3f},{d_lat_max:.3f}]."
    )


def _stack_depth_files(paths: list[str], depth_values: list[float]) -> xr.Dataset:
    datasets = []
    for p in paths:
        datasets.append(xr.open_dataset(p))

    # Concatenate along depth; ensure a consistent depth coord.
    stacked = xr.concat(datasets, dim="depth")
    stacked = stacked.assign_coords(depth=("depth", depth_values))
    return stacked


def fetch_and_merge_copernicus_data(username, password,
                                    minimum_longitude, maximum_longitude,
                                    minimum_latitude, maximum_latitude,
                                    start_datetime, end_datetime,
                                    minimum_depth, maximum_depth,
                                    output_filename,
                                    depths: list[float] | None = None,
                                    fallback_to_depth_list: bool = True,
                                    basic_dataset_id: str = "cmems_mod_glo_phy_my_0.083deg_P1D-m",
                                    include_tides: bool = False,
                                    tide_time_align: str = "nearest",
                                    tide_depth_profile: str = "broadcast",
                                    tide_z0_m: float = 50.0,
                                    tide_zmax_m: float = 200.0,
                                    output_directory: str | Path | None = None,
                                    overwrite: bool = False,
                                    keep_intermediate: bool = False):

    username = _env_or_value(username, "COPERNICUSMARINE_USERNAME")
    password = _env_or_value(password, "COPERNICUSMARINE_PASSWORD")
    if not username or not password:
        raise RuntimeError(
            "Missing Copernicus Marine credentials. "
            "Pass username/password or set COPERNICUSMARINE_USERNAME and COPERNICUSMARINE_PASSWORD."
        )

    base_dir = Path(__file__).resolve().parent
    out_dir = Path(output_directory) if output_directory is not None else (base_dir / "Data" / "GOPAF")
    out_dir.mkdir(parents=True, exist_ok=True)

    basic_path = out_dir / "cmems_basic_phy_output_data.nc"
    uv_path = out_dir / "cmems_detailed_uv_output_data.nc"
    combined_path = out_dir / output_filename

    def subset_basic(min_d: float, max_d: float, out_path: str, *, selection: str = "inside"):
        copernicusmarine.subset(
            dataset_id=basic_dataset_id,
            username=username,
            password=password,
            variables=["so", "thetao", "uo", "vo", "zos"],
            minimum_longitude=minimum_longitude,
            maximum_longitude=maximum_longitude,
            minimum_latitude=minimum_latitude,
            maximum_latitude=maximum_latitude,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            minimum_depth=min_d,
            maximum_depth=max_d,
            coordinates_selection_method=selection,
            output_filename=out_path,
            overwrite=overwrite,
        )

    def subset_uv(min_d: float, max_d: float, out_path: str, *, selection: str = "inside"):
        copernicusmarine.subset(
            dataset_id="cmems_mod_glo_phy_anfc_merged-uv_PT1H-i",
            username=username,
            password=password,
            variables=["utide", "utotal", "vtide", "vtotal"],
            minimum_longitude=minimum_longitude,
            maximum_longitude=maximum_longitude,
            minimum_latitude=minimum_latitude,
            maximum_latitude=maximum_latitude,
            start_datetime=start_datetime,
            end_datetime=end_datetime,
            minimum_depth=min_d,
            maximum_depth=max_d,
            coordinates_selection_method=selection,
            output_filename=out_path,
            overwrite=overwrite,
        )

    # 1) Try depth range subset (default behavior).
    if depths is None:
        subset_basic(minimum_depth, maximum_depth, str(basic_path))
        if include_tides:
            subset_uv(minimum_depth, maximum_depth, str(uv_path))

        # Validate that the subset bbox matches the requested region; fail fast if not.
        _validate_subset_bbox(
            xr.open_dataset(basic_path),
            min_lon=minimum_longitude,
            max_lon=maximum_longitude,
            min_lat=minimum_latitude,
            max_lat=maximum_latitude,
        )
        if include_tides:
            _validate_subset_bbox(
                xr.open_dataset(uv_path),
                min_lon=minimum_longitude,
                max_lon=maximum_longitude,
                min_lat=minimum_latitude,
                max_lat=maximum_latitude,
            )

        basic_depth_n = _open_depth_count(str(basic_path))

        # Some users observed depth range returning only one level for the *basic* dataset; optionally fall back.
        # Tides are frequently surface-only and should not force a depth fallback.
        if fallback_to_depth_list and (basic_depth_n <= 1):
            # Pick a small, task-relevant depth set if user didn't provide one.
            depths = [0.49402499198913574, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0]

    # 2) If depths provided (or inferred), request per-depth and stack.
    if depths is not None:
        basic_parts: list[str] = []
        basic_depths_ok: list[float] = []
        requested_depths_ok: list[float] = []

        for d in depths:
            tag = str(d).replace(".", "p")
            b = out_dir / f"cmems_basic_phy_depth_{tag}.nc"

            try:
                # Per-depth requests may not match dataset depth coordinates exactly; use nearest selection.
                subset_basic(d, d, str(b), selection="nearest")
                if _open_depth_count(str(b)) >= 1:
                    ds_b = xr.open_dataset(b)
                    actual_depth = float(ds_b["depth"].values[0]) if "depth" in ds_b.coords else float(d)
                    basic_parts.append(str(b))
                    basic_depths_ok.append(actual_depth)
                    requested_depths_ok.append(float(d))
            except Exception as e:
                print(f"Warning: basic subset failed for depth={d}: {e}")

        if not basic_parts:
            raise RuntimeError(
                "Per-depth fallback produced no data. "
                "Try a smaller bbox/time window, or use a different depth list."
            )

        basic_phy_data = _stack_depth_files(basic_parts, basic_depths_ok)
        basic_phy_data.attrs["requested_depth_values_m"] = ",".join(str(d) for d in requested_depths_ok)
        basic_phy_data.attrs["actual_depth_values_m"] = ",".join(str(d) for d in basic_depths_ok)
        if include_tides:
            subset_uv(minimum_depth, maximum_depth, str(uv_path))
            detailed_uv_data = xr.open_dataset(uv_path)
    else:
        basic_phy_data = xr.open_dataset(basic_path)
        detailed_uv_data = xr.open_dataset(uv_path) if include_tides else None

    # Merge datasets
    if detailed_uv_data is not None:
        # Align tides to the basic time grid to avoid exploding the merged time axis.
        if "time" in basic_phy_data.coords and "time" in detailed_uv_data.coords:
            if not np.array_equal(basic_phy_data["time"].values, detailed_uv_data["time"].values):
                if tide_time_align not in {"nearest", "linear"}:
                    raise ValueError(f"Unsupported tide_time_align={tide_time_align!r} (use 'nearest' or 'linear').")
                detailed_uv_data = detailed_uv_data.interp(time=basic_phy_data["time"], method=tide_time_align)

        # Tides dataset is often surface-only (depth=1). Broadcast to match the basic dataset depth grid if needed.
        if "depth" in basic_phy_data.dims and "depth" in detailed_uv_data.dims:
            if int(detailed_uv_data.sizes.get("depth", 0)) == 1 and int(basic_phy_data.sizes.get("depth", 0)) > 1:
                base_depth = basic_phy_data["depth"]
                detailed_uv_data = detailed_uv_data.isel(depth=0).expand_dims(depth=base_depth)
                detailed_uv_data = detailed_uv_data.assign_coords(depth=base_depth)

                depth0 = float(base_depth.values[0])
                z = np.asarray(base_depth.values, dtype=np.float64) - depth0
                z = np.maximum(z, 0.0)

                profile = tide_depth_profile
                if profile not in {"broadcast", "exp_decay", "linear"}:
                    raise ValueError(
                        f"Unsupported tide_depth_profile={profile!r} (use 'broadcast', 'exp_decay', or 'linear')."
                    )

                if profile == "broadcast":
                    w = np.ones_like(z, dtype=np.float64)
                elif profile == "exp_decay":
                    if tide_z0_m <= 0:
                        raise ValueError(f"tide_z0_m must be > 0, got {tide_z0_m}")
                    w = np.exp(-z / float(tide_z0_m))
                else:  # linear
                    if tide_zmax_m <= 0:
                        raise ValueError(f"tide_zmax_m must be > 0, got {tide_zmax_m}")
                    w = 1.0 - (z / float(tide_zmax_m))
                    w = np.clip(w, 0.0, 1.0)

                weights = xr.DataArray(w.astype(np.float32), dims=("depth",), coords={"depth": base_depth})
                for vname in ["utide", "vtide", "utotal", "vtotal"]:
                    if vname in detailed_uv_data.data_vars:
                        detailed_uv_data[vname] = detailed_uv_data[vname] * weights

                detailed_uv_data.attrs["tide_depth_profile"] = profile
                detailed_uv_data.attrs["tide_depth_profile_depth0_m"] = str(depth0)
                if profile == "exp_decay":
                    detailed_uv_data.attrs["tide_depth_profile_z0_m"] = str(float(tide_z0_m))
                if profile == "linear":
                    detailed_uv_data.attrs["tide_depth_profile_zmax_m"] = str(float(tide_zmax_m))
                detailed_uv_data.attrs["depth_broadcast_assumption"] = (
                    "Tide-related variables were available only at a single surface depth level and were "
                    "broadcast across all depth levels to match the basic dataset depth grid, optionally "
                    "scaled by a simple depth profile (see attrs: tide_depth_profile*)."
                )
        combined_data = xr.merge([basic_phy_data, detailed_uv_data])
    else:
        combined_data = basic_phy_data

    # Step 1: Fill missing values using rolling mean interpolation
    for var in combined_data.data_vars:
        if combined_data[var].isnull().any():
            combined_data[var] = combined_data[var].where(
                ~np.isnan(combined_data[var]),
                combined_data[var].rolling(latitude=3, longitude=3, center=True, min_periods=1).mean()
            )

    # Step 1b: If edge/corner NaNs remain (common in reanalysis products), fill with nearest values
    # along latitude/longitude. This avoids propagating NaNs into the combined environment grid.
    for var in combined_data.data_vars:
        if combined_data[var].isnull().any():
            da = combined_data[var]
            if "longitude" in da.dims:
                da = da.interpolate_na(dim="longitude", method="nearest", fill_value="extrapolate")
            if "latitude" in da.dims:
                da = da.interpolate_na(dim="latitude", method="nearest", fill_value="extrapolate")
            combined_data[var] = da

    # Step 2: Interpolate along the time dimension using linear interpolation
    combined_data = combined_data.interpolate_na(dim="time", method="linear")


    '''Check the nan'''
    # data_vars_nan = {var: combined_data[var].isnull().sum().values for var in combined_data.data_vars}
    #
    #
    # coords_nan = {coord: combined_data[coord].isnull().sum().values for coord in combined_data.coords if
    #               combined_data[coord].isnull().any()}
    #
    # dims_nan = {}
    # for dim in combined_data.dims:
    #     try:
    #         dims_nan[dim] = combined_data[dim].isnull().sum().values
    #     except AttributeError:
    #         dims_nan[dim] = 'Not Applicable'
    #
    # print("\n--- Detailed NaN Summary ---")
    # print("Data Variables (data_vars):")
    # for var, nan_count in data_vars_nan.items():
    #     print(f"  {var}: {nan_count} NaNs")
    # print("\nCoordinates (coords):")
    # for coord, nan_count in coords_nan.items():
    #     print(f"  {coord}: {nan_count} NaNs")
    # print("\nDimensions (dims):")
    # for dim, nan_info in dims_nan.items():
    #     print(f"  {dim}: {nan_info}")
    #
    # if any(count > 0 for count in data_vars_nan.values()) or coords_nan:
    #     raise ValueError("!!! NaN values still exist in the dataset after interpolation")

    combined_data.attrs["generated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    combined_data.attrs["source"] = "Copernicus Marine (CMEMS) via copernicusmarine Python client"
    combined_data.attrs["basic_dataset_id"] = basic_dataset_id
    combined_data.attrs["include_tides"] = str(bool(include_tides))
    if include_tides:
        combined_data.attrs["tide_time_align"] = tide_time_align
        combined_data.attrs["tide_depth_profile"] = tide_depth_profile
        combined_data.attrs["tide_z0_m"] = str(float(tide_z0_m))
        combined_data.attrs["tide_zmax_m"] = str(float(tide_zmax_m))
    if depths is not None:
        combined_data.attrs["depth_request_mode"] = "per-depth stack"
        combined_data.attrs["depth_values_m"] = ",".join(str(d) for d in depths)
    else:
        combined_data.attrs["depth_request_mode"] = "range"
        combined_data.attrs["minimum_depth_m"] = str(minimum_depth)
        combined_data.attrs["maximum_depth_m"] = str(maximum_depth)

    # Write a compact file for repeated iteration.
    to_write = combined_data.astype(np.float32)
    encoding: dict[str, dict] = {}
    for v in to_write.data_vars:
        encoding[v] = {"zlib": True, "complevel": 4, "dtype": "float32"}
    to_write.to_netcdf(combined_path, encoding=encoding)

    if not keep_intermediate:
        try:
            if basic_path.exists():
                basic_path.unlink()
        except Exception:
            pass
        try:
            if uv_path.exists():
                uv_path.unlink()
        except Exception:
            pass
        if depths is not None:
            for p in basic_parts:
                try:
                    Path(p).unlink()
                except Exception:
                    pass

    print(combined_data)


def get_shapes(file_path):
    ds = xr.open_dataset(file_path)

    for var_name in ds.data_vars:
        data = ds[var_name]
        shape_details = ', '.join([f"{dim_name}: {dim_size}" for dim_name, dim_size in zip(data.dims, data.shape)])
        print(f"Variable '{var_name}' shape: {data.shape} ({shape_details})")


def visualize_each_variable(dataset, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for var_name in dataset.data_vars:
        data = dataset[var_name]
        plt.figure(figsize=(16, 10), dpi=200)  # Larger figure size and higher DPI for better clarity
        plt.pcolormesh(data.longitude, data.latitude, data.isel(time=0, depth=0), shading='auto')
        plt.colorbar(label=f'{var_name} ({data.attrs.get("units", "unknown unit")})')
        plt.title(f'{data.attrs.get("long_name", var_name)}')
        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        file_path = os.path.join(output_directory, f"{var_name}.png")
        plt.savefig(file_path)
        plt.close()

def create_combined_image(image_directory, output_path):
    image_files = [os.path.join(image_directory, file) for file in os.listdir(image_directory) if file.endswith('.png')]
    images = [Image.open(file) for file in image_files]

    widths, heights = zip(*(img.size for img in images))
    total_width = max(widths) * 2
    total_height = (len(images) + 1) // 2 * max(heights)

    combined_image = Image.new('RGB', (total_width, total_height), (255, 255, 255))

    x_offset = 0
    y_offset = 0
    for i, img in enumerate(images):
        combined_image.paste(img, (x_offset, y_offset))
        if i % 2 == 1:
            x_offset = 0
            y_offset += img.height
        else:
            x_offset += img.width

    combined_image.save(output_path)
