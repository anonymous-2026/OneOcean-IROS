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
    ds = xr.open_dataset(path, engine="h5netcdf")
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
        datasets.append(xr.open_dataset(p, engine="h5netcdf"))

    # Concatenate along depth; ensure a consistent depth coord.
    stacked = xr.concat(datasets, dim="depth")
    stacked = stacked.assign_coords(depth=("depth", depth_values))
    return stacked


def fetch_and_merge_copernicus_data(
    username,
    password,
    minimum_longitude,
    maximum_longitude,
    minimum_latitude,
    maximum_latitude,
    start_datetime,
    end_datetime,
    minimum_depth,
    maximum_depth,
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
    keep_intermediate: bool = False,
):
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
    detailed_uv_data = None

    def _open_or_redownload(path: Path, *, subset_fn, label: str) -> xr.Dataset:
        """
        Copernicus downloads can occasionally leave a partially-written NetCDF file (e.g. HDF error on open).
        In that case, delete and redownload once to keep pipelines robust.
        """
        try:
            return xr.open_dataset(path, engine="h5netcdf")
        except OSError as e:
            msg = str(e)
            if ("NetCDF: HDF error" not in msg) and ("HDF error" not in msg):
                raise
            try:
                if path.exists():
                    path.unlink()
            except Exception:
                raise e
            subset_fn()
            return xr.open_dataset(path, engine="h5netcdf")

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

        _validate_subset_bbox(
            _open_or_redownload(
                basic_path,
                subset_fn=lambda: subset_basic(minimum_depth, maximum_depth, str(basic_path)),
                label="basic",
            ),
            min_lon=minimum_longitude,
            max_lon=maximum_longitude,
            min_lat=minimum_latitude,
            max_lat=maximum_latitude,
        )
        if include_tides:
            _validate_subset_bbox(
                _open_or_redownload(
                    uv_path,
                    subset_fn=lambda: subset_uv(minimum_depth, maximum_depth, str(uv_path)),
                    label="tides",
                ),
                min_lon=minimum_longitude,
                max_lon=maximum_longitude,
                min_lat=minimum_latitude,
                max_lat=maximum_latitude,
            )

        basic_depth_n = _open_depth_count(str(basic_path))

        # Some users observed depth range returning only one level for the *basic* dataset; optionally fall back.
        if fallback_to_depth_list and (basic_depth_n <= 1):
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
                subset_basic(d, d, str(b), selection="nearest")
                if _open_depth_count(str(b)) >= 1:
                    ds_b = xr.open_dataset(b, engine="h5netcdf")
                    actual_depth = float(ds_b["depth"].values[0]) if "depth" in ds_b.coords else float(d)
                    basic_parts.append(str(b))
                    basic_depths_ok.append(actual_depth)
                    requested_depths_ok.append(float(d))
            except Exception as e:
                print(f"Warning: basic subset failed for depth={d}: {e}")

        if not basic_parts:
            raise RuntimeError("No depth subsets succeeded; cannot proceed.")

        basic_phy_data = _stack_depth_files(basic_parts, basic_depths_ok)
        basic_phy_data.attrs["depth_request_mode"] = "per-depth stack"
        basic_phy_data.attrs["requested_depth_values_m"] = ",".join(map(str, requested_depths_ok))
        basic_phy_data.attrs["actual_depth_values_m"] = ",".join(map(str, basic_depths_ok))
        basic_phy_data.attrs["depth_values_m"] = ",".join(map(str, basic_depths_ok))
    else:
        basic_phy_data = xr.open_dataset(basic_path, engine="h5netcdf")
        basic_phy_data.attrs["depth_request_mode"] = "range"
        basic_phy_data.attrs["minimum_depth_m"] = str(float(minimum_depth))
        basic_phy_data.attrs["maximum_depth_m"] = str(float(maximum_depth))

    # 3) If tides requested, open and align to basic time grid.
    if include_tides:
        detailed_uv_data = xr.open_dataset(uv_path, engine="h5netcdf")

        # Align tides to the basic time grid to avoid exploding the merged time axis.
        if "time" in detailed_uv_data.coords and "time" in basic_phy_data.coords:
            if tide_time_align not in {"nearest", "linear"}:
                raise ValueError(f"Unsupported tide_time_align={tide_time_align!r} (use 'nearest' or 'linear').")
            detailed_uv_data = detailed_uv_data.interp(time=basic_phy_data["time"], method=tide_time_align)

        # If detailed data is surface-only but basic has multiple depths, broadcast with a simple profile.
        if ("depth" in basic_phy_data.coords) and ("depth" in detailed_uv_data.coords):
            if detailed_uv_data["depth"].size == 1 and basic_phy_data["depth"].size > 1:
                depth0 = float(detailed_uv_data["depth"].values[0])
                z = np.asarray(basic_phy_data["depth"].values, dtype=np.float64)
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
                else:
                    if tide_zmax_m <= 0:
                        raise ValueError(f"tide_zmax_m must be > 0, got {tide_zmax_m}")
                    w = 1.0 - (z / float(tide_zmax_m))
                    w = np.clip(w, 0.0, 1.0)

                detailed_uv_data = detailed_uv_data.squeeze("depth", drop=True)
                for vname in ["utide", "vtide", "utotal", "vtotal"]:
                    if vname in detailed_uv_data.data_vars:
                        detailed_uv_data[vname] = detailed_uv_data[vname].expand_dims(depth=basic_phy_data["depth"])
                        detailed_uv_data[vname] = detailed_uv_data[vname] * xr.DataArray(w, dims=["depth"])

                detailed_uv_data = detailed_uv_data.assign_coords(depth=basic_phy_data["depth"])
                detailed_uv_data.attrs["tide_depth_profile"] = profile
                detailed_uv_data.attrs["tide_depth_profile_depth0_m"] = str(depth0)
                if profile == "exp_decay":
                    detailed_uv_data.attrs["tide_depth_profile_z0_m"] = str(float(tide_z0_m))
                if profile == "linear":
                    detailed_uv_data.attrs["tide_depth_profile_zmax_m"] = str(float(tide_zmax_m))
                detailed_uv_data.attrs["tide_depth_broadcast_note"] = (
                    "Tide variables were surface-only and broadcast to basic depth grid, "
                    "scaled by a simple depth profile (see attrs: tide_depth_profile*)."
                )

    # 4) Merge datasets
    if include_tides and detailed_uv_data is not None:
        combined_data = xr.merge([basic_phy_data, detailed_uv_data], compat="override")
    else:
        combined_data = basic_phy_data

    combined_data.attrs["generated_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    combined_data.attrs["basic_dataset_id"] = basic_dataset_id
    combined_data.attrs["include_tides"] = str(bool(include_tides))
    if include_tides:
        combined_data.attrs["tide_time_align"] = tide_time_align
        combined_data.attrs["tide_depth_profile"] = tide_depth_profile
        combined_data.attrs["tide_z0_m"] = str(float(tide_z0_m))
        combined_data.attrs["tide_zmax_m"] = str(float(tide_zmax_m))

    # 5) Save combined file
    encoding: dict[str, dict] = {}
    for vname in combined_data.data_vars:
        encoding[vname] = {"compression": "gzip", "compression_opts": 4}

    to_write = combined_data
    if not keep_intermediate:
        # Avoid leaving gigantic intermediate copies around.
        for p in [basic_path, uv_path]:
            try:
                if p.exists():
                    p.unlink()
            except Exception:
                pass

    to_write.to_netcdf(combined_path, engine="h5netcdf", encoding=encoding)

    return str(combined_path)


def get_shapes(file_path):
    ds = xr.open_dataset(file_path, engine="h5netcdf")

    for var_name in ds.data_vars:
        data = ds[var_name]
        shape_details = ', '.join([f"{dim_name}: {dim_size}" for dim_name, dim_size in zip(data.dims, data.shape)])
        print(f"Variable '{var_name}' shape: {data.shape} ({shape_details})")


def visualize_each_variable(dataset, output_directory):
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for var_name in dataset.data_vars:
        data = dataset[var_name]
        plt.figure(figsize=(16, 10), dpi=200)
        plt.pcolormesh(data.longitude, data.latitude, data.isel(time=0, depth=0), shading='auto')
        plt.colorbar(label=f"{var_name} ({data.attrs.get('units', 'unknown unit')})")
        plt.title(f"{data.attrs.get('long_name', var_name)}")
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
