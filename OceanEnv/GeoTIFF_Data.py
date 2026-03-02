import glob
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds

# Folder path for GeoTIFF files
_HERE = Path(__file__).resolve().parent


def _pick_gebco_folder() -> Path:
    # Prefer newest available dataset; allow override via env var.
    env = os.environ.get("GEBCO_TIFF_DIR")
    if env:
        return Path(env)

    candidates = []
    for p in (_HERE / "Data").glob("gebco_*_sub_ice_topo_geotiff"):
        candidates.append(p)
    if not candidates:
        # Fallback to legacy folder name
        legacy = _HERE / "Data" / "gebco_2025_sub_ice_topo_geotiff"
        if legacy.exists():
            return legacy
        raise FileNotFoundError(
            "No GEBCO geotiff folder found under Data/. Expected something like "
            "Data/gebco_2025_sub_ice_topo_geotiff (or set GEBCO_TIFF_DIR)."
        )
    return sorted(candidates)[-1]


def Get_GeoTIFF_Data(lat_min, lat_max, lon_min, lon_max, elev_min, elev_max, save_path):
    """
    Extract a region from GEBCO GeoTIFF tiles and write:
      - filtered_data.tif
      - GeoTIFF_Terrain.png (quick viz)
    """
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)

    folder_path = _pick_gebco_folder()

    # Get all GeoTIFF files in the folder
    tiff_files = glob.glob(str(folder_path / "*.tif"))

    # Find all the matched GeoTIFF files
    matched_files = []
    for file in tiff_files:
        match = re.search(r'n(-?\d+(\.\d+)?)_s(-?\d+(\.\d+)?)_w(-?\d+(\.\d+)?)_e(-?\d+(\.\d+)?)', file)
        if match:
            n, s, w, e = map(float, [match.group(1), match.group(3), match.group(5), match.group(7)])
            if lat_min < n and lat_max > s and lon_min < e and lon_max > w:
                matched_files.append(file)
                print(f"match: n{n}_s{s}_w{w}_e{e}")

    if not matched_files:
        raise FileNotFoundError("No matching GeoTIFF files found for the specified region.")

    # Use the first match (tiles are large; bbox should fit within one tile for these use-cases)
    file_path = matched_files[0]
    print(f"Loading file: {file_path}")

    with rasterio.open(file_path) as src:
        # Read the data within the bounding box
        window = rasterio.windows.from_bounds(lon_min, lat_min, lon_max, lat_max, transform=src.transform)
        data = src.read(1, window=window)

        # Print sample (top-left) to help sanity check
        print("Sample elevation data from the top-left 5x5 pixels in the full file:")
        sample_window = rasterio.windows.Window(0, 0, 5, 5)
        sample_data = src.read(1, window=sample_window)
        print(sample_data)
        print("\n--------------------------------------------------\n")

        # Clip elevations; set out-of-range to NaN
        data = data.astype(np.float64)
        data[(data < elev_min) | (data > elev_max)] = np.nan

        print("Cropped elevation data statistics:")
        print(f"Minimum value (lowest point): {np.nanmin(data)} m")
        print(f"Maximum value (highest point): {np.nanmax(data)} m")
        print(f"Mean value: {np.nanmean(data):.2f} m")
        print(f"Data type: {data.dtype}")
        print("\n--------------------------------------------------\n")

        # Save as GeoTIFF
        transform = from_bounds(lon_min, lat_min, lon_max, lat_max, data.shape[1], data.shape[0])
        out_tif = save_path / "filtered_data.tif"
        profile = src.profile
        profile.update(
            {
                "height": data.shape[0],
                "width": data.shape[1],
                "transform": transform,
                "dtype": "float32",
                "count": 1,
                "compress": "lzw",
                "nodata": None,
            }
        )
        with rasterio.open(out_tif, "w", **profile) as dst:
            dst.write(np.asarray(data, dtype=np.float32), 1)

    # Save a quick visualization
    plt.figure(figsize=(10, 6), dpi=200)
    plt.imshow(data, cmap="terrain", origin="upper")
    plt.colorbar(label="Elevation (m)")
    plt.title("Terrain (filtered)")
    plt.tight_layout()
    plt.savefig(save_path / "GeoTIFF_Terrain.png")
    plt.close()

    # Optional CSV export for debugging
    df = pd.DataFrame(data)
    df.to_csv(save_path / "filtered_data.csv", index=False)

    return out_tif

