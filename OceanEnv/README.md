# OceanEnv (Data Pipeline + Web Dataset Export)

This folder contains the **Ocean environment dataset pipeline** (terrain + CMEMS ocean physics → `combined_environment.nc`) and a **web export** option (`.zarr`) for GitHub Pages in-browser loading/visualization.

## Quickstart

### 1) Credentials (Copernicus Marine)
Set env vars (do **not** hardcode):
- `COPERNICUSMARINE_USERNAME`
- `COPERNICUSMARINE_PASSWORD`

### 2) Generate standard variants (tiny/scene/public)
From this folder:
```bash
python generate_variants.py --which tiny,scene,public --overwrite
```

Notes:
- The daily CMEMS dataset used here currently clips time to the latest available day (observed max: **2025-12-23**), so requesting `2025-12-31` will be truncated.
- `scene` includes tides; `public` is faster and currently does **not** include tides.

### 3) Web export (Zarr) for GitHub Pages
For the (Boston nearshore) `tiny` variant:
```bash
python generate_variants.py --which tiny --overwrite --export-zarr --zarr-out-name oneocean_web_boston_surface.zarr
```

Dependencies for web export:
```bash
python -m pip install zarr numcodecs
```

The Zarr store is written under `Data/Combined/variants/tiny/web/oneocean_web_boston_surface.zarr` and is suitable for browser-side loading (e.g., using `zarr.js` + `xarray`-style processing in JS).

## CLI overrides (no script edits)

You can override bbox/time/depth/resolution directly:
```bash
python generate_variants.py --which public --bbox 30 40 -72 -62 --start 2025-01-01T00:00:00 --end 2025-12-31T00:00:00 --target-res-deg 0.25
```

Key flags:
- `--bbox LAT_MIN LAT_MAX LON_MIN LON_MAX`
- `--start`, `--end` (ISO datetime)
- `--min-depth`, `--max-depth`
- `--target-res-deg` (use `none` to disable)
- `--include-tides` / `--no-tides`
- `--export-zarr` (+ `--zarr-time-chunk`, `--zarr-keep-vars`)
- `--zarr-out-name` (override Zarr store name)

## Output layout

Variants are written under:
`Data/Combined/variants/<variant_name>/`

Each variant includes:
- `terrain/filtered_data.tif`
- `water/combined_gopaf_data.nc`
- `combined/combined_environment.nc`
- `variant.json` (includes requested params and **actual** output time range)

## Release naming (recommended)

The pipeline writes a canonical `combined/combined_environment.nc`. For public releases, we recommend renaming/copying the final artifacts to include:
- variant name (`scene` / `public` / `tiny`)
- actual available time range (CMEMS may truncate)
- key feature tags (e.g. `tides`, `surface`)

## Dataset schema (variables)

Core variables (always present in `combined_environment.nc`):
- `elevation` (meters; sea is negative): terrain/bathymetry on the `latitude×longitude` grid
- `land_mask` (0/1): invalid/no-data mask for terrain pixels (**not guaranteed to mean “land”**; treat as invalid mask)
- `so` (salinity), `thetao` (temperature), `uo` (eastward velocity), `vo` (northward velocity): 3D water variables on `time×depth×latitude×longitude`
- `zos` (sea surface height): 2D water variable on `time×latitude×longitude`

If tides are enabled (`scene` default, or `--include-tides`), additional variables are included:
- `utide`, `vtide`: tidal currents (approximate, see `generate_variants.py` tide settings)
- `utotal`, `vtotal`: total currents computed as `uo+utide`, `vo+vtide`

Coordinates:
- `time` (datetime64), `depth` (m), `latitude` (degrees_north), `longitude` (degrees_east)
