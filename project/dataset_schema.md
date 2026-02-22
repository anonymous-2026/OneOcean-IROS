# OneOcean Combined Dataset Schema (Contract)

This document is the schema/contract for `combined_environment.nc` produced by the OneOcean pipeline.

It is written for:
- Lane A (simulation/tasks) to load data safely,
- Lane C (experiments) to ensure results are comparable across variants,
- Lane E (demo/web) to visualize fields consistently,
- Lane B/F (paper/release) to describe what is included and what is optional.

---

## 1) Files and default locations

### Canonical combined (default pipeline output)
- `OceanEnv/Data_pipeline/Data/Combined/combined_environment.nc`

### Variants (recommended for most users)
- Root: `OceanEnv/Data_pipeline/Data/Combined/variants/`
- `tiny`: `.../variants/tiny/combined/combined_environment.nc`
- `scene`: `.../variants/scene/combined/combined_environment.nc`
- `public`: `.../variants/public/combined/combined_environment.nc`

Each variant folder also contains:
- `variant.json` (parameters + provenance)
- `water/combined_gopaf_data.nc` (CMEMS subset after merging/tide alignment)
- `terrain/filtered_data.tif` (cropped & filtered GEBCO GeoTIFF)

### Consumer default path resolution

The S1 simulation code resolves datasets as:
- explicit `--dataset-path`, or
- `--variant {tiny,scene,public}` under:
  - `ONEOCEAN_VARIANTS_ROOT/<variant>/combined/combined_environment.nc`

If `ONEOCEAN_VARIANTS_ROOT` is not set, it defaults to the workspace layout above.

Note:
- Some workspaces keep generated datasets outside the repo (e.g., `../OceanEnv/...`). Others generate them inside the repo.
- The default resolver prefers an in-repo `OceanEnv/.../variants` folder if it exists; otherwise it falls back to the workspace-sibling layout.

### Quick inspection tool (no xarray)

To inspect a combined file (dims/vars/tide presence) without xarray:
```bash
python -m pip install -r requirements.txt

python tools/inspect_combined.py --variant scene
python tools/inspect_combined.py --path /abs/path/to/combined_environment.nc
```

---

## 2) Coordinates and dimensions

Expected coordinates (always present):
- `latitude`: 1D (ascending)
- `longitude`: 1D (ascending; typically in `[-180, 180]`)
- `time`: 1D (datetime64)
- `depth`: 1D (meters; positive downward; reanalysis depth levels)

Expected core dimensions:
- `time`, `depth`, `latitude`, `longitude`

The merged output standardizes variable dimension order:
- 3D fields: `(time, depth, latitude, longitude)`
- 2D fields: `(latitude, longitude)`

---

## 3) Variables

### Required (consumers may assume these exist)

Coordinates (1D):
- `time`, `depth`, `latitude`, `longitude`

Core currents (3D; float32):
- `uo`: eastward velocity
- `vo`: northward velocity

Terrain (2D):
- `elevation` (float32): bathymetry/topography on the merged grid
- `land_mask` (uint8): mask of terrain pixels that were invalid (`NaN`) before filling

### Present-by-default today (but do not hard-require unless you need them)

Ocean state (3D; float32):
- `so`: salinity
- `thetao`: potential temperature
- `zos`: sea surface height

### Optional (consumers must check existence)

Tides (3D; float32; may be missing in `public` and canonical outputs):
- `utide`, `vtide`: tidal velocity components
- `utotal`, `vtotal`: total velocity components from tide product (if included)

Notes:
- The tide source product is surface-only; when included, it is aligned to the basic dataset time grid and broadcast to depth using an engineering approximation (see `DATA_PIPELINE_LOG.md`).

---

## 4) Missing data and masks (critical semantics)

### `land_mask` semantics

`land_mask` marks pixels where the terrain crop pipeline produced `NaN` (land/out-of-range/NoData).

Important:
- `land_mask == 1` (or non-zero) means **invalid terrain pixel** in the original crop.
- `elevation` is filled with `0.0` at these pixels to keep a dense array.
- Therefore, **do not interpret `elevation == 0` as land**. Always consult `land_mask`.

### NaNs in ocean variables

The water subset stage applies imputation/smoothing to reduce NaNs before merge; remaining NaNs after merge should be treated as invalid and handled by consumers (e.g., replace with zeros or terminate episodes).

---

## 5) Dtypes and storage

Default dtypes written by the merge:
- Ocean/tide variables: `float32`
- `elevation`: `float32`
- `land_mask`: `uint8`

NetCDF writing uses compression and chunking (see `OceanEnv/Data_pipeline/Combine.py`).

---

## 6) Current default artifact snapshot (2026-02-22, workspace)

This snapshot is for debugging and sanity checks; regenerate locally as needed.

### Canonical combined
- dims: `time=30`, `depth=26`, `latitude=240`, `longitude=240`
- vars: `so thetao uo vo zos elevation land_mask`
- file size: ~331 MiB

### Variant: `tiny`
- dims: `time=3`, `depth=18`, `latitude=48`, `longitude=48`
- vars: `so thetao uo vo zos utide utotal vtide vtotal elevation land_mask`
- file size: ~1.7 MiB

### Variant: `scene`
- dims: `time=30`, `depth=26`, `latitude=240`, `longitude=240`
- vars: `so thetao uo vo zos utide utotal vtide vtotal elevation land_mask`
- file size: ~569 MiB

### Variant: `public`
- dims: `time=48`, `depth=26`, `latitude=41`, `longitude=41`
- vars: `so thetao uo vo zos elevation land_mask` (tides excluded by default)
- file size: ~22 MiB
