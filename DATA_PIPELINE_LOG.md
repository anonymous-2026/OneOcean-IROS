# Data & Pipeline Optimization Log

This file is the single source of truth for **what changed**, **why**, and **how to reproduce** the ocean data pipeline outputs.

## Where this log lives
- Workspace copy: `DATA_PIPELINE_LOG.md`
- Code repo copy (versioned): `oneocean(iros-2026-code)/DATA_PIPELINE_LOG.md`

## 2026-02-21/22 — Multi-depth fix + bbox correctness + new combined output

### Goal (why we touched this at all)
- We need a reliable pipeline that produces `combined_environment.nc` suitable for:
  - robotics simulation scene/task construction (needs realistic 3D ocean fields, not a single surface layer),
  - experiment reproducibility,
  - and an eventual public dataset release in multiple sizes.

### Root problems found (what was broken)
1) **Single-depth return under our original request configuration**:
   - Under our then-current request configuration (dataset_id/bbox/time/vars/client version), dataset
     `cmems_mod_glo_phy_anfc_0.083deg_PT1H-m` returned only **one** depth coordinate:
     - `depth=[0.49402499198913574]` (meters)
   - Evidence (from `tools/test_cmems_depths.py`, 2026-02-22):
     - `range depth values: [0.49402499198913574]`
     - `single depth values: [0.49402499198913574]`
   - Consequence: any “3D” environment built from this source becomes effectively 2.5D (single depth).

2) **Historical bbox bug: lon/lat got swapped**, producing a scientifically incorrect water dataset:
   - Old `OceanEnv/Data_pipeline/Data/GOPAF/combined_gopaf_data.nc` contained:
     - latitude around `[-66, -65]`
     - longitude around `[32, 33]`
   - But the intended region was (as used by terrain crop and pipeline defaults):
     - latitude `[32, 33]`, longitude `[-66.5, -65.5]`
   - Consequence: pipeline could generate a “clean” combined file that is actually **wrong location water fields mapped onto the intended terrain region**.

### Fixes implemented (how we corrected it)
1) **Switched to a multi-depth CMEMS product for basic physics + currents**
   - Chosen dataset (multi-depth, multi-variable): `cmems_mod_glo_phy_my_0.083deg_P1D-m`
     - variables used: `so`, `thetao`, `uo`, `vo`, `zos`
     - confirmed multi-depth (e.g., 26 depth levels within ~0.494–186m for our test region)

2) **Strict bbox validation added to prevent “quietly wrong” outputs**
   - Water subset files are validated against the requested bbox.
   - If they look swapped (lon range overlaps requested lat, and vice versa), the pipeline fails fast.

3) **Removed interactive login dependency**
   - `copernicusmarine.login()` can prompt to overwrite credentials files.
   - Pipeline now passes credentials directly to `copernicusmarine.subset()` to avoid blocking.
   - Credentials are still sourced from environment variables:
     - `COPERNICUSMARINE_USERNAME`
     - `COPERNICUSMARINE_PASSWORD`

4) **New combined dataset successfully regenerated**
   - Output: `OceanEnv/Data_pipeline/Data/Combined/combined_environment.nc`
   - Key properties (scene-quality default):
     - dims: `time=30`, `depth=26`, `latitude=240`, `longitude=240`
     - vars: `so`, `thetao`, `uo`, `vo`, `zos`, `elevation`, `land_mask`
     - strict coordinate check enabled (prevents bbox mismatch)
   - Note: `land_mask` marks pixels where the terrain crop had NaNs (land/out-of-range); elevation is filled with `0.0` on those pixels.

### Current outputs (what to use right now)
- Canonical (default) combined:
  - `OceanEnv/Data_pipeline/Data/Combined/combined_environment.nc` (multi-depth, no tides by default)
- Canonical (default) water subset used by the canonical combined:
  - `OceanEnv/Data_pipeline/Data/GOPAF/combined_gopaf_data.nc`
- Multi-size variants:
  - `OceanEnv/Data_pipeline/Data/Combined/variants/tiny/combined/combined_environment.nc`
  - `OceanEnv/Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc`
  - `OceanEnv/Data_pipeline/Data/Combined/variants/public/combined/combined_environment.nc`

### Notes for paper / experiments (what to cite or describe)
- The “single depth” issue was not a bug in interpolation; it was a property of the previously used CMEMS dataset selection.
- The bbox swap bug is a correctness issue that can invalidate conclusions if not caught.
- We now generate multi-depth `uo/vo` fields, enabling depth-dependent flow in simulation.
- Remaining NaNs in CMEMS reanalysis products are handled as **imputation/smoothing** before writing water subsets:
  - rolling mean (3x3) over lat/lon, then nearest-fill along lon/lat for edges/corners.

### Repro commands (safe; no credentials printed)
- Validate depth behavior of the old dataset:
  - After activating your environment: `python tools/test_cmems_depths.py`
- Fetch correct water subset (multi-depth):
  - Run `OceanEnv/Data_pipeline/run_pipeline.py` (defaults updated)
- Generate tiny/scene/public variants:
  - After activating your environment: `python OceanEnv/Data_pipeline/generate_variants.py --which tiny,scene,public --overwrite`

### One-click generation & reuse
- Canonical (default) one-click:
  - `python OceanEnv/Data_pipeline/run_pipeline.py --overwrite`
- Canonical reuse (skip refetch):
  - `python OceanEnv/Data_pipeline/run_pipeline.py --skip-water-fetch --water-file OceanEnv/Data_pipeline/Data/GOPAF/combined_gopaf_data.nc --overwrite`
- Variants one-click:
  - `python OceanEnv/Data_pipeline/generate_variants.py --which tiny,scene,public --overwrite`
- Variants reuse (skip recrop/refetch if cached files exist in each variant folder):
  - `python OceanEnv/Data_pipeline/generate_variants.py --which tiny,scene,public --overwrite --reuse-existing`

### Depth fallback note (per-depth stacking)
- If the basic dataset depth-range request ever returns only one level, the pipeline can fall back to requesting a list of depths and stacking them.
- Risk/mitigation:
  - When requesting a single depth, CMEMS may return the **nearest available** depth level (not the exact requested number).
  - We record both the **requested depth list** and the **actual returned depth values** in the resulting water file attributes when this mode is used.

---

## 2026-02-21/22 — Variants + tides + coarse-grid public release

### Tides decision (what we do and why)
- We include tides for **tiny** and **scene** variants (short time windows, manageable download).
- For **public** variant we currently **exclude** tides by default:
  - tide product is hourly and would massively increase download volume for multi-year spans,
  - the public variant targets “easy-to-download” (coarse spatial grid + monthly time).
- Implementation detail:
  - tide dataset (`cmems_mod_glo_phy_anfc_merged-uv_PT1H-i`) is surface-only (`depth=1`), so we broadcast it to all depths and align it to the basic dataset time grid.
  - Limitation: time alignment uses **nearest sampling** onto the basic dataset `time` grid (daily/monthly).
    This is an engineering trade-off for size/speed and is **not suitable** if you need realistic high-frequency tidal cycles in simulation.
  - Limitation: broadcasting tides to all depths is a strong engineering approximation and does **not** guarantee physically accurate 3D tidal structure.

### Tide vertical profile upgrade (depth model)
- We added an optional, lightweight tide depth model that applies when tide data is surface-only and must be broadcast:
  - `broadcast`: constant with depth (previous behavior)
  - `exp_decay`: multiply by `exp(-(z-z0)/tide_z0_m)` (engineering attenuation with depth)
  - `linear`: multiply by `max(0, 1-(z-z0)/tide_zmax_m)`
- Current variant defaults (2026-02-22):
  - `tiny`: `exp_decay` with `tide_z0_m=30m`
  - `scene`: `exp_decay` with `tide_z0_m=50m`
  - `public`: tides disabled

### Variant parameters (current defaults)
- `tiny`:
  - bbox: lat `[32.40, 32.60]`, lon `[-66.20, -66.00]`
  - time: `2024-06-01 .. 2024-06-03` (daily)
  - depth: `0 .. 50m`
  - dataset_id: `cmems_mod_glo_phy_my_0.083deg_P1D-m`
  - tides: enabled
- `scene`:
  - bbox: lat `[32.0, 33.0]`, lon `[-66.5, -65.5]`
  - time: `2024-06-01 .. 2024-06-30` (daily)
  - depth: `0 .. 200m`
  - dataset_id: `cmems_mod_glo_phy_my_0.083deg_P1D-m`
  - tides: enabled
- `public`:
  - bbox: lat `[30.0, 40.0]`, lon `[-72.0, -62.0]`
  - time: `2021-01-01 .. 2024-12-01` (monthly; constrained by dataset availability)
  - depth: `0 .. 200m`
  - dataset_id: `cmems_mod_glo_phy_my_0.083deg_P1M-m`
  - tides: disabled
  - combined grid: `target_res_deg=0.25`

### Multi-variant outputs (what to use for what)
- `tiny`: fast iteration/debug (small bbox + short time + shallow depth)
- `scene`: scene/task construction (high-res terrain grid + multi-depth + daily time + optional tides)
- `public`: publishable (bigger bbox + multi-year monthly + coarse 0.25° grid; tides excluded)

### Output grid & masks
- `Combine.interpolate_and_merge_fast` now supports:
  - `target_res_deg` to downsample the terrain/output grid (used by `public`)
  - `land_mask` (uint8) to mark where terrain crop had NaNs (typically land/out-of-range)
  - elevation NaNs are filled with 0.0 and the original NaN locations are retained in `land_mask`

### Where variant files are written
- `OceanEnv/Data_pipeline/Data/Combined/variants/<variant>/combined/combined_environment.nc`
- Each variant folder also contains:
  - `terrain/filtered_data.tif`
  - `water/combined_gopaf_data.nc`
  - `variant.json` (parameters + provenance)

### Cleanup / archive
- Obsolete artifacts (bad bbox water files, invalid combined outputs, one-off intermediate downloads, `.DS_Store`, caches) were deleted after validation.
- Per-variant intermediate subset files are not required; we keep only `water/combined_gopaf_data.nc` for each variant.
- Old `combined_environment.nc.bak.*` backups under each variant were deleted to avoid wasting disk space and confusing “latest” selection.

### Disk usage snapshot (as of 2026-02-22)
- Canonical combined file: `OceanEnv/Data_pipeline/Data/Combined/combined_environment.nc` ≈ 331 MiB
- `tiny` variant folder: `OceanEnv/Data_pipeline/Data/Combined/variants/tiny` ≈ 2 MiB (combined file ≈ 1.7 MiB)
- `scene` variant folder: `OceanEnv/Data_pipeline/Data/Combined/variants/scene` ≈ 572 MiB (combined file ≈ 569 MiB)
- `public` variant folder: `OceanEnv/Data_pipeline/Data/Combined/variants/public` ≈ 183 MiB (combined file ≈ 22 MiB)

### Environment snapshot (as of 2026-02-22)
- python: 3.13.11
- copernicusmarine: 2.3.0
- xarray: 2026.2.0
- netCDF4: 1.7.4
- rasterio: 1.5.0

### Redistribution / licensing note (public release)
- CMEMS/Copernicus data may have redistribution constraints. For a public release, we should confirm whether we:
  - publish only **derived** products (e.g., our `combined_environment.nc`) under allowed terms, and/or
  - publish **scripts + parameters** (with instructions to fetch from CMEMS) rather than redistributing raw subsets.
