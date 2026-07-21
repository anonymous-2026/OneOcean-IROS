# OneOcean Environment Dataset Contract

This document defines the public boundary between an environment product and the OneOcean benchmark. The validator accepts a small set of aliases, reports the resolved mapping, and does not silently infer missing units or mask semantics.

## Canonical schema

| Item | Required | Canonical name | Accepted aliases | Dimensions | Units / meaning |
|---|---:|---|---|---|---|
| Latitude | Yes | `latitude` | `lat` | `latitude` | `degrees_north`, strictly ascending |
| Longitude | Yes | `longitude` | `lon` | `longitude` | `degrees_east`, strictly ascending |
| Time | No | `time` | -- | `time` | CF-compatible time coordinate |
| Depth | No | `depth` | -- | `depth` | meters, positive downward |
| Eastward current | Yes | selected pair | `utotal`, `uo`, `water_u`, or `u` | normally `time × depth × latitude × longitude` | m/s, positive east |
| Northward current | Yes | selected pair | `vtotal`, `vo`, `water_v`, or `v` | same as eastward current | m/s, positive north |
| Terrain | Yes | `elevation` | `bathymetry`, `terrain_elevation` | `latitude × longitude` | meters relative to sea surface, positive upward |
| Validity mask | Yes | `land_mask` | `invalid_mask`, `mask` | `latitude × longitude` | 0 valid water; 1 invalid terrain, land, or NoData |

Current pairs are resolved in this order: `utotal/vtotal`, `uo/vo`, `water_u/water_v`, then `u/v`. Use `--u-var` and `--v-var` when a product has different names or when the automatic priority is not desired. Tide components (`utide/vtide`), salinity (`so`), temperature (`thetao`), sea-surface height (`zos`), and pollution fields are optional.

The `land_mask` is a validity mask, not a guaranteed coastline classification. A terrain crop can also mark out-of-range or NoData cells. The merge step fills invalid `elevation` cells with zero for storage, so consumers must apply the mask; zero elevation at a masked cell is not a physical seabed measurement.

Single-depth products are valid 2D-current snapshots over time, but they are not evidence of a resolved 3D current field. The `tiny` product intentionally uses the near-surface layer. The `scene` and `public` products retain the actual depth levels returned by Copernicus Marine. Tide-enabled products use an explicitly documented engineering depth profile and must not be interpreted as a validated 3D tidal model.

## Coordinates and simulation frame

NetCDF current components are east/north. OneOcean's core frame is `[x, y, z] = [east, depth-down, north]` in meters. Longitude maps to local `x`, latitude maps to local `z`, and the local geographic origin is `(latitude_min, longitude_min)`. The current pair therefore maps to `[u, 0, v]`. ROS uses ENU, so the optional bridge maps core `[x, y, z]` to ROS `[x, z, -y]`.

## Provenance

A release product should record, where applicable:

- source dataset and product identifier;
- requested and actual region, time interval, and depth levels;
- whether tides are included and how they are aligned and depth-profiled;
- coordinate repair, interpolation, extrapolation, and missing-value handling;
- generation time and software/environment versions.

Newly generated OneOcean products include portable source file names in NetCDF attributes and relative paths in `variant.json`; private absolute workspace paths are not part of the release contract.

## Validate a product

Install the data-pipeline dependencies from `requirements.txt`, then run:

```bash
python3 tools/validate_environment_product.py \
  Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc \
  --json-out runs/validation/scene.json
```

The exit code is `0` for `PASS` and `PASS_WITH_WARNINGS`, and `2` for `FAIL`. Missing units and provenance produce warnings by default. Use `--strict-units` to make unexpected units fail validation.

For a custom pair:

```bash
python3 tools/validate_environment_product.py custom_environment.nc \
  --u-var east_velocity --v-var north_velocity \
  --strict-units --json-out runs/validation/custom.json
```

The complete JSON report contains dimensions, canonical mappings, sampled numeric ranges, NaN/Inf fractions, coordinate monotonicity, variable attributes, and machine-readable issue codes.

## NetCDF-to-benchmark round trip

The standard release files use current dimensions `time × depth × latitude × longitude`. Export one fixed slice and compare it with its NetCDF source:

```bash
python3 -m benchmark_core.cli.export_drift_cache \
  --nc Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc \
  --u-var utotal --v-var vtotal --time-index 0 --depth-index 0 \
  --out runs/benchmark_core/_cache/drift_scene_t0_d0.npz

python3 tools/validate_environment_product.py \
  Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc \
  --u-var utotal --v-var vtotal \
  --drift-npz runs/benchmark_core/_cache/drift_scene_t0_d0.npz \
  --time-index 0 --depth-index 0
```

The round-trip check requires identical latitude/longitude arrays and compares both current components at absolute tolerance `1e-6` by default.

## Standard variants

The current generated release set is:

| Variant | Region and period | Depth | Grid / time | Combined NetCDF | Full variant directory | Intended use |
|---|---|---|---|---:|---:|---|
| `tiny` | 42.1--42.7 N, 71.2--70.2 W; 2025 | requested 0--1 m; one returned layer at about 0.494 m | 62 × 101; 357 times | 28.97 MB | about 45 MB | tests, demos, browser assets |
| `scene` | 32--33 N, 66.5--65.5 W; Dec. 2025 | 0--200 m; 26 layers | 240 × 240; 23 times | 689.88 MB | about 660 MiB | high-resolution simulation grounding |
| `public` | 30--40 N, 72--62 W; 2025 | 0--200 m; 26 layers | 41 × 41; 357 times | 175.37 MB | about 1.2 GiB | broad-area public distribution |

File bytes and directory allocation answer different questions: the NetCDF column reports the merged product itself, while the directory includes terrain crops, source subsets, metadata, and derived assets. Existing products generated before this contract pass with explained attribute warnings; regeneration writes the missing units and mask semantics.

## Run a validated episode

After exporting the cache, use the normal benchmark CLI and require run validation:

```bash
python3 -m benchmark_core.cli.run \
  --drift-npz runs/benchmark_core/_cache/drift_scene_t0_d0.npz \
  --task go_to_goal_current --difficulty medium --controller go_to_goal \
  --pollution-model gaussian --n-agents 1 --seed 0 \
  --dynamics-model 6dof --constraint-mode hard --bathy-mode hard --validate
```
