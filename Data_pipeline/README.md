# Data Pipeline

This directory is the canonical environment-data pipeline used by the project.
It builds `combined_environment.nc` from terrain and ocean-physics sources, then exports paper-facing dataset variants.

## Scope

- `GeoTIFF_Data.py`: crop and preprocess GEBCO bathymetry terrain
- `GOPAF_Data.py`: fetch and assemble CMEMS ocean variables
- `Combine.py`: align terrain and water fields onto one grid
- `run_pipeline.py`: end-to-end generation of the canonical combined dataset
- `generate_variants.py`: produce `tiny`, `scene`, and `public` release variants

## Outputs

Main outputs are written under `Data/`:

- `Data/Combined/combined_environment.nc`: canonical merged environment dataset
- `Data/Combined/variants/tiny/combined/combined_environment.nc`: smallest debug/web variant
- `Data/Combined/variants/scene/combined/combined_environment.nc`: scene-building variant with richer fidelity
- `Data/Combined/variants/public/combined/combined_environment.nc`: lighter public-release variant

The detailed generation log, design decisions, and size summary are recorded in `../DATA_PIPELINE_LOG.md`.

## Quickstart

Generate the canonical combined dataset:

```bash
python Data_pipeline/run_pipeline.py --overwrite
```

Generate the standard paper variants:

```bash
python Data_pipeline/generate_variants.py --which tiny,scene,public --overwrite
```

Reuse existing downloads and regenerate only derived outputs:

```bash
python Data_pipeline/generate_variants.py --which tiny,scene,public --overwrite --reuse-existing
```

## Requirements

- `COPERNICUSMARINE_USERNAME`
- `COPERNICUSMARINE_PASSWORD`
- GEBCO GeoTIFF tiles available under `Data/`

## Dataset schema

Core variables in `combined_environment.nc`:

- `elevation`: bathymetry / terrain on the `latitude × longitude` grid
- `land_mask`: invalid-terrain mask; use as a validity mask, not a literal land label
- `so`, `thetao`, `uo`, `vo`: water variables on `time × depth × latitude × longitude`
- `zos`: surface-height field on `time × latitude × longitude`

Optional tide-enabled variants also include:

- `utide`, `vtide`
- `utotal`, `vtotal`

## Notes

- `scene` is the main dataset used for simulator grounding.
- `public` is the lighter distribution target.
- `tiny` is primarily for sanity checks, demos, and browser-facing assets.
- Depth handling, tide alignment, and release-policy decisions are documented in `../DATA_PIPELINE_LOG.md`.
