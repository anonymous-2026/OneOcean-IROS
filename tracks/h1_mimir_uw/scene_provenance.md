# H1 scene provenance — MIMIR-UW → reconstructed stage (Habitat-Sim)

This track uses **MIMIR-UW** as an upstream source of underwater RGB-D + pose sequences and reconstructs a textured mesh patch from a single frame (RGB + depth + pose) to serve as a Habitat-Sim stage.

## Upstream source

- Name: **MIMIR-UW**
- Upstream repo: `remaro-network/MIMIR-UW`
- Dataset host: Zenodo record `10406384` (e.g., `OceanFloor.zip`, `SandPipe.zip`)
- License: **GPL-3.0**

License handling policy for this project:
- Treat upstream assets as **internal-only** references.
- Do **not** commit MIMIR-UW datasets or extracted files to our public repos.
- Commit only our *code* and small derived text configs (and our own tiny proxy meshes).

## What we actually use from MIMIR-UW

MIMIR-UW Zenodo zips are **offline sensor datasets** (not packaged Unreal/AirSim worlds). A typical sequence contains:
- `cam*/data/*.png` (RGB)
- `depth*/data/*.exr` (depth)
- `depth*/sensor.yaml` (intrinsics + extrinsics `T_BS`)
- `pose_groundtruth/data.csv` (timestamped poses)

## How the stage is built

1) Choose one timestamp `t` and its aligned RGB + depth frame.
2) Use the sensor intrinsics to backproject depth pixels to 3D points.
3) Use `T_BS` + pose to transform points into a consistent frame.
4) Triangulate a mesh from the pixel grid (skipping depth discontinuities).
5) Write a textured `stage.obj` (with `preview_rgb.png` / `preview_depth.png`) and `scene_meta.json`.

Implementation: `tracks/h1_mimir_uw/scripts/build_stage_from_mimir_frame.py`.

## How to fetch upstream data (local-only)

We intentionally avoid committing upstream zips. Use HTTP-range probing/extraction:
- `tools/external_scenes/mimir_uw_zip_range_list.py`
- `tools/external_scenes/mimir_uw_zip_range_extract.py`

Then build the stage with `build_stage_from_mimir_frame.py`.

## What makes this track “ours”

- **Our dynamics**: current/drift is sampled from our generated `combined_environment.nc` (variants supported).
- **Our tasks**: plume source localization + multi-agent plume cleanup/containment (N=2–10).
- **Our eval**: metrics + manifests (`media_manifest.json`, `results_manifest.json`) for reproducible runs.

