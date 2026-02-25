# Track H1 — MIMIR-UW (offline dataset → reconstructed 3D stage → underwater renders)

This track treats **MIMIR-UW Zenodo zips** as an **offline sensor dataset** (not packaged Unreal worlds) and builds a *reproducible* path to:
- reconstruct a textured 3D mesh patch from (RGB + depth + pose),
- render it as a Habitat-Sim stage,
- postprocess frames to look underwater (haze + color attenuation + particles),
- run our plume tasks (single- and multi-agent) grounded in our `combined_environment.nc`.

## Prereqs

- Mesh reconstruction (Python env with `opencv` + `trimesh` + `PIL`)
- Rendering + task execution (Habitat-Sim Python env)

## Step 0 — (Optional) selectively extract files from Zenodo zips

We support HTTP Range based probing/extraction (no multi-GB full download):
- `tools/external_scenes/mimir_uw_zip_range_list.py`
- `tools/external_scenes/mimir_uw_zip_range_extract.py`

Example (OceanFloor):
```bash
python3 tools/external_scenes/mimir_uw_zip_range_list.py \
  --url "https://zenodo.org/api/records/10406384/files/OceanFloor.zip/content" \
  --out-dir runs/external_scene_probes/mimir_uw/oceanfloor
```

## Step 1 — build a textured stage mesh from one (RGB, depth, pose) timestamp

```bash
python tracks/h1_mimir_uw/scripts/build_stage_from_mimir_frame.py \
  --rgb <path/to/rgb.png> \
  --depth-exr <path/to/depth.exr> \
  --sensor-yaml <path/to/depth/sensor.yaml> \
  --pose-csv <path/to/pose_groundtruth/data.csv> \
  --out-dir runs/h1_mimir_uw/stages/<name> \
  --stride 4
```

Outputs (in `--out-dir`):
- `stage.obj` (+ `material.mtl` + texture PNGs emitted by trimesh)
- `stage.object_config.json` (if you want to add as an object elsewhere)
- `scene_meta.json` (center/radius hints for render)
- `preview_rgb.png`, `preview_depth.png`

## Step 2 — render orbit video with underwater postprocess

```bash
python tracks/h1_mimir_uw/scripts/render_habitat_underwater_orbit.py \
  --scene runs/h1_mimir_uw/stages/<name>/stage.obj \
  --out-dir runs/h1_mimir_uw/renders/<name> \
  --frames 180 \
  --radius-mult 1.6
```

Outputs:
- `orbit_underwater.mp4`
- `keyframe_underwater.png`
- `render_manifest.json`

## Step 3 — export dataset-driven drift cache (from our NetCDF)

This extracts a single `(time_index, depth_index)` slice from our generated dataset into a compact `.npz` drift cache.

```bash
python tracks/h1_mimir_uw/scripts/export_drift_cache_from_nc.py \
  --nc OceanEnv/Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc \
  --u-var utotal \
  --v-var vtotal \
  --time-index 0 \
  --depth-index 0 \
  --out runs/h1_mimir_uw/drift/drift_scene_utotal_vtotal_t0_d0.npz
```

## Step 4 — run plume tasks (Habitat-Sim; multi-agent supported)

Required (minimum): ≥2 tasks, including ≥1 multi-agent run (N=2–10).

```bash
python tracks/h1_mimir_uw/run_habitat_plume_tasks.py \
  --scene runs/h1_mimir_uw/stages/<name>/stage.obj \
  --drift-npz runs/h1_mimir_uw/drift/drift_scene_utotal_vtotal_t0_d0.npz \
  --task localize_source \
  --n-agents 4 \
  --out-dir runs/h1_mimir_uw/tasks/localize_seed0
```

Hero multi-agent (recommended):
```bash
python tracks/h1_mimir_uw/run_habitat_plume_tasks.py \
  --scene runs/h1_mimir_uw/stages/<name>/stage.obj \
  --drift-npz runs/h1_mimir_uw/drift/drift_scene_utotal_vtotal_t0_d0.npz \
  --task cleanup_contain \
  --n-agents 10 \
  --out-dir runs/h1_mimir_uw/tasks/cleanup_n10_seed0
```

Outputs (per run directory):
- `rollout.mp4` + `keyframe.png`
- `metrics.json`
- `media_manifest.json` + `results_manifest.json`
- `scene_provenance.md` is at `tracks/h1_mimir_uw/scene_provenance.md`
