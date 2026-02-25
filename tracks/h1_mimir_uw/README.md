# Track H1 — MIMIR-UW (offline dataset → reconstructed 3D stage → underwater renders)

This track treats **MIMIR-UW Zenodo zips** as an **offline sensor dataset** (not packaged Unreal worlds) and builds a *reproducible* path to:
- reconstruct a textured 3D mesh patch from (RGB + depth + pose),
- render it as a Habitat-Sim stage,
- postprocess frames to look underwater (haze + color attenuation + particles),
- then (next milestone) use this stage to host our multi-agent tasks with our dataset-driven currents/pollution.

## Prereqs

- Mesh reconstruction tools (Python):
  - `/data/private/user2/workspace/robosuite_learning/.venv/bin/python`
- Rendering tools (Habitat-Sim):
  - `/home/shuaijun/miniconda3/envs/habitat/bin/python`

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
/data/private/user2/workspace/robosuite_learning/.venv/bin/python \
  tracks/h1_mimir_uw/scripts/build_stage_from_mimir_frame.py \
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
/home/shuaijun/miniconda3/envs/habitat/bin/python \
  tracks/h1_mimir_uw/scripts/render_habitat_underwater_orbit.py \
  --scene runs/h1_mimir_uw/stages/<name>/stage.obj \
  --out-dir runs/h1_mimir_uw/renders/<name> \
  --frames 180 \
  --radius-mult 1.6
```

Outputs:
- `orbit_underwater.mp4`
- `keyframe_underwater.png`
- `render_manifest.json`

