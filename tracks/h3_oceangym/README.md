# H3 — OceanGym / HoloOcean

This track focuses on **high-quality underwater world visuals** (HoloOcean packaged worlds) and
then layering our **currents + pollution** tasks + evaluation on top.

Outputs are written under `runs/oceangym_h3/` and are ignored by git.

## 1) Render scene media (all Ocean worlds)

```bash
cd "oneocean(iros-2026-code)"
SSL_CERT_FILE=$(.venv_h3_oceangym/bin/python -c "import certifi; print(certifi.where())") \
  .venv_h3_oceangym/bin/python tracks/h3_oceangym/render_scene_media.py --preset ocean_worlds_camera
```

This exports, for each world:
- third-person orbit MP4 + keyframe PNG (`ViewportCapture`)
- vehicle-moving MP4 (`ViewportCapture`)
- first-person MP4 (`LeftCamera`)

## 2) Run the task suite (metrics + compact evidence)

```bash
cd "oneocean(iros-2026-code)"
SSL_CERT_FILE=$(.venv_h3_oceangym/bin/python -c "import certifi; print(certifi.where())") \
  .venv_h3_oceangym/bin/python tracks/h3_oceangym/run_task_suite.py --preset ocean_worlds_camera --episodes 3
```

This produces a top-level `results_manifest.json`, plus per-task `results_manifest.json` files and `metrics.json` summaries.

Difficulty presets:
- `--difficulty easy|medium|hard`

## 2.1) (Optional) Data-grounded currents from combined_environment.nc

H3’s runtime venv (`.venv_h3_oceangym`) does not include xarray/netCDF, so we export a small current series to NPZ
using a Python that has `xarray` (on this machine: the conda Python).

Export a `(time, depth)` current series at a chosen lat/lon (nearest):

```bash
cd "oneocean(iros-2026-code)"
/home/shuaijun/miniconda3/bin/python tracks/h3_oceangym/export_current_series_npz.py \
  --dataset /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/combined_environment.nc \
  --out_npz runs/_cache/data_grounding/currents/cmems_center_uovo.npz
```

Then run the suite with dataset-grounded forcing (uniform current sampled from NPZ):

```bash
cd "oneocean(iros-2026-code)"
SSL_CERT_FILE=$(.venv_h3_oceangym/bin/python -c "import certifi; print(certifi.where())") \
  .venv_h3_oceangym/bin/python tracks/h3_oceangym/run_task_suite.py \
    --scenarios PierHarbor-HoveringCamera \
    --episodes 1 \
    --current_npz runs/_cache/data_grounding/currents/cmems_center_uovo.npz
```

## 3) Postprocess: generate GIFs + per-video screenshots

The suite/render scripts export MP4s. For paper/demo convenience, we also generate:
- `*.gif` next to each `*.mp4`
- 3 keyframes per MP4: `*_keyframe_000.png`, `*_keyframe_<mid>.png`, `*_keyframe_<last>.png` (frame indices depend on video length)

This requires `imageio+Pillow`. On this machine, use the conda Python:

```bash
cd "oneocean(iros-2026-code)"
/home/shuaijun/miniconda3/bin/python tracks/h3_oceangym/postprocess_media.py \
  --roots runs/oceangym_h3/scene_media_20260226_043035 runs/oceangym_h3/task_suite_20260226_043900
```

It writes `postprocess_media_manifest.json` into each root for traceability.
