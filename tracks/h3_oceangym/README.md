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

This produces a top-level `results_manifest.json`, plus per-task `results_manifest.json` files.

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
