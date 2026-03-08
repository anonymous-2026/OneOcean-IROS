# OceanGym Benchmark

This directory contains the final scene-grounded benchmark used in the paper.
It builds on HoloOcean packaged underwater worlds and layers in:

- data-grounded currents exported from the combined environment dataset,
- pollution localization and containment tasks,
- multi-agent scaling runs,
- media generation for screenshots, GIFs, and MP4s.

Outputs are written to `runs/oceangym_benchmark/` and are ignored by git.

## External dependency

The benchmark requires a working HoloOcean client plus the Ocean world package.
Repository-local code here assumes that runtime environment already exists.

The HoloOcean/OceanGym-facing integration points are mainly:

- `run_task_suite.py`: benchmark execution entrypoint.
- `holoocean_patch.py`: scenario patching and agent insertion.
- `scenarios.py`: scene presets and camera/task configuration.
- `export_current_series_npz.py`: bridge from combined dataset to simulator currents.
- `ocpnet_plume.py`: pollution/plume coupling used by the scene-grounded tasks.

## Render scene media

```bash
python tracks/oceangym_benchmark/render_scene_media.py \
  --preset ocean_worlds_camera
```

## Export data-grounded current series

```bash
python tracks/oceangym_benchmark/export_current_series_npz.py \
  --dataset Data_pipeline/Data/Combined/combined_environment.nc \
  --out_npz runs/_cache/data_grounding/currents/cmems_center_uovo.npz
```

The exported NPZ stores:

- `time_ns`
- `depth_m`
- `uo`
- `vo`
- `latitude`
- `longitude`

## Run the task suite

```bash
python tracks/oceangym_benchmark/run_task_suite.py \
  --preset ocean_worlds_camera \
  --episodes 3 \
  --current_npz runs/_cache/data_grounding/currents/cmems_center_uovo.npz
```

## Postprocess media

```bash
python tracks/oceangym_benchmark/postprocess_media.py \
  --roots runs/oceangym_benchmark/<run_root>
```

## One-command bundle

```bash
bash tracks/oceangym_benchmark/run_hero_bundle.sh
```

## Main scripts

- `render_baseline_scene.py`: minimal scene sanity check.
- `render_scene_media.py`: multi-scene media capture.
- `run_task_suite.py`: main benchmark runner.
- `run_scaling_sweep.py`: multi-agent scaling evaluation.
- `postprocess_media.py`: GIF and keyframe generation.
- `export_current_series_npz.py`: dataset to current-series export.
- `scene_provenance.md`: scene source and licensing notes.
