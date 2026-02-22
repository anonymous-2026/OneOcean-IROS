# OneOcean S2 Habitat Track

This package implements Agent H's differentiated S2 track:
- Habitat-Lab navigation prototype,
- coarse drift injection,
- screenshot/video export for website/demo material.

## Environment

Use the locked Habitat environment:

```bash
/home/shuaijun/miniconda3/envs/habitat/bin/python -c "import habitat, habitat_sim; print('ok')"
```

## Quick run

```bash
cd /data/private/user2/workspace/ocean/oneocean(iros-2026-code)
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.run_habitat_ocean_proxy \
  --episodes 1 --max-steps 120
```

Compact preset (recommended for E2 handoff):

```bash
cd /data/private/user2/workspace/ocean/oneocean(iros-2026-code)
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.run_habitat_ocean_proxy \
  --preset compact \
  --episodes 1 --max-steps 120
```

Use dataset-driven drift (optional):

1) Prepare cache from combined dataset (run with an interpreter that has `h5py`):

```bash
cd /data/private/user2/workspace/ocean/oneocean(iros-2026-code)
/home/shuaijun/miniconda3/bin/python -m oneocean_sim_habitat.cli.prepare_drift_cache \
  --dataset-path /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc \
  --output-path runs/s2_drift_cache_scene_t0_d0.npz \
  --time-index 0 --depth-index 0
```

By default this also stores `land_mask`/`elevation` in the cache for optional obstacle proxy. Disable with `--disable-bathymetry` if you only need currents.

2) Run Habitat with cache-backed drift:

```bash
cd /data/private/user2/workspace/ocean/oneocean(iros-2026-code)
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.run_habitat_ocean_proxy \
  --preset compact \
  --drift-cache-path runs/s2_drift_cache_scene_t0_d0.npz \
  --episodes 1 --max-steps 120
```

Enable mask-based obstacle proxy (terminates an episode after entering a blocked cell):

```bash
cd /data/private/user2/workspace/ocean/oneocean(iros-2026-code)
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.run_habitat_ocean_proxy \
  --preset compact \
  --drift-cache-path runs/s2_drift_cache_scene_t0_d0.npz \
  --obstacle-proxy-mode terminate \
  --episodes 1 --max-steps 120
```

## Outputs

Per run directory (under `runs/oneocean_habitat_s2_*`):
- `metrics.json`
- `run_config.json`
- `trajectories/episode_*.csv`
- `screenshots/*.png`
- `topdown/*.png`
- `videos/episode_*.mp4`

## Export to `demo_ref` schema (for E2)

```bash
cd /data/private/user2/workspace/ocean/oneocean(iros-2026-code)
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.export_demo_assets \
  --run-dir runs/oneocean_habitat_s2_smoke \
  --episode 0
```

Outputs:
- `demo_export/drone_map_data.json`
- `demo_export/drone_path_data.json`
- `demo_export/assets_manifest.json`

## Export S1-compatible metrics (for Lane C parsers)

```bash
cd /data/private/user2/workspace/ocean/oneocean(iros-2026-code)
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.export_s1_compatible_metrics \
  --run-dir runs/oneocean_habitat_s2_smoke \
  --episode 0
```

Outputs:
- `compat/metrics_s1_compat.json`
- `compat/metrics_s1_compat.csv`

## Build compact bundle

```bash
cd /data/private/user2/workspace/ocean/oneocean(iros-2026-code)
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.build_compact_bundle \
  --run-dir runs/oneocean_habitat_s2_smoke \
  --screenshot-count 4 \
  --topdown-count 6
```

Outputs:
- `compact_bundle/metadata/*`
- `compact_bundle/screenshots/*`
- `compact_bundle/topdown/*`
- `compact_bundle/videos/*`
- `compact_bundle/compact_bundle_manifest.json`

## Publish to demo assets directory (E2 direct handoff)

```bash
cd /data/private/user2/workspace/ocean/oneocean(iros-2026-code)
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.publish_e2_demo_assets \
  --run-dir runs/oneocean_habitat_s2_smoke \
  --target-dir /data/private/user2/workspace/ocean/demo/assets/data
```

Outputs in target dir:
- `ocean_map_data.json`
- `ocean_path_data.json`
- `oneocean_e2_sync_manifest.json`
- existing files are backed up as `*.bak.json` before overwrite

## One-command run and package

```bash
cd /data/private/user2/workspace/ocean/oneocean(iros-2026-code)
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.run_and_package \
  --preset compact \
  --build-media-package \
  --publish-e2 \
  --e2-target-dir /data/private/user2/workspace/ocean/demo/assets/data \
  --episodes 1 \
  --max-steps 60
```

Output:
- `run_and_package_manifest.json` in the run directory

## Build media package from an existing run

```bash
cd /data/private/user2/workspace/ocean/oneocean(iros-2026-code)
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.build_media_package \
  --run-dir runs/oneocean_habitat_s2_smoke
```

Media outputs:
- `media_package/scene/*` (multi-angle/context screenshots)
- `media_package/progress/*` (task progression frames)
- `media_package/video/*` (MP4 copy if present)
- `media_package/gifs/hero_rollout.gif`
- `media_package/scene_overview.png`
- `media_package/media_manifest.json`

## Batch regression (multi-case robustness)

```bash
cd /data/private/user2/workspace/ocean/oneocean(iros-2026-code)
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.run_batch_regression \
  --cases synthetic_compact,cache_compact,cache_obstacle \
  --drift-cache-path runs/s2_drift_cache_scene_t0_d0_bathy.npz \
  --episodes 1 \
  --max-steps 20 \
  --no-video \
  --bundle-no-video \
  --build-best-media-package \
  --publish-best-e2 \
  --e2-target-dir /data/private/user2/workspace/ocean/demo/assets/data
```

Batch outputs:
- `batch_manifest.json` (all case outputs + best-case selection)
- `batch_summary.csv` (case-level metrics summary)
- per-case run folders under batch root
