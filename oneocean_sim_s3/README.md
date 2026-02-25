# OneOcean S3 Simulation (SAPIEN backup, A2)

This module implements the **S3 backup experiment track** under the updated quality gate:
- **3D underwater scene** (terrain mesh + obstacles + 3D vehicle pose),
- **recognizably underwater look** (haze/fog + suspended particles + textured seafloor),
- **≥2 tasks**, including **≥1 multi-agent** task,
- per-task **3D execution media** (GIF) + screenshots,
- **dataset-grounded** currents/terrain from `combined_environment.nc` variants.

Note: on this machine, SAPIEN Vulkan renderers are unavailable (no Vulkan device), so we render with a **CPU software renderer** while still using SAPIEN for scene/rigid-body stepping.

## Runtime

Use the locked S3 interpreter:

```bash
/data/private/user2/workspace/embodied_platforms/maniskill2_env/.venv/bin/python -c "import sapien; print(getattr(sapien,'__version__','?'))"
```

## Run a single task

Single-agent task (reef navigation with terrain-following + obstacles):

```bash
/data/private/user2/workspace/embodied_platforms/maniskill2_env/.venv/bin/python \
  -m oneocean_sim_s3.cli.run_navigation_task_s3 \
  --task reef_navigation \
  --variant scene \
  --external-scene polyhaven:dutch_ship_large_01 \
  --episodes 2 \
  --seed 120
```

Multi-agent task (formation navigation; 2 vehicles):

```bash
/data/private/user2/workspace/embodied_platforms/maniskill2_env/.venv/bin/python \
  -m oneocean_sim_s3.cli.run_navigation_task_s3 \
  --task formation_navigation \
  --variant scene \
  --external-scene polyhaven:dutch_ship_large_01 \
  --episodes 2 \
  --seed 220
```

Outputs go to `runs/s3_3d_<task>_<timestamp>/` and include:
- `metrics.csv`, `metrics.json`, `run_config.json`
- `trajectories/episode_*.csv`
- `media/scene.png`, `media/final.png`, `media/rollout.gif`, `media/media_manifest.json` (if rendering enabled)

## Run the A2 quality-gate suite (recommended)

This runs both tasks with tide on/off and generates compact media:

```bash
/data/private/user2/workspace/embodied_platforms/maniskill2_env/.venv/bin/python \
  -m oneocean_sim_s3.experiments.run_s3_quality_gate \
  --output-root runs/s3_3d_underwater_hero_v1 \
  --variants scene \
  --tasks reef_navigation,formation_navigation \
  --tide-modes on,off \
  --external-scene polyhaven:dutch_ship_large_01 \
  --episodes 2
```

Suite outputs:
- `runs/s3_3d_underwater_hero_v1/suite_manifest.json`
- `runs/s3_3d_underwater_hero_v1/suite_summary.md`

## Export canonical manifests (paper/web handoff)

```bash
/data/private/user2/workspace/embodied_platforms/maniskill2_env/.venv/bin/python \
  -m oneocean_sim_s3.experiments.build_s3_results_manifest \
  --suite-root runs/s3_3d_underwater_hero_v1 \
  --output-json project/results_manifest.json \
  --output-md project/results_summary.md \
  --output-csv project/results_summary.csv

/data/private/user2/workspace/embodied_platforms/maniskill2_env/.venv/bin/python \
  -m oneocean_sim_s3.experiments.build_s3_media_manifest \
  --suite-root runs/s3_3d_underwater_hero_v1 \
  --output-json project/media_manifest.json \
  --output-md project/media_summary.md
```

Notes:
- External scene assets are cached locally under `runs/_cache/` and are not committed to Git.
