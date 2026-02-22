# OneOcean (IROS 2026) — Code

This repository contains the **ocean environment data pipeline** used to generate `combined_environment.nc` for simulation, experiments, and (eventually) public releases.

## What to run

Prerequisites:
- Set CMEMS credentials in your environment:
  - `export COPERNICUSMARINE_USERNAME=...`
  - `export COPERNICUSMARINE_PASSWORD=...`

### Canonical combined dataset (default)
```bash
python OceanEnv/Data_pipeline/run_pipeline.py --overwrite
```

### Multi-size datasets (tiny / scene / public)
```bash
python OceanEnv/Data_pipeline/generate_variants.py --which tiny,scene,public --overwrite
```

See `DATA_PIPELINE_LOG.md` for the full rationale, assumptions (tides), and reproducibility notes.

## S1 Navigation Task (MuJoCo primary track)

Run the first runnable simulation task (goal navigation with dataset-driven currents):

```bash
python -m oneocean_sim.cli.run_navigation_task \
  --task navigation \
  --variant scene \
  --episodes 3 \
  --seed 42
```

Key outputs are written under `runs/oneocean_nav_<timestamp>/`:
- `metrics.csv` (per-episode metrics)
- `metrics.json` (summary + metadata)
- `trajectory_overview.png` (trajectory with current field)
- `trajectories/episode_*.csv` (per-step trajectory traces)

Detailed CLI arguments and metrics schema: `oneocean_sim/README.md`.

Run a compact S1 experiment matrix:
```bash
python -m oneocean_sim.experiments.run_s1_matrix \
  --output-root runs/s1_matrix_v3 \
  --variants tiny,scene \
  --tasks navigation,station_keeping \
  --controller-modes compensated,naive \
  --tide-modes on,off \
  --episodes 2
```

Export paper/web-ready S1 report artifacts:
```bash
python -m oneocean_sim.experiments.export_s1_report \
  --matrix-root runs/s1_matrix_v3 \
  --output-dir runs/s1_reports/s1_matrix_v3
```

## S3 Navigation Task (SAPIEN backup track)

Run the S3 backup implementation with the locked SAPIEN interpreter:

```bash
/data/private/user2/workspace/embodied_platforms/maniskill2_env/.venv/bin/python \
  -m oneocean_sim_s3.cli.run_navigation_task_s3 \
  --task navigation \
  --variant tiny \
  --episodes 1 \
  --seed 42
```

Key outputs are written under `runs/oneocean_nav_s3_<timestamp>/`:
- `metrics.csv` (per-episode metrics)
- `metrics.json` (summary + metadata)
- `run_config.json` (dataset/backend/run config snapshot)
- `trajectories/episode_*.csv` (per-step trajectory traces)

Station-keeping (S3):

```bash
/data/private/user2/workspace/embodied_platforms/maniskill2_env/.venv/bin/python \
  -m oneocean_sim_s3.cli.run_navigation_task_s3 \
  --task station_keeping \
  --variant scene \
  --episodes 1
```

S3 matrix runner:

```bash
/data/private/user2/workspace/embodied_platforms/maniskill2_env/.venv/bin/python \
  -m oneocean_sim_s3.experiments.run_s3_matrix \
  --output-root runs/s3_matrix_v1 \
  --variants tiny,scene \
  --tasks navigation,station_keeping \
  --episodes 2
```

S3 result manifest + media:

```bash
/data/private/user2/workspace/embodied_platforms/maniskill2_env/.venv/bin/python \
  -m oneocean_sim_s3.experiments.build_s3_results_manifest \
  --matrix-roots runs/s3_matrix_v1,runs/s3_matrix_v1_notides \
  --output-json project/s3_results_manifest.json \
  --output-md project/s3_results_summary.md

/home/shuaijun/miniconda3/bin/python \
  -m oneocean_sim_s3.experiments.render_s3_media \
  --matrix-root runs/s3_matrix_v1 \
  --output-dir runs/s3_matrix_v1/media
```

## S2 Habitat Visual Track (parallel differentiated track)

Run the Habitat-Lab prototype with coarse drift injection (for qualitative visuals and demo material):

```bash
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.run_habitat_ocean_proxy \
  --episodes 1 \
  --max-steps 120
```

Use a compact capture preset for faster E2/web handoff:

```bash
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.run_habitat_ocean_proxy \
  --preset compact \
  --episodes 1 \
  --max-steps 120
```

Optional dataset-driven drift cache (instead of synthetic drift):

```bash
/home/shuaijun/miniconda3/bin/python -m oneocean_sim_habitat.cli.prepare_drift_cache \
  --dataset-path /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc \
  --output-path runs/s2_drift_cache_scene_t0_d0.npz \
  --time-index 0 --depth-index 0
```

```bash
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.run_habitat_ocean_proxy \
  --preset compact \
  --drift-cache-path runs/s2_drift_cache_scene_t0_d0.npz \
  --episodes 1 \
  --max-steps 120
```

Optional mask-based obstacle proxy (requires a cache generated with bathymetry, default behavior):

```bash
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.run_habitat_ocean_proxy \
  --preset compact \
  --drift-cache-path runs/s2_drift_cache_scene_t0_d0.npz \
  --obstacle-proxy-mode terminate \
  --episodes 1 \
  --max-steps 120
```

Key outputs are written under `runs/oneocean_habitat_s2_<timestamp>/`:
- `metrics.json` (episode summary)
- `run_config.json` (runtime and drift config)
- `trajectories/episode_*.csv` (position/action/drift traces)
- `screenshots/*.png`, `topdown/*.png`, `videos/episode_*.mp4`

Export S2 outputs for `demo_ref`-compatible front-end ingestion:

```bash
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.export_demo_assets \
  --run-dir runs/oneocean_habitat_s2_smoke \
  --episode 0
```

Build a compact bundle for demo integration:

```bash
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.build_compact_bundle \
  --run-dir runs/oneocean_habitat_s2_smoke
```

Publish S2 JSON directly into demo default data files:

```bash
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.publish_e2_demo_assets \
  --run-dir runs/oneocean_habitat_s2_smoke \
  --target-dir /data/private/user2/workspace/ocean/demo/assets/data
```

One-command run + export + bundle:

```bash
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.run_and_package \
  --preset compact \
  --build-media-package \
  --publish-e2 \
  --e2-target-dir /data/private/user2/workspace/ocean/demo/assets/data \
  --episodes 1 \
  --max-steps 60
```

Build media package from an existing S2 run:

```bash
PYTHONPATH=. /home/shuaijun/miniconda3/envs/habitat/bin/python \
  -m oneocean_sim_habitat.cli.build_media_package \
  --run-dir runs/oneocean_habitat_s2_smoke
```

Batch robustness regression (multi-case S2 run matrix):

```bash
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

## Websites (E1 lane)

Static website artifacts live in `docs/`:
- `docs/index.html`: web hub
- `docs/project/index.html`: project website (paper-facing)
- `docs/platform/index.html`: platform website (usage-facing)

Local preview:
```bash
cd docs
python -m http.server 8000
```

Then open:
- `http://localhost:8000/`
- `http://localhost:8000/project/`
- `http://localhost:8000/platform/`
