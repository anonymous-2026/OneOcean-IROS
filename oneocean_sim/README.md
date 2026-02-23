# OneOcean S1 Simulation (MuJoCo) — 3D Underwater Scenes + Tasks

This package is the **S1 primary (MuJoCo) simulator backend** for Lane A.

Quality gate note:
- Official S1 deliverables are **3D underwater scenes/tasks** with **3D media evidence** (PNG + MP4/GIF).
- Any 2D plotting-only outputs are **smoke-test scaffolding only**.

## Run (3D, quality-gate compliant)

Fetch underwater textures (local cache; do not commit):

```bash
cd oneocean(iros-2026-code)
/data/private/user2/workspace/robosuite_learning/.venv/bin/python \
  -m oneocean_sim.cli.fetch_underwater_assets
```

```bash
cd oneocean(iros-2026-code)
MUJOCO_GL=egl /data/private/user2/workspace/robosuite_learning/.venv/bin/python \
  -m oneocean_sim.cli.run_3d_task \
  --task nav_obstacles_3d \
  --controller compensated \
  --variant scene \
  --episodes 1 \
  --max-steps 900
```

Multi-agent task (2 vehicles):

```bash
cd oneocean(iros-2026-code)
MUJOCO_GL=egl /data/private/user2/workspace/robosuite_learning/.venv/bin/python \
  -m oneocean_sim.cli.run_3d_task \
  --task plume_source_localization_3d \
  --controller compensated \
  --variant scene \
  --episodes 1 \
  --max-steps 900
```

Optional dataset override:

```bash
MUJOCO_GL=egl /data/private/user2/workspace/robosuite_learning/.venv/bin/python \
  -m oneocean_sim.cli.run_3d_task \
  --dataset-path /abs/path/to/combined_environment.nc
```

## Tasks (3D)

- `nav_obstacles_3d`: 3D goal navigation under dataset currents with obstacles + bathymetry heightfield.
- `plume_source_localization_3d` (**multi-agent**): 2 vehicles cooperate via cast-and-surge to localize a source.
  - Plume concentration is a **coarse advection-diffusion proxy** driven by dataset currents (see `media_manifest.json` notes).

## CLI arguments (3D runner)

- `--variant {tiny,scene,public}` or `--dataset-path`
- `--task {nav_obstacles_3d,plume_source_localization_3d}`
- `--controller {auto,compensated,naive}`
- `--episodes`, `--seed`
- `--time-index`, `--depth-index`
- `--disable-tides`
- scene/sim:
  - `--dt-sec`, `--max-steps`, `--target-domain-size-m`, `--meters-per-sim-meter`
  - `--current-scale` (amplify dataset currents in sim for visibility/ablation)
- media:
  - `--no-media`, `--record-all-episodes`
  - `--render-width`, `--render-height`, `--fps`, `--camera` (`cam_main|cam_low|orbit`, default: `orbit`)

## Outputs (3D runner)

Written under `runs/oneocean_<task>_3d_<timestamp>/` (or `--output-dir`):
- `metrics.csv`, `metrics.json`
- `run_config.json`
- `trajectories/episode_*_agent*.csv`
- `media/episode_000_start.png`, `media/episode_000_end.png`, `media/episode_000.mp4`
- `media_manifest.json`

## Quality-gate experiment suite

Run a small suite (tasks × controller):

```bash
cd oneocean(iros-2026-code)
MUJOCO_GL=egl /data/private/user2/workspace/robosuite_learning/.venv/bin/python \
  -m oneocean_sim.experiments.run_s1_3d_quality_gate \
  --output-root runs/s1_3d_quality_v1 \
  --variant scene \
  --episodes 2
```

## Legacy (2D smoke only)

The older 2D runner (`oneocean_sim.cli.run_navigation_task` and scripts under `oneocean_sim/experiments/`)
is retained for debugging, but does not satisfy the 3D underwater quality gate.
