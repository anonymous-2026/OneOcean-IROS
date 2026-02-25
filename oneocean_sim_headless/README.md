# H1 — Headless Simulation + Recorder (no UI)

This package provides a **headless (no-UI)** simulation runner and a **MIMIR-inspired recorder layout** for OneOcean.

It is intended as a reproducible backend that other visual tracks can reuse (consistent tasks + metrics + provenance + recording).

## 0) Export drift cache from our dataset (combined_environment.nc)

Export a compact `(time_index, depth_index)` slice:

```bash
python3 -m oneocean_sim_headless.cli.export_drift_cache \
  --nc OceanEnv/Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc \
  --u-var utotal --v-var vtotal \
  --time-index 0 --depth-index 0 \
  --out runs/headless/_cache/drift_scene_t0_d0.npz
```

This writes:
- `runs/headless/_cache/drift_scene_t0_d0.npz`
- `runs/headless/_cache/drift_scene_t0_d0.json` (provenance: nc path + indices + variable names)

## 1) Run a headless episode (records CSV streams + metrics)

Example: go-to-goal under dataset current:

```bash
python3 -m oneocean_sim_headless.cli.run \
  --drift-npz runs/headless/_cache/drift_scene_t0_d0.npz \
  --task go_to_goal_current \
  --controller go_to_goal \
  --pollution-model gaussian \
  --n-agents 4 \
  --seed 0 \
  --validate
```

Example: multi-agent pollution containment (N=10):

```bash
python3 -m oneocean_sim_headless.cli.run \
  --drift-npz runs/headless/_cache/drift_scene_t0_d0.npz \
  --task pollution_containment_multiagent \
  --controller containment_ring \
  --pollution-model ocpnet_3d \
  --n-agents 10 \
  --seed 0 \
  --validate
```

## Outputs (recording layout)

Runs are written under:
- `runs/headless/<run_id>/`

Streams:
- `run_meta.json`, `metrics.json`, `metrics.csv`
- `agents/agent_000/pose_groundtruth/data.csv` (t,x,y,z,qx,qy,qz,qw)
- `agents/agent_000/actions/data.csv`
- `agents/agent_000/obs/local_current/data.csv`
- `agents/agent_000/obs/pollution_probe/data.csv`
- `environment_samples/global_time_index.csv`

Validation:
- `python3 -c "from oneocean_sim_headless.validators import validate_run_dir; print(validate_run_dir('...'))"`

Replay + summarize:
```bash
python3 -m oneocean_sim_headless.cli.replay --run-dir runs/headless/<run_id>
```
