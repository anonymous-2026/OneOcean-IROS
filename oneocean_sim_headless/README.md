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
  --difficulty medium \
  --controller go_to_goal \
  --pollution-model gaussian \
  --n-agents 4 \
  --seed 0 \
  --constraint-mode hard \
  --bathy-mode off \
  --validate
```

Example: demo-family surface pollution cleanup (multi-agent; N=10) with a tiny top-down MP4:

```bash
python3 -m oneocean_sim_headless.cli.run \
  --drift-npz runs/headless/_cache/drift_scene_t0_d0.npz \
  --task surface_pollution_cleanup_multiagent \
  --difficulty medium \
  --controller go_to_goal \
  --pollution-model gaussian \
  --n-agents 10 \
  --seed 0 \
  --constraint-mode hard \
  --bathy-mode off \
  --validate \
  --render
```

## Outputs (recording layout)

Runs are written under:
- `runs/headless/<run_id>/`

Streams:
- `run_meta.json`, `metrics.json`, `metrics.csv`
- `results_manifest.json` (root index)
- `summary.csv` (row = episode; used for paper tables)
- `agents/agent_000/pose_groundtruth/data.csv` (t,x,y,z,qx,qy,qz,qw)
- `agents/agent_000/actions/data.csv`
- `agents/agent_000/obs/local_current/data.csv`
- `agents/agent_000/obs/pollution_probe/data.csv`
- `agents/agent_000/obs/latlon/data.csv`
- `agents/agent_000/obs/bathymetry/data.csv` (elevation + land_mask-at-agent, if present in cache)
- `environment_samples/global_time_index.csv`
- `environment_samples/semantics.jsonl` (optional; task semantics for replay export)

Validation:
- `python3 -c "from oneocean_sim_headless.validators import validate_run_dir; print(validate_run_dir('...'))"`

Replay + summarize:
```bash
python3 -m oneocean_sim_headless.cli.replay --run-dir runs/headless/<run_id>
```

## Canonical task ids (H1 implements the 10-task list)

Base:
- `go_to_goal_current`
- `station_keeping`

Demo must-have:
- `surface_pollution_cleanup_multiagent`
- `underwater_pollution_lift_5uuv` (requires `--n-agents 5`)
- `fish_herding_8uuv` (requires `--n-agents 8`)
- `area_scan_terrain_recon`
- `pipeline_inspection_leak_detection`

Core add-ons:
- `route_following_waypoints`
- `depth_profile_tracking`
- `formation_transit_multiagent`

## Constraints (bathymetry / land_mask)

This headless backend supports explicit **hard constraint** checks so tasks are not “just planar drift”:

- `--constraint-mode hard` (default): reject invalid regions using `land_mask` (agent stays put; increments `constraint_violations`).
- `--bathy-mode hard`: reject “touchdown / too-shallow” states using `elevation` vs agent depth `y` (positive down) with a clearance margin `--seafloor-clearance-m`.

Notes:
- These constraints are an engineering realism gate, not a full contact/hydrodynamics model.
- `--constraint-mode hard` requires `land_mask` in the drift cache `.npz`; `--bathy-mode hard` requires `elevation`.
- When hard constraints are enabled, `--validate` will also check recordings for invalid-region / touchdown samples.

## 2) Run an experiment matrix (and aggregate CSV)

```bash
python3 -m oneocean_sim_headless.cli.run_matrix \
  --drift-npz runs/headless/_cache/drift_scene_t0_d0.npz \
  --preset smoke \
  --constraint-mode hard \
  --bathy-mode off \
  --validate
```

Outputs:
- `runs/headless/matrix_<timestamp>/summary.csv`
- `runs/headless/matrix_<timestamp>/matrix_summary.json`
- `runs/headless/matrix_<timestamp>/results_manifest.json`

## 3) Export a run to demo-compatible JSON (env + multi-agent paths)

This does **not** modify `demo/`. It produces JSON files that can be imported by the demo UI.

```bash
python3 -m oneocean_sim_headless.cli.export_demo_replay \
  --run-dir runs/headless/<run_id>/episode_000 \
  --out-dir runs/headless/<run_id>/demo_export \
  --stride 4
```

Writes:
- `drone_map_data.json`
- `drone_path_data.json`
