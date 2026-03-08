# Benchmark Core

`benchmark_core` is the final quantitative benchmark used in the paper.
It provides:

- dataset-grounded current injection from `combined_environment.nc`,
- canonical task definitions,
- deterministic episode recording,
- constraint checks based on `land_mask` and bathymetry,
- sweep utilities for paper tables and ablations.

## Drift cache export

```bash
python -m benchmark_core.cli.export_drift_cache \
  --nc Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc \
  --u-var utotal \
  --v-var vtotal \
  --time-index 0 \
  --depth-index 0 \
  --out runs/benchmark_core/_cache/drift_scene_t0_d0.npz
```

## Single run

```bash
python -m benchmark_core.cli.run \
  --drift-npz runs/benchmark_core/_cache/drift_scene_t0_d0.npz \
  --task go_to_goal_current \
  --difficulty medium \
  --controller go_to_goal \
  --pollution-model gaussian \
  --n-agents 1 \
  --seed 0 \
  --dynamics-model 6dof \
  --constraint-mode hard \
  --bathy-mode hard \
  --validate
```

## Sweep run

```bash
python -m benchmark_core.cli.run_matrix \
  --drift-npz runs/benchmark_core/_cache/drift_scene_t0_d0.npz \
  --preset paper_v1 \
  --dynamics-model 6dof \
  --constraint-mode hard \
  --bathy-mode hard \
  --validate
```

## Outputs

Each run root under `runs/benchmark_core/` contains:

- `run_meta.json`
- `spec_snapshot.json`
- `metrics.json`
- `metrics.csv`
- `summary.csv`
- `results_manifest.json`
- `agents/agent_*/...` recorder streams

## Canonical tasks

- `go_to_goal_current`
- `station_keeping`
- `route_following_waypoints`
- `depth_profile_tracking`
- `formation_transit_multiagent`
- `fish_herding_8uuv`
- `pipeline_inspection_leak_detection`
- `area_scan_terrain_recon`
- `surface_pollution_cleanup_multiagent`
- `underwater_pollution_lift_5uuv`

## Notes

- The default paper setting is `--dynamics-model 6dof`.
- Constraint enforcement is an engineering realism gate, not a full hydrodynamics model.
- BC training utilities live in `benchmark_core/ml/`.
- Replay export for the web demo is available through `python -m benchmark_core.cli.export_demo_replay`.
