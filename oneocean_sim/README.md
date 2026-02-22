# OneOcean S1 Simulation (MuJoCo)

This module implements the S1 primary simulation track for Lane A.

## Run

```bash
python -m oneocean_sim.cli.run_navigation_task \
  --task navigation \
  --variant tiny \
  --episodes 1
```

Optional dataset override:

```bash
python -m oneocean_sim.cli.run_navigation_task --dataset-path /abs/path/to/combined_environment.nc
```

## CLI arguments (core)

- `--variant {tiny,scene,public}` or `--dataset-path`
- `--task {navigation,station_keeping}`
- `--controller {auto,goal_seek,goal_seek_naive,station_keep,station_keep_naive}`
- `--episodes`
- `--seed`
- `--time-index`
- `--depth-index`
- `--max-steps`
- `--goal-distance-m`
- `--goal-tolerance-m`
- `--station-success-radius-m`
- `--station-mean-radius-m`
- `--disable-tides`
- `--allow-invalid-region`
- `--output-dir`

## Compact experiment matrix

```bash
python -m oneocean_sim.experiments.run_s1_matrix \
  --output-root runs/s1_matrix_v3 \
  --variants tiny,scene \
  --tasks navigation,station_keeping \
  --controller-modes compensated,naive \
  --tide-modes on,off \
  --episodes 2
```

Generate paper/web-ready report artifacts from a matrix run:

```bash
python -m oneocean_sim.experiments.export_s1_report \
  --matrix-root runs/s1_matrix_v3 \
  --output-dir runs/s1_reports/s1_matrix_v3
```

Run robustness suite on official generated dataset:

```bash
python -m oneocean_sim.experiments.run_s1_robustness_suite \
  --output-root runs/s1_robustness_v1 \
  --variant scene \
  --tasks navigation,station_keeping \
  --controller-modes compensated,naive \
  --tide-modes on,off \
  --time-indices 0,-1 \
  --depth-indices 0 \
  --episodes 4
```

Render S1 media package (screenshots + GIF + MP4 + `media_manifest.json`):

```bash
python -m oneocean_sim.experiments.render_s1_media \
  --matrix-root runs/s1_matrix_v3 \
  --output-dir runs/s1_media/s1_matrix_v3 \
  --case-limit 8 \
  --max-frames 90 \
  --fps 10
```

## Outputs

- `metrics.csv`: per-episode metrics
- `metrics.json`: summary metrics + per-episode metrics
- `run_config.json`: run and dataset selection snapshot
- `trajectories/episode_*.csv`: per-step trajectories
- `trajectory_overview.png`: first-episode trajectory and current field
- report outputs (from `export_s1_report.py`):
  - `summary_table.md`
  - `ablation_summary.md`
  - `fig_success_rate_ablation.png/.pdf`
  - `fig_final_distance_ablation.png/.pdf`
  - `results_manifest.json`
- robustness outputs (from `run_s1_robustness_suite.py`):
  - `robustness_summary.md`
  - `robustness_manifest.json`
- media outputs (from `render_s1_media.py`):
  - `cases/*/scene_overview.png`
  - `cases/*/rollout.gif`
  - `cases/*/rollout.mp4`
  - `hero/rollout.gif`, `hero/rollout.mp4`
  - `media_manifest.json`

## Metrics schema (per episode)

- `success`
- `timeout`
- `invalid_terminated`
- `steps`
- `time_sec`
- `final_distance_to_goal_m`
- `path_length_m`
- `goal_distance_m`
- `displacement_m`
- `path_efficiency`
- `mean_cross_track_error_m`
- `mean_commanded_speed_mps`
- `energy_proxy`
- `invalid_steps`
- `episode_wall_clock_sec`
- `sim_steps_per_sec`
- `controller_compensates_current`
- `episode`
