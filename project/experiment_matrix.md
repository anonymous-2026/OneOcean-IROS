# Experiment Matrix (S1 / MuJoCo Primary)

This matrix is owned by Agent A-S1 and is executable without waiting for other lanes.

## Claims

1. **C1 (Task executability):** The S1 task runner can execute deterministic episodes on both `tiny` and `scene` variants and export machine-readable metrics.
2. **C2 (Control feasibility):** Baseline controllers can solve:
   - goal navigation (within configured tolerance),
   - station keeping (bounded drift around target station).
3. **C3 (Reproducibility):** The same command+seed produces repeatable run artifacts (`metrics.csv/json`, config snapshot, trajectory traces).

## Matrix

| Experiment ID | Claim(s) | Purpose | Dataset/Variant | Task | Controller | Metrics | Seeds | Status | Run IDs |
|---|---|---|---|---|---|---|---:|---|---|
| S1-E1 | C1,C2 | Smoke test task runner | `tiny` | `navigation` | `goal_seek` | success, final_distance, path_efficiency | 1 | done | `oneocean_nav_task_smoke` |
| S1-E2 | C1,C2 | Smoke test station keeping | `tiny` | `station_keeping` | `station_keep` | success, final_distance, station_mean_radius | 1 | done | `oneocean_station_task_smoke2` |
| S1-E3 | C1,C2,C3 | Compact matrix v1 | `tiny,scene` | both tasks | auto | success_rate by case | 1 per case | done | `s1_matrix_v1` |
| S1-E4 | C1,C2,C3 | Compact matrix v2 (step budget tuned) | `tiny,scene` | both tasks | auto | success_rate by case | 2 per case | done | `s1_matrix_v2` |
| S1-E5 | C1,C2,C3 | Controller+tide ablation matrix | `tiny,scene` | both tasks | compensated vs naive | success/final_distance/energy/sim_speed | 2 per case | in-progress | `s1_matrix_v3` |
| S1-E6 | C1,C2,C3 | Robustness suite (time/speed/goal stress) | `scene` | both tasks | compensated vs naive | aggregated success/final_distance/energy under stress factors | 4 per stress case | in-progress | `s1_robustness_v1` |
| S1-M1 | C2 (qualitative) | Media package from S1 runs | `scene` (from matrix cases) | navigation + station_keeping | compensated/naive | screenshots + GIF + MP4 + media manifest | n/a | in-progress | `s1_media/s1_matrix_v3` |

## Execution commands

Single task run:

```bash
python -m oneocean_sim.cli.run_navigation_task --task navigation --variant tiny --episodes 1 --seed 11
python -m oneocean_sim.cli.run_navigation_task --task station_keeping --variant tiny --episodes 1 --seed 11 --max-steps 300
```

Matrix run (current default):

```bash
python -m oneocean_sim.experiments.run_s1_matrix \
  --output-root runs/s1_matrix_v3 \
  --variants tiny,scene \
  --tasks navigation,station_keeping \
  --controller-modes compensated,naive \
  --tide-modes on,off \
  --episodes 2 \
  --seed-base 300
```

Report export:

```bash
python -m oneocean_sim.experiments.export_s1_report \
  --matrix-root runs/s1_matrix_v3 \
  --output-dir runs/s1_reports/s1_matrix_v3
```

Robustness suite:

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

Media package:

```bash
python -m oneocean_sim.experiments.render_s1_media \
  --matrix-root runs/s1_matrix_v3 \
  --output-dir runs/s1_media/s1_matrix_v3 \
  --case-limit 8 \
  --max-frames 90 \
  --fps 10
```

## Reporting notes

- `run_s1_matrix.py` default `max_steps=600` is intentional. `400` caused near-miss navigation runs (not controller failure).
- S1-E5 is the first matrix that directly supports reviewer-facing comparison:
  - current-compensated controller vs naive controller,
  - tides on vs tides off.
- S1-E6 extends to robustness dimensions (time index, speed cap, and navigation distance stress) using official generated data only.
- S1-M1 is the required qualitative package for paper/website/demo handoff (`media_manifest.json` + screenshots/GIF/MP4).
