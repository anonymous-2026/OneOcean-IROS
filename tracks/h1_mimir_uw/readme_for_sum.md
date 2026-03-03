# H1 (Headless / No-UI) — Summary of Deliverables + Selected Numerical Evidence

Date: 2026-03-03  
Status: **Summary mode (experiments stopped)**  
Scope: **numeric-only, table-ready** evidence for paper/web + how to reproduce.

This file is intended to be readable without following the day-to-day logs.

---

## 1) What H1 delivered

H1 provides a headless (no-UI) simulation backend that supports:

- **Headless runner**: `oneocean_sim_headless.cli.run` and `oneocean_sim_headless.cli.run_matrix` (+ `run_matrix_farm` for parallel sharding).
- **Deterministic recording** (MIMIR-inspired layout): per-agent streams under `agents/agent_XXX/` and run-level `run_meta.json`, `metrics.json`, `metrics.csv`, `summary.csv`.
- **Validators**: `oneocean_sim_headless.validators.validate_run_dir` (timestamps monotonic, row counts aligned, constraints respected when enabled).
- **Dynamics options** (DoF ablation + official mode):
  - `--dynamics-model kinematic` (legacy/pre-final; simple integrator)
  - `--dynamics-model 3dof` (debugging ablation)
  - `--dynamics-model 6dof` (**official**; used for FINAL suite)

Key policy/UI decision:
- This track is **no-UI** by design; it focuses on reproducible numeric artifacts that other visual tracks can consume.

---

## 2) Data grounding (currents) used by H1

All official runs below are grounded in our dataset via a drift-cache slice:

- Drift cache NPZ: `runs/headless/_cache/drift_scene_utotal_vtotal_t0_d0.npz`
- Drift cache provenance JSON: `runs/headless/_cache/drift_scene_utotal_vtotal_t0_d0.json`
  - Source NC: `/data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc`
  - Vars: `utotal`, `vtotal`
  - Indices: `time_index=0`, `depth_index=0`

Notes:
- The headless dynamics uses **relative-velocity convention** for currents (recorded in `spec_snapshot.json`).
- Tide is not included in the current official suite (optional future extension behind a flag).

---

## 2.5) Design + constraints (paper-facing; implemented + recorded)

This section captures the **device/constraint contract** that is actually implemented in H1 and recorded in each episode’s `spec_snapshot.json`.

### Coordinate / state convention

- World frame: `x=east`, `z=north`, `y=depth` (**positive down**).
- Pose per agent: position `p=[x,y,z]` plus attitude `(roll,pitch,yaw)` (stored as quaternion stream in `agents/*/pose_groundtruth/data.csv`).
- Currents: sampled from the drift cache as a world-frame horizontal velocity `c(p)=[u(x,z), 0, v(x,z)]`.

### Action interface + energy proxy

- Action space (recorded in `spec_snapshot.json`): `desired_relative_velocity_world_xyz`.
- Speed limit: action is clipped to `max_speed_mps` before integration.
- Energy proxy used in summaries:
  ```tex
  E := \\sum_t \\lVert a_t \\rVert_2^2 \\; \\Delta t
  ```

### Hard constraints (land/bathymetry)

Constraints are controlled by the two config flags (and recorded in `spec_snapshot.json`):
- `constraint_mode ∈ {off, hard}` (land-mask constraint)
- `bathy_mode ∈ {off, hard}` (seafloor clearance constraint)

The hard constraint check is:
- land-mask: reject if `land_mask(x,z) ≥ land_mask_threshold`
- bathymetry: reject if elevation is missing / non-finite, or if
  ```tex
  y + \\text{seafloor_clearance_m} > \\text{water_depth}(x,z), \\quad \\text{where } \\text{water_depth}(x,z) := -\\text{elevation}(x,z)
  ```
  (elevation is expected negative underwater; `elevation ≥ 0` is treated as invalid).

Hard-mode handling:
- if a step proposes an invalid next state, the position update is **rejected** (agent stays in place) and `constraint_violations += 1`.
- in dynamics modes, rejecting a step also zeros the **linear** relative body velocity (`nu[:3]=0`) as a stabilizing “stop” behavior.

Recorded violation metric:
- `constraint_violations` is a **count of rejected steps** (aggregated over all agents and timesteps), not seconds.

### 6DoF dynamics model (official runs)

Official runs use `dynamics_model=\"6dof\"` (see `dynamics_spec` in `spec_snapshot.json`).
Implementation is a **minimal diagonal** relative-velocity model (engineering-stable; paper-defensible) with a deterministic velocity-tracking controller:

```tex
\\nu := [u,v,w,p,q,r]^\\top \\quad (\\text{body-frame relative linear + angular velocity})
```

World integration (relative velocity + current):
```tex
\\dot{p} = R(\\eta)\\,\\nu_{lin} + c(p)
```

Diagonal “PID-lite” tracking in body frame:
```tex
\\tau_{lin} = K_p^{lin}(\\nu^{cmd}_{lin} - \\nu_{lin}),\\qquad
\\tau_{ang} = K_p^{ang}(\\nu^{cmd}_{ang} - \\nu_{ang}) - K_d^{ang}\\nu_{ang}
```
```tex
\\dot{\\nu}_{lin} = (\\tau_{lin} - D_{lin}\\nu_{lin}) / M_{lin},\\qquad
\\dot{\\nu}_{ang} = (\\tau_{ang} - D_{ang}\\nu_{ang}) / M_{ang}
```

All parameters (`M_*`, `D_*`, gains, angular-rate limits, angle convention) are recorded per episode in `spec_snapshot.json` under `dynamics_spec`.

---

## 3) Tasks covered (canonical 10-task list)

Canonical task ids (as required by the unified contract):

Base:
- `go_to_goal_current`
- `station_keeping`

Demo must-have:
- `surface_pollution_cleanup_multiagent` (default N=10; supports N=2–10)
- `underwater_pollution_lift_5uuv` (**requires N=5**)
- `fish_herding_8uuv` (**requires N=8**)
- `area_scan_terrain_recon` (paper-facing default N=8 for multi-agent coverage)
- `pipeline_inspection_leak_detection`

Core add-ons:
- `route_following_waypoints`
- `depth_profile_tracking`
- `formation_transit_multiagent` (supports N=2–10)

Difficulties used in the official suite:
- `medium`, `hard`

Multi-agent scaling policy:
- Code supports `N=2–10` (task-dependent; fixed-N tasks fail fast on mismatch).
- Official scaling sweep reported here evaluates `N in {2,4,8,10}` on formation.

---

## 4) Selected experiment inventory (numeric-only, table-ready)

### 4.1 Official FINAL suite (6DoF; use this for the main paper table)

- Run root: `runs/headless/FINAL_6dof_hero_full10_mh_20260303_farm/`
- Summary CSV: `runs/headless/FINAL_6dof_hero_full10_mh_20260303_farm/summary.csv`
- Per-episode artifacts (for audit/replay; path pattern):
  - `runs/headless/FINAL_6dof_hero_full10_mh_20260303_farm/shard_XX/<task>/<diff>/<pollution_model>/nN/seed_SSS/episode_EEE/`
  - Contains: `run_meta.json`, `metrics.json`, `metrics.csv`, `spec_snapshot.json`, `validation.json`, and recorded streams under `agents/`.
- Seeds/episodes:
  - `seeds=0..9`, `episodes=2` (total 20 episodes per task×difficulty group)
- Settings (fixed):
  - `--dynamics-model 6dof`
  - `--constraint-mode hard`
  - `--bathy-mode hard`
  - `--validate`
- Strong requirement satisfied:
  - every episode dir contains `spec_snapshot.json` with `dynamics_spec.dynamics_model="6dof"`

**Official FINAL suite table (per task × difficulty):**

| task | diff | N | eps | SR | Tsucc | E | Viol |
| --- | --- | --- | --- | --- | --- | --- | --- |
| area_scan_terrain_recon | hard | 8 | 20 | 5.0% | 758.0 | 19835.2 | 0.0 |
| area_scan_terrain_recon | medium | 8 | 20 | 100.0% | 195.0 | 2070.5 | 0.0 |
| depth_profile_tracking | hard | 8 | 20 | 100.0% | 311.3 | 3586.2 | 0.0 |
| depth_profile_tracking | medium | 8 | 20 | 100.0% | 199.6 | 2298.8 | 0.0 |
| fish_herding_8uuv | hard | 8 | 20 | 100.0% | 244.3 | 2275.6 | 0.0 |
| fish_herding_8uuv | medium | 8 | 20 | 100.0% | 224.6 | 2165.0 | 0.0 |
| formation_transit_multiagent | hard | 10 | 20 | 50.0% | 293.0 | 5484.3 | 0.0 |
| formation_transit_multiagent | medium | 10 | 20 | 95.0% | 203.7 | 3060.7 | 0.0 |
| go_to_goal_current | hard | 8 | 20 | 100.0% | 114.5 | 1319.6 | 0.0 |
| go_to_goal_current | medium | 8 | 20 | 100.0% | 97.3 | 1120.9 | 0.0 |
| pipeline_inspection_leak_detection | hard | 8 | 20 | 100.0% | 303.3 | 3494.0 | 0.0 |
| pipeline_inspection_leak_detection | medium | 8 | 20 | 100.0% | 261.1 | 3007.9 | 0.0 |
| route_following_waypoints | hard | 8 | 20 | 100.0% | 340.4 | 3920.8 | 0.0 |
| route_following_waypoints | medium | 8 | 20 | 100.0% | 192.2 | 2214.1 | 0.0 |
| station_keeping | hard | 8 | 20 | 100.0% | 151.6 | 1659.9 | 0.0 |
| station_keeping | medium | 8 | 20 | 100.0% | 129.2 | 1445.0 | 0.0 |
| surface_pollution_cleanup_multiagent | hard | 10 | 20 | 85.0% | 256.2 | 4258.5 | 0.0 |
| surface_pollution_cleanup_multiagent | medium | 10 | 20 | 100.0% | 168.3 | 2423.5 | 0.0 |
| underwater_pollution_lift_5uuv | hard | 5 | 20 | 75.0% | 29.9 | 608.8 | 0.0 |
| underwater_pollution_lift_5uuv | medium | 5 | 20 | 75.0% | 29.9 | 481.8 | 0.0 |

Legend:
- `SR`: success rate
- `Tsucc`: mean time-to-success (seconds; computed over successful episodes)
- `E`: mean energy proxy
- `Viol`: mean constraint violations (per-episode)

**Pollution-related metrics (selected tasks; from per-episode `metrics.json` → `final.*`):**

| task | diff | N | eps | SR | Tsucc | sources_done_frac | leaks_detected_mean | probe_mean_final |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| surface_pollution_cleanup_multiagent | medium | 10 | 20 | 100.0% | 168.3 | 1.00 |  | 0.020918 |
| surface_pollution_cleanup_multiagent | hard | 10 | 20 | 85.0% | 256.2 | 0.97 |  | 0.037470 |
| pipeline_inspection_leak_detection | medium | 8 | 20 | 100.0% | 261.1 |  | 3.00 | 0.004606 |
| pipeline_inspection_leak_detection | hard | 8 | 20 | 100.0% | 303.3 |  | 4.00 | 0.053023 |

Notes:
- `probe_mean_final` is the **final-step** pollution probe mean recorded in `metrics.json` (lower is better; units inherit from the pollution model configuration).
- `sources_done_frac` is `sources_done / sources_total` at the end of the episode (cleanup task).
- `leaks_detected_mean` is the mean number of detected leaks at episode end (pipeline task).

### 4.2 Official FINAL scaling sweep (6DoF; multi-agent scaling evidence)

- Task: `formation_transit_multiagent` (medium)
- Runs:
  - `runs/headless/FINAL_6dof_scaling_formation_medium_20260303_n02/summary.csv`
  - `runs/headless/FINAL_6dof_scaling_formation_medium_20260303_n04/summary.csv`
  - `runs/headless/FINAL_6dof_scaling_formation_medium_20260303_n08/summary.csv`
  - `runs/headless/FINAL_6dof_scaling_formation_medium_20260303_n10/summary.csv`

**Scaling table (medium):**

| task | diff | N | eps | SR | Tsucc | E | Viol |
| --- | --- | --- | --- | --- | --- | --- | --- |
| formation_transit_multiagent | medium | 2 | 10 | 80.0% | 214.6 | 697.0 | 0.0 |
| formation_transit_multiagent | medium | 4 | 10 | 100.0% | 221.7 | 1277.0 | 0.0 |
| formation_transit_multiagent | medium | 8 | 10 | 80.0% | 210.1 | 2673.5 | 0.0 |
| formation_transit_multiagent | medium | 10 | 10 | 90.0% | 201.9 | 3163.7 | 0.0 |

### 4.3 Dynamics ablations (kinematic + 3DoF; **not official**, but useful as an ablation table)

These runs use the same seeds/episodes/difficulties/constraints as the official FINAL suite, but different `--dynamics-model`.

- Kinematic: `runs/headless/ABL_dynamics_kinematic_full10_mh_20260303_farm/summary.csv`
- 3DoF: `runs/headless/ABL_dynamics_3dof_full10_mh_20260303_farm/summary.csv`

Hard-difficulty success rates across dynamics (selected summary):

| task | N | SR (kinematic) | SR (3DoF) | SR (6DoF / official) |
| --- | --- | --- | --- | --- |
| area_scan_terrain_recon | 8 | 10.0% | 10.0% | 5.0% |
| depth_profile_tracking | 8 | 100.0% | 100.0% | 100.0% |
| fish_herding_8uuv | 8 | 100.0% | 100.0% | 100.0% |
| formation_transit_multiagent | 10 | 55.0% | 50.0% | 50.0% |
| go_to_goal_current | 8 | 100.0% | 100.0% | 100.0% |
| pipeline_inspection_leak_detection | 8 | 100.0% | 100.0% | 100.0% |
| route_following_waypoints | 8 | 100.0% | 100.0% | 100.0% |
| station_keeping | 8 | 100.0% | 100.0% | 100.0% |
| surface_pollution_cleanup_multiagent | 10 | 95.0% | 85.0% | 85.0% |
| underwater_pollution_lift_5uuv | 5 | 100.0% | 75.0% | 75.0% |

### 4.4 Other H1 headless runs (pre-final / not selected for the official paper table)

These runs are preserved under `runs/headless/` for additional analysis, but they were executed **before** the official 6DoF requirement was enforced (most are `kinematic`). Do **not** mix them into the “official FINAL” table without re-running under `--dynamics-model 6dof`.

Controller / planner comparisons (kinematic-era evidence):
- BC baseline vs learned MLP (Option A):
  - `runs/headless/bc_teacher_nav5_mh_20260302/`
  - `runs/headless/bc_mlp_big_nav5_mh_eval_v2spec_20260302/`
  - `runs/headless/bc_teacher_demo26_mh_20260302/`
  - `runs/headless/bc_mlp_demo26_mh_eval_v2spec_20260302/`
- Stress tests (`current_gain=2.0`):
  - `runs/headless/baseline_demo26_mh_cg2_20260302/`
  - `runs/headless/bc_mlp_demo26_mh_eval_cg2_20260302/`
  - `runs/headless/deep_demo26_mh_cg2_20260302_baseline_farm/`
  - `runs/headless/deep_demo26_mh_cg2_20260303_bc_farm/`
- LLM planner pilots (Qwen2.5-7B):
  - cleanup: `runs/headless/llm_deep_cleanup_mh_qwen2p5_7b_20260302/`
  - scan+pipeline: `runs/headless/llm_scanpipe_mh_n8_qwen2p5_20260302/`
  - deep stress scan+pipeline: `runs/headless/deep_scanpipe_mh_n8_cg2_20260303_llm_qwen2p5_7b_farm/`

---

## 5) Repro commands (do not hardcode user-specific python paths)

Assuming you have the repo’s Python deps available (see `requirements.txt`) and you are at repo root `oneocean(iros-2026-code)/`:

### (A) Export drift cache from `combined_environment.nc`

```bash
python3 -m oneocean_sim_headless.cli.export_drift_cache \\
  --nc /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc \\
  --u-var utotal --v-var vtotal \\
  --time-index 0 --depth-index 0 \\
  --out runs/headless/_cache/drift_scene_utotal_vtotal_t0_d0.npz
```

### (B) Re-run the official FINAL suite (parallel farm)

```bash
python3 -m oneocean_sim_headless.cli.run_matrix_farm \\
  --out-dir runs/headless/FINAL_6dof_hero_full10_mh_REPRO \\
  --shards 16 --gpu-ids 0,1,2,3,4,5,6,7 --max-parallel 16 \\
  --poll-s 1.0 -- \\
  --drift-npz runs/headless/_cache/drift_scene_utotal_vtotal_t0_d0.npz \\
  --preset hero_full10 \\
  --dynamics-model 6dof \\
  --constraint-mode hard --bathy-mode hard \\
  --validate
```

### (C) Quick validator check for a single episode dir

```bash
python3 -c \"from oneocean_sim_headless.validators import validate_run_dir; print(validate_run_dir('runs/headless/FINAL_6dof_hero_full10_mh_REPRO/shard_00/go_to_goal_current/medium/gaussian/n8/seed_000/episode_000'))\"
```

---

## 6) Known limitations / caveats (avoid over-claiming)

- Dynamics model is **minimal/diagonal** (engineering-stable, paper-defensible) and does not model full hydrodynamic coupling, buoyancy/restoring forces, or contact physics.
- Policy action is a **high-level desired relative velocity**; the low-level mapping is a simple velocity-tracking controller (recorded in `spec_snapshot.json`).
- Constraint violations are **0** in the reported official suite. This does not indicate a bug, but it means the sampled tiles did not force collisions/touchdowns under these seeds/settings.
- No sensor noise/bias model is used in the official suite (observations are deterministic numerical streams).
- Some task families (cleanup/lift/fish/scan/pipeline) use simplified semantic dynamics; they are designed for controllable benchmarking, not photorealism.
