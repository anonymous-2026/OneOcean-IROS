# H1 (Headless / No-UI) — Summary of Deliverables + Selected Numerical Evidence

Date: 2026-03-05  
Scope: **numeric-only, table-ready** evidence + exact run roots to reproduce.

This file is written from the *project perspective*: H1 is the backend benchmark harness that turns
data-grounded tasks into auditable, paper-ready tables (and replayable recordings), without any UI/renderer.

---

## 1) What H1 delivered

- Headless runners:
  - `python -m oneocean_sim_headless.cli.run` (single episode)
  - `python -m oneocean_sim_headless.cli.run_matrix` (sweeps)
  - `python -m oneocean_sim_headless.cli.run_matrix_farm` (parallel sharding)
- Stable run-dir contract (per episode):
  - `run_meta.json`, `spec_snapshot.json`, `metrics.json`/`metrics.csv`, per-agent streams under `agents/`
  - run-root aggregate: `summary.csv` (paper tables come from this, not spreadsheets)
- 10 canonical tasks implemented with `easy|medium|hard` configs; multi-agent tasks support `N=2–10` (task-dependent).
- Official dynamics mode for paper tables: `--dynamics-model 6dof`

---

## 2) Data grounding (what is “from our data”)

Official runs use a drift cache exported from our `combined_environment.nc`:
- Drift cache NPZ: `runs/headless/_cache/drift_scene_utotal_vtotal_t0_d0.npz`
- Drift cache provenance JSON: `runs/headless/_cache/drift_scene_utotal_vtotal_t0_d0.json`
  - Source NC: `/data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc`
  - Vars: `utotal`, `vtotal` (+ `land_mask`, `elevation` for constraints when enabled)
  - Indices: `time_index=0`, `depth_index=0`

Grounded in data (current suite):
- Currents sampled as `c(p)=[u(x,z),0,v(x,z)]` and added as drift.
- Hard constraints from `land_mask` + `elevation` when enabled (`constraint_mode`, `bathy_mode`).

Not grounded in data (current suite):
- Pollution is a controllable model used for observation/metrics (official runs use `pollution_model=gaussian`).

---

## 3) Design + constraints (paper-facing; implemented)

- World frame: `x=east`, `z=north`, `y=depth` (**positive down**).
- Action space: desired relative velocity in world frame (clipped to `max_speed_mps`).
- Energy proxy:
  ```tex
  E := \\sum_t \\lVert a_t \\rVert_2^2 \\; \\Delta t
  ```

Hard constraints:
- Land: reject when `land_mask(x,z) ≥ land_mask_threshold`.
- Bathymetry clearance (when `bathy_mode=hard`):
  ```tex
  y + \\text{seafloor_clearance_m} > \\text{water_depth}(x,z),\\quad \\text{water_depth}(x,z) := -\\text{elevation}(x,z)
  ```
- On rejection: position update is rejected and `constraint_violations += 1`.

---

## 4) Canonical 10-task coverage (H1)

Base:
- `go_to_goal_current`
- `station_keeping`

Demo must-have:
- `surface_pollution_cleanup_multiagent` (default N=10; supports N=2–10)
- `underwater_pollution_lift_5uuv` (**requires N=5**)
- `fish_herding_8uuv` (**requires N=8**)
- `area_scan_terrain_recon` (supports N=2–10)
- `pipeline_inspection_leak_detection`

Core add-ons:
- `route_following_waypoints`
- `depth_profile_tracking`
- `formation_transit_multiagent` (default N=10; supports N=2–10)

All paper tables below are from `summary.csv` and use:
- `--preset paper_v1`
- `--dynamics-model 6dof --constraint-mode hard --bathy-mode hard`
- `difficulties=medium,hard`

Table conventions:
- `SR` = mean of `success`
- `Tsucc_s` = mean `time_to_success_s` over **successful** episodes only (blank if SR=0%)
- `E_mean` = mean `energy_proxy` over all episodes
- `Viol_mean` = mean `constraint_violations` over all episodes

---

## 5) Selected paper-ready numeric evidence

### 5.1 Baseline controller (paper_v1; 6DoF; medium+hard)

- Run root: `runs/headless/V15PAPER_paper_v1_6dof_baseline_mh_bathyhard_20260305_060047_farm/`
- Summary CSV: `runs/headless/V15PAPER_paper_v1_6dof_baseline_mh_bathyhard_20260305_060047_farm/summary.csv`
- Seeds/episodes: `seeds=0..9`, `episodes=1` (10 episodes per task×difficulty)

| task | diff | N | eps | SR | Tsucc_s (succ only) | E_mean | Viol_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| area_scan_terrain_recon | hard | 1 | 10 | 40.0% | 2499.8 | 3643.1 | 0.0 |
| area_scan_terrain_recon | medium | 1 | 10 | 90.0% | 1702.7 | 2523.5 | 0.0 |
| depth_profile_tracking | hard | 1 | 10 | 100.0% | 634.6 | 913.8 | 0.0 |
| depth_profile_tracking | medium | 1 | 10 | 100.0% | 293.7 | 422.9 | 0.0 |
| fish_herding_8uuv | hard | 8 | 10 | 0.0% |  | 6673.3 | 0.0 |
| fish_herding_8uuv | medium | 8 | 10 | 100.0% | 190.7 | 1813.3 | 0.0 |
| formation_transit_multiagent | hard | 10 | 10 | 20.0% | 365.0 | 6263.8 | 0.0 |
| formation_transit_multiagent | medium | 10 | 10 | 100.0% | 218.5 | 3146.4 | 0.0 |
| go_to_goal_current | hard | 1 | 10 | 90.0% | 218.7 | 335.2 | 0.0 |
| go_to_goal_current | medium | 1 | 10 | 90.0% | 154.0 | 239.9 | 0.0 |
| pipeline_inspection_leak_detection | hard | 1 | 10 | 10.0% | 585.0 | 1677.0 | 0.0 |
| pipeline_inspection_leak_detection | medium | 1 | 10 | 100.0% | 549.2 | 790.8 | 0.0 |
| route_following_waypoints | hard | 1 | 10 | 90.0% | 490.6 | 727.9 | 0.0 |
| route_following_waypoints | medium | 1 | 10 | 100.0% | 251.8 | 362.6 | 0.0 |
| station_keeping | hard | 1 | 10 | 70.0% | 163.3 | 185.8 | 0.0 |
| station_keeping | medium | 1 | 10 | 100.0% | 90.0 | 55.7 | 0.0 |
| surface_pollution_cleanup_multiagent | hard | 10 | 10 | 30.0% | 365.7 | 5571.0 | 0.0 |
| surface_pollution_cleanup_multiagent | medium | 10 | 10 | 80.0% | 344.9 | 6016.2 | 0.0 |
| underwater_pollution_lift_5uuv | hard | 5 | 10 | 90.0% | 96.2 | 600.4 | 0.0 |
| underwater_pollution_lift_5uuv | medium | 5 | 10 | 90.0% | 96.2 | 530.2 | 0.0 |

### 5.2 Option A (MLP BC) learned baseline (end-to-end; 6DoF; medium+hard)

Dense demos (for BC dataset):
- `runs/headless/V16PAPER_paper_v1_6dof_demos_rec5_20260305_060125_farm/`

BC dataset:
- `runs/headless/_models/bc_dataset_paper_v1_20260305_060141/bc_dataset_v1.npz`
- `runs/headless/_models/bc_dataset_paper_v1_20260305_060141/bc_dataset_v1_meta.json`

BC weights:
- `runs/headless/_models/bc_mlp_paper_v1_20260305_060151/bc_mlp_v1_weights.npz`

BC evaluation suite:
- Run root: `runs/headless/V17PAPER_paper_v1_6dof_mlp_bc_mh_20260305_060234_farm/`
- Summary CSV: `runs/headless/V17PAPER_paper_v1_6dof_mlp_bc_mh_20260305_060234_farm/summary.csv`

| task | diff | N | eps | SR | Tsucc_s (succ only) | E_mean | Viol_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| area_scan_terrain_recon | hard | 1 | 10 | 0.0% |  | 1501.1 | 0.0 |
| area_scan_terrain_recon | medium | 1 | 10 | 60.0% | 1922.2 | 1421.5 | 0.0 |
| depth_profile_tracking | hard | 1 | 10 | 0.0% |  | 71.8 | 0.0 |
| depth_profile_tracking | medium | 1 | 10 | 0.0% |  | 48.6 | 0.0 |
| fish_herding_8uuv | hard | 8 | 10 | 10.0% | 542.0 | 1452.5 | 0.0 |
| fish_herding_8uuv | medium | 8 | 10 | 100.0% | 221.2 | 196.6 | 0.0 |
| formation_transit_multiagent | hard | 10 | 10 | 0.0% |  | 3847.3 | 0.0 |
| formation_transit_multiagent | medium | 10 | 10 | 40.0% | 217.8 | 2288.1 | 0.0 |
| go_to_goal_current | hard | 1 | 10 | 20.0% | 243.5 | 221.8 | 0.0 |
| go_to_goal_current | medium | 1 | 10 | 20.0% | 199.5 | 130.0 | 0.0 |
| pipeline_inspection_leak_detection | hard | 1 | 10 | 0.0% |  | 304.0 | 0.0 |
| pipeline_inspection_leak_detection | medium | 1 | 10 | 40.0% | 973.0 | 291.7 | 0.0 |
| route_following_waypoints | hard | 1 | 10 | 0.0% |  | 37.1 | 0.0 |
| route_following_waypoints | medium | 1 | 10 | 0.0% |  | 26.8 | 0.0 |
| station_keeping | hard | 1 | 10 | 0.0% |  | 6.0 | 0.0 |
| station_keeping | medium | 1 | 10 | 0.0% |  | 4.6 | 0.0 |
| surface_pollution_cleanup_multiagent | hard | 10 | 10 | 0.0% |  | 1304.6 | 0.0 |
| surface_pollution_cleanup_multiagent | medium | 10 | 10 | 0.0% |  | 1025.2 | 0.0 |
| underwater_pollution_lift_5uuv | hard | 5 | 10 | 100.0% | 27.0 | 58.1 | 0.0 |
| underwater_pollution_lift_5uuv | medium | 5 | 10 | 100.0% | 27.0 | 58.1 | 0.0 |

### 5.3 LLM high-level planner pilot (pool triage; seeds 0–1; 14B excluded)

This is a pilot to ensure the “LLM-as-high-level planner” wiring is runnable on local open-source models.
LLM calls are **schema-validated + cached**, and failures fall back to deterministic non-LLM assignments
so runs still finish and produce `summary.csv`.

Run roots (each contains `summary.csv`):
- `runs/headless/V10PAPER_paper_v1_llm_pool_s0-1_20260305_051331/` (llama3/mistral/qwen2/qwen2.5)
- `runs/headless/V12PAPER_paper_v1_llm_pending_s0-1_20260305_053838/` (chatglm3/olmo)
- `runs/headless/V14PAPER_paper_v1_llm_glm4_llama2_strideHuge_s0-1_20260305_054913/` (glm4/llama2)

Hard-SR snapshot (per model; seeds 0–1; **not** a paper main table, just triage signal):

| model | cleanup_hard | formation_hard | fish_hard | scan_hard | pipeline_hard |
| --- | --- | --- | --- | --- | --- |
| llama3_8b | 50% | 50% | 0% | 0% | 0% |
| mistral7b | 50% | 50% | 0% | 0% | 0% |
| qwen2_7b | 0% | 50% | 0% | 0% | 0% |
| qwen2p5_7b | 0% | 50% | 0% | 0% | 0% |
| chatglm3_6b | 50% | 50% | 0% | 0% | 0% |
| olmo7b | 50% | 50% | 0% | 0% | 0% |
| glm4_9b | 50% | 50% | 0% | 0% | 0% |
| llama2_7b | 50% | 50% | 0% | 0% | 0% |

---

## 6) Cleanup guidance (post-verification)

Keep (paper-relevant):
- `runs/headless/V15PAPER_paper_v1_6dof_baseline_mh_bathyhard_20260305_060047_farm/`
- `runs/headless/V16PAPER_paper_v1_6dof_demos_rec5_20260305_060125_farm/`
- `runs/headless/_models/bc_dataset_paper_v1_20260305_060141/`
- `runs/headless/_models/bc_mlp_paper_v1_20260305_060151/`
- `runs/headless/V17PAPER_paper_v1_6dof_mlp_bc_mh_20260305_060234_farm/`
- LLM triage roots: `V10PAPER_*`, `V12PAPER_*`, `V14PAPER_*`

Safe to delete after verification:
- calibration-only roots under `runs/headless/_calib_*`
