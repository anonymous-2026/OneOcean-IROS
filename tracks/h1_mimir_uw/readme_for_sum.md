# H1 (Headless / No-UI) — Summary of Deliverables + Selected Numerical Evidence

Date: 2026-03-05  
Scope: **numeric-only, table-ready** evidence + exact run roots to reproduce.

This file is written from the *project perspective*: H1 is the backend benchmark harness that turns
data-grounded tasks into auditable, paper-ready tables (and replayable recordings), without any UI/renderer.

---

## 0) Latest artifacts (as of 2026-03-05) — LATEST

These are the newest paper-facing outputs produced in this iteration (with clear deltas vs earlier artifacts):

- **LATEST (Main suite)**: `runs/headless/_tables_20260305/table_main_hard.md`
  - Delta vs older main tables: uses `paper_v1` knobs retuned to avoid saturated hard results + enforces bathy/land hard constraints by default.
- **LATEST (Robustness + tides)**: `runs/headless/_tables_20260305_v66disturb/table_disturbances_hard.md`
  - Delta vs `table_currentsweep_hard.md`: adds a **synthetic tidal disturbance** term on top of dataset currents (so robustness is not only “current_gain”).
- **LATEST (Planning-suite, stable seeds)**: `runs/headless/_tables_20260305_v65planning/table_planning_suite_medium.md`
  - Delta vs earlier planning triage tables (e.g. `_tables_20260305_v64c/`): more seeds, includes BC, and keeps `current_gain=2.0` to prevent all-100% SR.
- **LATEST (Planning-suite cost)**: `runs/headless/_tables_20260305_v67cost/table_planning_suite_cost_medium.md`
  - Delta vs older planning tables: adds **LLM cost metrics** (latency + token counts) and corresponding new `summary.csv` fields.
- **LATEST (Difficulty ladder)**: `runs/headless/_tables_20260305_v68ladder/table_difficulty_ladder.md`
  - Delta vs earlier reporting: makes the **easy→medium→hard** progression explicit (task×difficulty ladder, subset).

## 1) What H1 delivered

- Headless runners:
  - `python -m oneocean_sim_headless.cli.run` (single episode)
  - `python -m oneocean_sim_headless.cli.run_matrix` (sweeps)
  - `python -m oneocean_sim_headless.cli.run_matrix_farm` (parallel sharding)
- Demo replay export (bridge to the demo UI without touching `workspace/ocean/demo/`):
  - `python -m oneocean_sim_headless.cli.export_demo_replay --run-dir <H1_RUN_DIR> --out-dir <OUT_DIR>`
  - Output: `replay_bundle.json` (single-file import for `h1_demo_player/tasks.html`)
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
- Action space: desired **relative** velocity in world frame, `a_t∈R^3` (clipped to `max_speed_mps`).
- Energy proxy:
  ```tex
  E := \\sum_t \\lVert a_t \\rVert_2^2 \\; \\Delta t
  ```

### 3.1 Dynamics + drift (what “data-grounded” means in H1)

At every step we sample a local horizontal current from the cached dataset field:
```tex
c(x,z,t) := \\big[u_{data}(x,z)\\cdot g,\\;0,\\;v_{data}(x,z)\\cdot g\\big] + c_{tide}(t)
```
where `g = current_gain` is a stress-test knob. The default “official” setting is `g=1.0`.

State update is an engineering drift model (not full CFD):
```tex
p_{t+1} = p_t + (a_t + c(p_t,t))\\,\\Delta t
```
(`p=[x,y,z]`, with `y` positive down). For `--dynamics-model 6dof`, attitude/velocity are additionally integrated with bounded roll/pitch and a yaw update; however the **controller interface stays the same** (it always outputs `a_t`).

### 3.2 Hard constraints (land + bathymetry)

Constraints are enforced during the state update. If the proposed `p_{t+1}` violates constraints, the update is rejected (velocity is canceled for that step) and a counter increments.

Land constraint:
- Land: reject when `land_mask(x,z) ≥ land_mask_threshold`.

Bathymetry clearance (when `bathy_mode=hard`):
- Bathymetry clearance (when `bathy_mode=hard`):
  ```tex
  y + \\text{seafloor_clearance_m} > \\text{water_depth}(x,z),\\quad \\text{water_depth}(x,z) := -\\text{elevation}(x,z)
  ```
On rejection:
- `p_{t+1} := p_t`
- `constraint_violations += 1`

Implementation note (reproducibility guardrail):
- If `constraint_mode!=off` then the drift cache must include `land_mask`; if `bathy_mode=hard` it must include `elevation` (run fails fast otherwise).

### 3.3 Near-collision metric (multi-agent safety proxy)

H1 logs both `min_pairwise_dist_m` (episode min) and a step-based rate:
```tex
\\text{collision\\_rate} := \\frac{\\#\\{t : \\min_{i<j}\\|p^i_t - p^j_t\\|_2 \\le r\\}}{T}
```
where `r = collision_radius_m` is a reporting knob (it does **not** change dynamics).

### 3.4 Synthetic tide disturbance (robustness ablation)

To avoid depending on external tide downloads while still stress-testing policies with oscillatory flow, we add an optional synthetic term:
```tex
c_{tide}(t) := A\\,[\\sin(\\omega(t+\\phi)),\\;0,\\;\\cos(\\omega(t+\\phi))]
```
with amplitude `A=tide_amp_mps`, period `2π/ω = tide_period_s`, and phase `φ=tide_phase_s`.

This is a deliberate **engineering disturbance**: it is spatially uniform (same for all agents/positions) and does not claim to model realistic depth-dependent tides.

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
- `collision_rate` = fraction of steps with `min_pairwise_dist_m ≤ collision_radius_m` (multi-agent only)

---

## 5) Selected paper-ready numeric evidence

### 5.1 Baseline controller (paper_v1; 6DoF; medium+hard) — **current**

This supersedes older V15/V17 tables: task knobs were retuned to avoid saturated hard results and to better reflect multi-agent difficulty.

- Run root: `runs/headless/V24PAPER_paper_v1_6dof_heuristic_mh_bathyhard_20260305_065933_farm/`
- Summary CSV: `runs/headless/V24PAPER_paper_v1_6dof_heuristic_mh_bathyhard_20260305_065933_farm/summary.csv`
- Seeds/episodes: `seeds=0..9`, `episodes=2` (20 episodes per task×difficulty)

| task | diff | N | eps | SR | Tsucc_s (succ only) | E_mean | Viol_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| area_scan_terrain_recon | hard | 1 | 20 | 15.0% | 1619.7 | 2430.6 | 0.00 |
| area_scan_terrain_recon | medium | 1 | 20 | 90.0% | 1731.7 | 2532.2 | 0.00 |
| depth_profile_tracking | hard | 1 | 20 | 50.0% | 606.3 | 954.9 | 0.00 |
| depth_profile_tracking | medium | 1 | 20 | 85.0% | 379.9 | 586.0 | 0.00 |
| fish_herding_8uuv | hard | 8 | 20 | 0.0% |  | 6487.4 | 0.00 |
| fish_herding_8uuv | medium | 8 | 20 | 95.0% | 240.0 | 2441.1 | 0.00 |
| formation_transit_multiagent | hard | 10 | 20 | 20.0% | 379.5 | 6338.9 | 0.00 |
| formation_transit_multiagent | medium | 10 | 20 | 100.0% | 225.1 | 3241.4 | 0.00 |
| go_to_goal_current | hard | 1 | 20 | 75.0% | 193.5 | 317.0 | 0.00 |
| go_to_goal_current | medium | 1 | 20 | 95.0% | 159.2 | 238.0 | 0.00 |
| pipeline_inspection_leak_detection | hard | 1 | 20 | 30.0% | 781.7 | 1628.0 | 0.00 |
| pipeline_inspection_leak_detection | medium | 1 | 20 | 100.0% | 499.6 | 719.4 | 0.00 |
| route_following_waypoints | hard | 1 | 20 | 55.0% | 504.4 | 762.3 | 0.00 |
| route_following_waypoints | medium | 1 | 20 | 85.0% | 313.8 | 483.5 | 0.00 |
| station_keeping | hard | 1 | 20 | 15.0% | 300.7 | 339.6 | 0.00 |
| station_keeping | medium | 1 | 20 | 100.0% | 110.0 | 70.4 | 0.00 |
| surface_pollution_cleanup_multiagent | hard | 10 | 20 | 10.0% | 840.5 | 12314.3 | 0.00 |
| surface_pollution_cleanup_multiagent | medium | 10 | 20 | 85.0% | 388.9 | 6289.4 | 0.00 |
| underwater_pollution_lift_5uuv | hard | 5 | 20 | 20.0% | 496.8 | 1973.1 | 0.00 |
| underwater_pollution_lift_5uuv | medium | 5 | 20 | 20.0% | 439.2 | 1840.4 | 0.00 |

### 5.2 Option A (MLP BC) learned baseline (end-to-end; 6DoF; medium+hard) — **current**

Method summary (paper-facing):
- Teacher: the built-in heuristic controllers (per task) under the same physics/constraints as evaluation.
- Data: we record (i) a downsampled semantic stream containing `goal_for_action_xyz`, plus (ii) per-agent pose/actions/local_current/pollution_probe.
- Supervision target: the **relative velocity** action `a_t` that the teacher executed.
- Model: a tiny 2-layer MLP trained with MSE on standardized features/targets; exported as NumPy weights and run without torch during evaluation.

Dense demos (for BC dataset):
- `runs/headless/V26DEMOS_paper_v1_6dof_heuristic_rec5_20260305_070150_farm/`

BC dataset:
- `runs/headless/_models/bc_dataset_paper_v1_curfeat_20260305_073457/bc_dataset_v1.npz`
- `runs/headless/_models/bc_dataset_paper_v1_curfeat_20260305_073457/bc_dataset_v1_meta.json`
  - Feature schema (per agent):
    - `goal_delta = goal_for_action_xyz - pre_step_pose_xyz` (3)
    - `depth_y` (1)
    - `pollution_probe` (1)
    - `local_current_xz` (2)
    - `task_onehot` (10)
  - Total input dim: `7 + |task_vocab|`, output dim: `3` (`action_xyz`)

BC weights:
- `runs/headless/_models/bc_mlp_paper_v1_curfeat_20260305_073510/bc_mlp_v1_weights.npz`
  - Training script: `python -m oneocean_sim_headless.ml.train_bc_mlp_torch ...`
  - Architecture: `Linear(D→64)→ReLU→Linear(64→64)→ReLU→Linear(64→3)`
  - Inference: pure NumPy forward pass with stored `x_mean/x_std/y_mean/y_std`, then speed clipping to `max_speed_mps`

BC evaluation suite:
- Run root: `runs/headless/V34PAPER_paper_v1_6dof_mlp_bc_curfeat_mh_bathyhard_20260305_073530_farm/`
- Summary CSV: `runs/headless/V34PAPER_paper_v1_6dof_mlp_bc_curfeat_mh_bathyhard_20260305_073530_farm/summary.csv`

| task | diff | N | eps | SR | Tsucc_s (succ only) | E_mean | Viol_mean |
| --- | --- | --- | --- | --- | --- | --- | --- |
| area_scan_terrain_recon | hard | 1 | 20 | 5.0% | 1681.0 | 2037.5 | 0.00 |
| area_scan_terrain_recon | medium | 1 | 20 | 85.0% | 1797.4 | 1988.8 | 0.00 |
| depth_profile_tracking | hard | 1 | 20 | 0.0% |  | 260.5 | 0.00 |
| depth_profile_tracking | medium | 1 | 20 | 55.0% | 465.0 | 302.1 | 0.00 |
| fish_herding_8uuv | hard | 8 | 20 | 5.0% | 381.0 | 1644.5 | 0.00 |
| fish_herding_8uuv | medium | 8 | 20 | 100.0% | 258.1 | 544.6 | 0.00 |
| formation_transit_multiagent | hard | 10 | 20 | 0.0% |  | 4670.3 | 0.00 |
| formation_transit_multiagent | medium | 10 | 20 | 40.0% | 230.2 | 2963.9 | 0.00 |
| go_to_goal_current | hard | 1 | 20 | 35.0% | 207.6 | 214.3 | 0.00 |
| go_to_goal_current | medium | 1 | 20 | 45.0% | 160.2 | 140.6 | 0.00 |
| pipeline_inspection_leak_detection | hard | 1 | 20 | 10.0% | 1109.0 | 760.4 | 0.00 |
| pipeline_inspection_leak_detection | medium | 1 | 20 | 100.0% | 629.8 | 429.5 | 0.00 |
| route_following_waypoints | hard | 1 | 20 | 0.0% |  | 142.5 | 0.00 |
| route_following_waypoints | medium | 1 | 20 | 20.0% | 383.0 | 176.0 | 0.00 |
| station_keeping | hard | 1 | 20 | 0.0% |  | 10.7 | 0.00 |
| station_keeping | medium | 1 | 20 | 0.0% |  | 9.4 | 0.00 |
| surface_pollution_cleanup_multiagent | hard | 10 | 20 | 0.0% |  | 6136.1 | 0.00 |
| surface_pollution_cleanup_multiagent | medium | 10 | 20 | 45.0% | 507.1 | 6241.6 | 0.00 |
| underwater_pollution_lift_5uuv | hard | 5 | 20 | 5.0% | 146.0 | 75.0 | 0.00 |
| underwater_pollution_lift_5uuv | medium | 5 | 20 | 90.0% | 83.1 | 58.8 | 0.00 |

### 5.3 LLM high-level planner pilot (pipeline-only; hard; seeds 0–2; 14B + OLMo excluded)

Method summary (paper-facing):
- LLM is **not** used for low-level control. The low-level controller stays as a deterministic goal-following policy.
- The LLM only proposes **high-level discrete plans** at a fixed stride (every `llm_call_stride_steps`), and every output is schema-validated with deterministic fallbacks.
- Two high-level APIs are used:
  1) cleanup assignment: JSON `{ "assign": [s_i or -1] }` (one entry per agent; cannot assign already-done sources)
  2) waypoint assignment: JSON `{ "assign_wp": [k_i] }` (one entry per agent; each in `[0,K-1]`)
- Determinism + reproducibility:
  - generation uses `do_sample=False` and `temperature=0.0`
  - per-step payloads are hashed and cached to disk (`llm_cache_dir/<sha>.json`) so repeated runs reuse identical plans
  - invalid JSON / schema violations automatically fall back to a deterministic heuristic assignment

Pilot goal: demonstrate that the benchmark can differentiate an LLM high-level planner on a planning-sensitive multi-agent task.

- Run root: `runs/headless/V32PILOT_paper_v1_llm_pipelineOnly_hard_s0-2_20260305_072659/`
- Preset: `paper_v1_llm` (pipeline runs with `N=8`)
- Task: `pipeline_inspection_leak_detection` (hard)
- Seeds/episodes: seeds 0–2, episodes=1 (3 episodes per model)

Table: SR + mean leaks detected (final state) + time-to-success (successful eps only).

| model | eps | SR | leaks_detected_mean | Tsucc_s (succ only) |
| --- | --- | --- | --- | --- |
| heuristic | 3 | 0.0% | 5.67 |  |
| llm_chatglm3_6b | 3 | 0.0% | 5.67 |  |
| llm_glm4_9b | 3 | 0.0% | 1.33 |  |
| llm_llama2_7b | 3 | 0.0% | 1.33 |  |
| llm_llama3_8b | 3 | 0.0% | 3.33 |  |
| llm_mistral7b | 3 | 0.0% | 5.00 |  |
| llm_qwen2_7b | 3 | 0.0% | 6.00 |  |
| llm_qwen2p5_7b | 3 | 33.3% | 6.67 | 1448.0 |

### 5.4 LLM high-level planner (planning-suite; **medium**; current_gain=2.0; seeds 0–1; fast triage)

Goal: provide a **multi-task** LLM comparison table (cleanup + scan + pipeline) that does *not* saturate at all-0% or all-100%.

- Run root: `runs/headless/V64CLLM_planningTasks_medium_cg2_s0-1_ultrafast_20260305_191500/`
  - Heuristic baseline: `runs/headless/V64CLLM_planningTasks_medium_cg2_s0-1_ultrafast_20260305_191500/heuristic/summary.csv`
  - LLM models: `runs/headless/V64CLLM_planningTasks_medium_cg2_s0-1_ultrafast_20260305_191500/llm_*/summary.csv`
- Preset: `paper_v1_llm` (scan/pipeline multi-agent; single-agent tasks kept single-agent by preset)
- Tasks: `surface_pollution_cleanup_multiagent`, `area_scan_terrain_recon`, `pipeline_inspection_leak_detection`
- Difficulty: `medium` (with stronger currents via `--current-gain 2.0`)
- Seeds/episodes: `seeds=0..1`, `episodes=1`
- LLM budget: `--llm-call-stride-steps 200`, `--llm-max-new-tokens 96`
- Exported paper table: `runs/headless/_tables_20260305_v64c/table_planning_suite_medium.md`

Notes on skipped local models (environment dependency, not a benchmark decision):
- `GLM-4-9B-Chat` requires `tiktoken` (not available in this host environment).
- `OLMo-7B-Instruct` requires `hf_olmo` (not available in this host environment).

### 5.5 LLM high-level planner (planning-suite; **medium**; current_gain=2.0; seeds 0–9; includes BC)

Goal: a **stable, paper-facing** planning-suite table (more seeds) that differentiates:
heuristic vs learned BC vs multiple local LLM planners (high-level phase switching + waypoint assignment).

- Run root: `runs/headless/V65PLANNING_planningSuite_medium_cg2_s0-9_20260305_191000/`
  - Heuristic: `runs/headless/V65PLANNING_planningSuite_medium_cg2_s0-9_20260305_191000/heuristic/summary.csv`
  - BC: `runs/headless/V65PLANNING_planningSuite_medium_cg2_s0-9_20260305_191000/mlp_bc/summary.csv`
  - LLM models: `runs/headless/V65PLANNING_planningSuite_medium_cg2_s0-9_20260305_191000/llm_*/summary.csv`
- Preset: `paper_v1_llm`
- Tasks: `surface_pollution_cleanup_multiagent`, `area_scan_terrain_recon`, `pipeline_inspection_leak_detection`
- Difficulty: `medium` + stronger currents (`--current-gain 2.0`)
- Seeds/episodes: `seeds=0..9`, `episodes=1` (30 episodes / method)
- Exported paper table: `runs/headless/_tables_20260305_v65planning/table_planning_suite_medium.md`

### 5.6 LLM planner efficiency (planning-suite; **medium**; current_gain=2.0; seeds 0–2; latency + token usage)

Goal: support the “LLM planner comparison” table in `paper/.../docs/suggestion.md` with **cost metrics**:
planning latency and token usage (measured on uncached local inference calls).

- Run root: `runs/headless/V67LLMCOST_planningSuite_medium_cg2_s0-2_20260305_203000/`
- Exported paper table: `runs/headless/_tables_20260305_v67cost/table_planning_suite_cost_medium.md`
- New recorded fields in `summary.csv` (per-episode):
  - `llm_uncached_calls`, `llm_cached_calls`
  - `llm_latency_ms_total`, `llm_latency_ms_mean`
  - `llm_prompt_tokens_total`, `llm_output_tokens_total`

---

## 6) Cleanup guidance (post-verification)

Keep (paper-relevant):
- Baseline: `runs/headless/V24PAPER_paper_v1_6dof_heuristic_mh_bathyhard_20260305_065933_farm/`
- Baseline (metric-complete; for suggestion.md tables): `runs/headless/V61PAPER_paper_v1_6dof_heuristic_mh_bathyhard_metrics_20260305_091220/`
- Dense demos: `runs/headless/V26DEMOS_paper_v1_6dof_heuristic_rec5_20260305_070150_farm/`
- BC dataset: `runs/headless/_models/bc_dataset_paper_v1_curfeat_20260305_073457/`
- BC weights: `runs/headless/_models/bc_mlp_paper_v1_curfeat_20260305_073510/`
- BC eval: `runs/headless/V34PAPER_paper_v1_6dof_mlp_bc_curfeat_mh_bathyhard_20260305_073530_farm/`
- BC eval (metric-complete; for suggestion.md tables): `runs/headless/V62PAPER_paper_v1_6dof_mlp_bc_mh_bathyhard_metrics_20260305_091300/`
- LLM pilot: `runs/headless/V32PILOT_paper_v1_llm_pipelineOnly_hard_s0-2_20260305_072659/`
- Currentsweep (Table 2): `runs/headless/V40SWEEP_paper_v1_cg0_heuristic_hard_20260305_090000/`, `runs/headless/V41SWEEP_paper_v1_cg1_heuristic_hard_20260305_090050/`, `runs/headless/V42SWEEP_paper_v1_cg2_heuristic_hard_20260305_090120/`
- Currentsweep (BC): `runs/headless/V43SWEEP_paper_v1_cg0_mlpbc_hard_20260305_090220/`, `runs/headless/V44SWEEP_paper_v1_cg1_mlpbc_hard_20260305_090310/`, `runs/headless/V45SWEEP_paper_v1_cg2_mlpbc_hard_20260305_090400/`
- Scaling (Table 3; cleanup, medium): `runs/headless/V52SCALE_cg1_medium_N02_heuristic_20260305_090900/`, `runs/headless/V54SCALE_cg1_medium_N04_heuristic_20260305_090900/`, `runs/headless/V58SCALE_cg1_medium_N08_heuristic_20260305_090900/`, `runs/headless/V510SCALE_cg1_medium_N10_heuristic_20260305_090900/`
- Exported paper tables (Markdown; do not edit by hand): `runs/headless/_tables_20260305/`
- Planning-suite (medium; cg=2.0; stable seeds): `runs/headless/_tables_20260305_v65planning/table_planning_suite_medium.md`

Safe to delete after verification:
- calibration-only roots under `runs/headless/_calib_*`

---

## 7) suggestion.md table coverage (current state)

The following tables are now exportable from the run artifacts (no spreadsheets):

- Table 1 (main suite; hard) — **LATEST**: `runs/headless/_tables_20260305/table_main_hard.md`
  - per-task breakdown: `runs/headless/_tables_20260305/table_per_task_hard.md`
- Table 2 (robustness vs current strength; hard; SR averaged over canonical 10 tasks):
  - `runs/headless/_tables_20260305/table_currentsweep_hard.md`
- Table 2b (robustness with **tidal disturbance**; hard; SR averaged over canonical 10 tasks) — **LATEST**:
  - `runs/headless/_tables_20260305_v66disturb/table_disturbances_hard.md`
- Table 3 (multi-agent scaling; cleanup; medium; N=2/4/8/10):
  - `runs/headless/_tables_20260305/table_scaling_surface_pollution_cleanup_multiagent_medium.md`
- Table 4 (difficulty ladder; easy→medium→hard; task subset) — **LATEST**:
  - `runs/headless/_tables_20260305_v68ladder/table_difficulty_ladder.md`
- LLM planner comparison (planning-sensitive tasks only; hard; cg=1.0):
  - `runs/headless/_tables_20260305/table_planning_suite_hard.md`
  - (triage rerun, seeds 0–2; includes LLM instrumentation fields): `runs/headless/_tables_20260305_v63full/table_planning_suite_hard.md`
  - (fast triage, **medium**; current_gain=2.0): `runs/headless/_tables_20260305_v64c/table_planning_suite_medium.md`
  - (stable, **medium**; current_gain=2.0; includes BC) — **LATEST**: `runs/headless/_tables_20260305_v65planning/table_planning_suite_medium.md`
  - (LLM efficiency add-on; **medium**; latency + token usage) — **LATEST**: `runs/headless/_tables_20260305_v67cost/table_planning_suite_cost_medium.md`

Notes:
- Collision is currently a **near-collision** metric: `collision_rate := (#steps with min_pairwise_dist_m <= collision_radius_m) / steps`.
  Official tables above use `--collision-radius-m 2.0` (does not affect dynamics).
- `GLM-4-9B-Chat` currently falls back to heuristic because it requires `tiktoken` which is not available in the host environment (pip network unavailable).
