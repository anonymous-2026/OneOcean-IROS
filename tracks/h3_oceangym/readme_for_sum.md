# H3 (OceanGym / HoloOcean 2.0.1) — Summary of Deliverables + Selected Evidence

Date: 2026-03-03  
Status: **Summary mode (experiments stopped)**  
Scope: H3 OceanGym integration + data-grounded runs + curated media pointers for paper/web.

> This file is intended to be **committable** (text-only). It references **local run outputs** under
> `oneocean(iros-2026-code)/runs/oceangym_h3/` which are not meant to be committed.

## 0) Quick evidence (underwater sanity check)

These are the fastest “open and verify it’s truly underwater” pointers.

- SCREENSHOT (PNG):
  - `/data/private/user2/workspace/ocean/oneocean(iros-2026-code)/runs/oceangym_h3/FINAL_scene_media_pierharbor_20260303_062633/PierHarbor-HoveringCamera/orbit_keyframe.png`
- VIDEO (MP4):
  - `/data/private/user2/workspace/ocean/oneocean(iros-2026-code)/runs/oceangym_h3/FINAL_scene_media_pierharbor_20260303_062633/PierHarbor-HoveringCamera/orbit_viewport.mp4`
  - `/data/private/user2/workspace/ocean/oneocean(iros-2026-code)/runs/oceangym_h3/FINAL_scene_media_pierharbor_20260303_062633/PierHarbor-HoveringCamera/move_viewport.mp4`

---

## 1) What H3 delivered

H3 provides a structured “benchmark harness” on top of HoloOcean 2.0.1 Ocean worlds:

- **Scenario patching for recording + multi-agent**:
  - ensures consistent camera capture (viewport + FPV),
  - supports `N=2–10` by cloning the base HoveringAUV agent config,
  - forces `PoseSensor/VelocitySensor` to the correct frame for control (see §7).
- **Task ladder (v1)** implemented in `tracks/h3_oceangym/run_task_suite.py`:
  - `go_to_goal_current`
  - `station_keeping`
  - `route_following_waypoints`
  - `depth_profile_tracking`
  - `formation_transit_multiagent`
  - pollution family (canonical id `surface_pollution_cleanup_multiagent`) via:
    - localization variant (`task_variant=localization`)
    - containment/cleanup variant (`task_variant=containment`)
- **Data grounding hook**:
  - currents are driven by an exported NPZ time series (from `combined_environment.nc` → `export_current_series_npz.py`)
  - optional “dataset clock” via `--dataset_days_per_sim_second`.
- **Artifacts (per requirements)**:
  - Root: `results_manifest.json`, `metrics.json`, `summary.csv`, `media_manifest.json`
  - Per task: `metrics.json`, `results_manifest.json`, `media_manifest.json`
  - Per episode: `epXXX/metrics.json`
  - Media: MP4s for `ViewportCapture` (third-person) + `LeftCamera` (FPV) plus postprocessed GIF + keyframes.

---

## 2) Data grounding used by H3

Official runs below use:

- Currents NPZ: `oneocean(iros-2026-code)/runs/_cache/data_grounding/currents/cmems_center_uovo.npz`
  - Produced from our combined dataset (see `tracks/h3_oceangym/export_current_series_npz.py`).
  - NPZ schema (verified on this machine): `time_ns` (T), `depth_m` (Z), `uo`/`vo` (T×Z), and scalar `latitude`/`longitude`.
  - Example metadata in the bundled NPZ:
    - `latitude=32.50209205020921`, `longitude=-65.9979079497908`
    - `depth_m` has 26 levels (surface→~186 m)
    - `source_dataset=/data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/combined_environment.nc`
- Dataset clock: `--dataset_days_per_sim_second 0.1`
  - advances the current index as simulation time runs.

Notes:
- H3 currently applies currents as a **drift/teleport-style hook** (engineering approximation), not full hydrodynamics.
- Tide is **not** included in H3 official runs (see §7).

## 2.1 Scene provenance / licensing notes

See `oneocean(iros-2026-code)/tracks/h3_oceangym/scene_provenance.md` for:
- which packaged HoloOcean worlds we used,
- what external zips exist in the local cache,
- and redistribution / EULA constraints.

---

## 3) Tasks covered + difficulties + multi-agent scaling

### Tasks actually executed in the FINALFIX runs

- `go_to_goal_current` (single-agent)
- `station_keeping` (single-agent)
- `route_following_waypoints` (single-agent)
- `depth_profile_tracking` (single-agent)
- `formation_transit_multiagent` (multi-agent)
- `surface_pollution_cleanup_multiagent` (multi-agent) with:
  - `task_variant=localization`
  - `task_variant=containment`

### Difficulty presets used

- Official FINALFIX runs: `easy` only.

### Multi-agent scaling

- Harness supports `N=2–10`.
- Evaluated values:
  - hero: `N=8` for the multi-agent tasks
  - scaling sweep: `N in {2,4,8,10}` for `formation_transit_multiagent`

---

## 4) Official experiment inventory (table-ready)

All paths below are relative to the **ocean workspace root** (`/data/private/user2/workspace/ocean`).

### 4.1 Official FINALFIX hero suite (use this as the H3 “main table” evidence)

Run root:
- `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/`

Key artifacts:
- `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/summary.csv`
- `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/results_manifest.json`
- `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/metrics.json`
- `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/media_manifest.json`
- Postprocess index (GIF + keyframes):
  - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/postprocess_media_manifest.json`

Settings (from the run manifest):
- scenarios: `PierHarbor-HoveringCamera`, `OpenWater-HoveringCamera`
- episodes: `10`
- `N=8` (multi-agent tasks)
- pollution model: `ocpnet_3d`
- plume steps: localization `400` (20s), containment `200` (10s)
- currents: `runs/_cache/data_grounding/currents/cmems_center_uovo.npz`, `dataset_days_per_sim_second=0.1`

Storage footprint:
- `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/` ≈ **627 MB**
- media counts: **28 MP4**, **28 GIF**, **84 keyframe PNG**

**Metric snapshot (grouped; from `summary.csv`):**

| scenario | task_id | task_variant | N | episodes | SR | mean time_s (succ-only) |
|---|---|---|---:|---:|---:|---:|
| PierHarbor-HoveringCamera | go_to_goal_current |  | 1 | 10 | 100.0% | 4.4 |
| PierHarbor-HoveringCamera | station_keeping |  | 1 | 10 | 100.0% | 8.0 |
| PierHarbor-HoveringCamera | route_following_waypoints |  | 1 | 10 | 100.0% | 11.4 |
| PierHarbor-HoveringCamera | depth_profile_tracking |  | 1 | 10 | 100.0% | 8.0 |
| PierHarbor-HoveringCamera | formation_transit_multiagent |  | 8 | 10 | 0.0% |  |
| PierHarbor-HoveringCamera | surface_pollution_cleanup_multiagent | localization | 8 | 10 | 90.0% | 20.0 |
| PierHarbor-HoveringCamera | surface_pollution_cleanup_multiagent | containment | 8 | 10 | 100.0% | 10.0 |
| OpenWater-HoveringCamera | go_to_goal_current |  | 1 | 10 | 100.0% | 4.4 |
| OpenWater-HoveringCamera | station_keeping |  | 1 | 10 | 100.0% | 8.0 |
| OpenWater-HoveringCamera | route_following_waypoints |  | 1 | 10 | 100.0% | 11.4 |
| OpenWater-HoveringCamera | depth_profile_tracking |  | 1 | 10 | 100.0% | 8.0 |
| OpenWater-HoveringCamera | formation_transit_multiagent |  | 8 | 10 | 0.0% |  |
| OpenWater-HoveringCamera | surface_pollution_cleanup_multiagent | localization | 8 | 10 | 100.0% | 20.0 |
| OpenWater-HoveringCamera | surface_pollution_cleanup_multiagent | containment | 8 | 10 | 100.0% | 10.0 |

Paper-table exports (generated from the hero suite `summary.csv`):
- `project/results/h3_FINALFIX_hero_easy_ep10_N8_20260303_103437__paper_table.md`
- `project/results/h3_FINALFIX_hero_easy_ep10_N8_20260303_103437__paper_table.csv`
- `project/results/h3_FINALFIX_hero_easy_ep10_N8_20260303_103437__paper_table.tex`

### 4.1b Minimal “scene-only” media bundle (fastest underwater check)

Run root:
- `oneocean(iros-2026-code)/runs/oceangym_h3/FINAL_scene_media_pierharbor_20260303_062633/`

Key artifacts:
- `oneocean(iros-2026-code)/runs/oceangym_h3/FINAL_scene_media_pierharbor_20260303_062633/PierHarbor-HoveringCamera/orbit_keyframe.png`
- `oneocean(iros-2026-code)/runs/oceangym_h3/FINAL_scene_media_pierharbor_20260303_062633/PierHarbor-HoveringCamera/orbit_viewport.mp4`
- `oneocean(iros-2026-code)/runs/oceangym_h3/FINAL_scene_media_pierharbor_20260303_062633/PierHarbor-HoveringCamera/move_viewport.mp4`

### 4.2 Official FINALFIX scaling sweep (formation; N in {2,4,8,10})

Run root:
- `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_scaling_formation_easy_ep10_20260303_114936/`

Key artifacts:
- `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_scaling_formation_easy_ep10_20260303_114936/summary.csv`
- `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_scaling_formation_easy_ep10_20260303_114936/results_manifest.json`
- children:
  - `.../n02/summary.csv`
  - `.../n04/summary.csv`
  - `.../n08/summary.csv`
  - `.../n10/summary.csv`

Settings:
- task: `formation_transit_multiagent`
- difficulty: `easy`
- episodes per N: `10`
- quant-only: `--no_media`

**Scaling snapshot (from sweep `summary.csv`):**

| N | episodes | SR | mean time_s (succ-only) | mean rms_formation_error_m |
|---:|---:|---:|---:|---:|
| 2 | 10 | 30.0% | 6.5 | 8.12 |
| 4 | 10 | 0.0% |  | 6.91 |
| 8 | 10 | 0.0% |  | 6.47 |
| 10 | 10 | 0.0% |  | 6.43 |

Paper-table exports:
- `project/results/h3_FINALFIX_scaling_formation_easy_ep10_20260303_114936__paper_table.md`
- `project/results/h3_FINALFIX_scaling_formation_easy_ep10_20260303_114936__paper_table.csv`
- `project/results/h3_FINALFIX_scaling_formation_easy_ep10_20260303_114936__paper_table.tex`

### 4.3 Lean alternative final suite (PierHarbor-only; smaller/faster)

This suite is useful when you want a smaller “single-world” table/manifest, but still with all key tasks.

Run root:
- `oneocean(iros-2026-code)/runs/oceangym_h3/FINAL_suite_multidiff_pierharbor_N8_ocpnet_20260303_062633/`

Key artifacts:
- Easy (full coverage):
  - `oneocean(iros-2026-code)/runs/oceangym_h3/FINAL_suite_multidiff_pierharbor_N8_ocpnet_20260303_062633/easy/summary.csv`
  - `oneocean(iros-2026-code)/runs/oceangym_h3/FINAL_suite_multidiff_pierharbor_N8_ocpnet_20260303_062633/easy/results_manifest.json`
- Medium/hard partial (best-effort; see IPC caveat in §7):
  - `oneocean(iros-2026-code)/runs/oceangym_h3/FINAL_suite_multidiff_pierharbor_N8_ocpnet_20260303_062633/summary.csv`

---

## 5) Curated media index (use these first)

All media below come from:
`oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/`

### Paper-selected screenshots (confirmed)

These two screenshots are selected for the paper (user-confirmed as **[Image #1]** and **[Image #2]**):

- [Image #1] (viewport keyframe; pollution containment/cleanup context):
  - Experiment context:
    - run: `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/`
    - scenario: `PierHarbor-HoveringCamera`
    - task_id: `surface_pollution_cleanup_multiagent`, task_variant: `containment`
    - difficulty: `easy`, multi-agent `N=8`, episode: `ep000`
    - episode metrics: `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/surface_pollution_cleanup_multiagent__containment/ep000/metrics.json`
  - `/data/private/user2/workspace/ocean/oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/surface_pollution_cleanup_multiagent__containment/ep000/contain_viewport_keyframe_000.png`
- [Image #2] (viewport keyframe; multi-agent formation context):
  - Experiment context:
    - run: `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/`
    - scenario: `PierHarbor-HoveringCamera`
    - task_id: `formation_transit_multiagent`
    - difficulty: `easy`, multi-agent `N=8`, episode: `ep000`
    - episode metrics: `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/formation_transit_multiagent/ep000/metrics.json`
  - `/data/private/user2/workspace/ocean/oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/formation_transit_multiagent/ep000/formation_viewport_keyframe_000.png`

### PierHarbor (ep000; easy; N=8 where applicable)

- Navigation (third-person GIF + keyframe):
  - Context: run=`h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437`, scenario=`PierHarbor-HoveringCamera`, task=`go_to_goal_current`, diff=`easy`, N=`1`, episode=`ep000`
  - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/go_to_goal_current/ep000/nav_viewport.gif`
  - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/go_to_goal_current/ep000/nav_viewport_keyframe_040.png`
  - MP4 (same view):
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/go_to_goal_current/ep000/nav_viewport.mp4`
  - FPV (LeftCamera; GIF + keyframe):
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/go_to_goal_current/ep000/nav_leftcamera.gif`
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/go_to_goal_current/ep000/nav_leftcamera_keyframe_040.png`
- Pollution localization (third-person GIF + keyframe):
  - Context: run=`h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437`, scenario=`PierHarbor-HoveringCamera`, task=`surface_pollution_cleanup_multiagent`, variant=`localization`, diff=`easy`, N=`8`, episode=`ep000`
  - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/surface_pollution_cleanup_multiagent__localization/ep000/plume_viewport.gif`
  - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/surface_pollution_cleanup_multiagent__localization/ep000/plume_viewport_keyframe_200.png`
  - MP4 (same view):
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/surface_pollution_cleanup_multiagent__localization/ep000/plume_viewport.mp4`
  - FPV (LeftCamera; GIF + keyframe):
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/surface_pollution_cleanup_multiagent__localization/ep000/plume_leftcamera.gif`
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/surface_pollution_cleanup_multiagent__localization/ep000/plume_leftcamera_keyframe_200.png`
- Pollution containment/cleanup (third-person GIF + keyframe):
  - Context: run=`h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437`, scenario=`PierHarbor-HoveringCamera`, task=`surface_pollution_cleanup_multiagent`, variant=`containment`, diff=`easy`, N=`8`, episode=`ep000`
  - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/surface_pollution_cleanup_multiagent__containment/ep000/contain_viewport.gif`
  - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/surface_pollution_cleanup_multiagent__containment/ep000/contain_viewport_keyframe_100.png`
  - MP4 (same view):
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/surface_pollution_cleanup_multiagent__containment/ep000/contain_viewport.mp4`
  - FPV (LeftCamera; GIF + keyframe):
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/surface_pollution_cleanup_multiagent__containment/ep000/contain_leftcamera.gif`
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/surface_pollution_cleanup_multiagent__containment/ep000/contain_leftcamera_keyframe_100.png`
- Formation transit (dynamic; third-person GIF + FPV GIF):
  - Context: run=`h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437`, scenario=`PierHarbor-HoveringCamera`, task=`formation_transit_multiagent`, diff=`easy`, N=`8`, episode=`ep000`
  - Third-person:
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/formation_transit_multiagent/ep000/formation_viewport.gif`
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/formation_transit_multiagent/ep000/formation_viewport.mp4`
  - FPV:
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/PierHarbor-HoveringCamera/formation_transit_multiagent/ep000/formation_leftcamera.gif`

### OpenWater (ep000; easy; N=8 where applicable)

- Navigation (third-person GIF + keyframe):
  - Context: run=`h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437`, scenario=`OpenWater-HoveringCamera`, task=`go_to_goal_current`, diff=`easy`, N=`1`, episode=`ep000`
  - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/go_to_goal_current/ep000/nav_viewport.gif`
  - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/go_to_goal_current/ep000/nav_viewport_keyframe_043.png`
  - MP4 (same view):
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/go_to_goal_current/ep000/nav_viewport.mp4`
  - FPV (LeftCamera; GIF + keyframe):
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/go_to_goal_current/ep000/nav_leftcamera.gif`
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/go_to_goal_current/ep000/nav_leftcamera_keyframe_043.png`

- Pollution localization (third-person GIF + keyframe):
  - Context: run=`h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437`, scenario=`OpenWater-HoveringCamera`, task=`surface_pollution_cleanup_multiagent`, variant=`localization`, diff=`easy`, N=`8`, episode=`ep000`
  - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/surface_pollution_cleanup_multiagent__localization/ep000/plume_viewport.gif`
  - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/surface_pollution_cleanup_multiagent__localization/ep000/plume_viewport_keyframe_200.png`
  - FPV (LeftCamera; GIF + keyframe):
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/surface_pollution_cleanup_multiagent__localization/ep000/plume_leftcamera.gif`
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/surface_pollution_cleanup_multiagent__localization/ep000/plume_leftcamera_keyframe_200.png`
- Pollution containment/cleanup (third-person GIF + keyframe):
  - Context: run=`h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437`, scenario=`OpenWater-HoveringCamera`, task=`surface_pollution_cleanup_multiagent`, variant=`containment`, diff=`easy`, N=`8`, episode=`ep000`
  - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/surface_pollution_cleanup_multiagent__containment/ep000/contain_viewport.gif`
  - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/surface_pollution_cleanup_multiagent__containment/ep000/contain_viewport_keyframe_100.png`
  - FPV (LeftCamera; GIF + keyframe):
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/surface_pollution_cleanup_multiagent__containment/ep000/contain_leftcamera.gif`
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/surface_pollution_cleanup_multiagent__containment/ep000/contain_leftcamera_keyframe_100.png`

- Station keeping (third-person GIF + FPV GIF):
  - Context: run=`h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437`, scenario=`OpenWater-HoveringCamera`, task=`station_keeping`, diff=`easy`, N=`1`, episode=`ep000`
  - Third-person:
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/station_keeping/ep000/station_viewport.gif`
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/station_keeping/ep000/station_viewport_keyframe_080.png`
  - FPV:
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/station_keeping/ep000/station_leftcamera.gif`
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/station_keeping/ep000/station_leftcamera_keyframe_080.png`
- Route following (third-person GIF + FPV GIF):
  - Context: run=`h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437`, scenario=`OpenWater-HoveringCamera`, task=`route_following_waypoints`, diff=`easy`, N=`1`, episode=`ep000`
  - Third-person:
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/route_following_waypoints/ep000/route_viewport.gif`
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/route_following_waypoints/ep000/route_viewport_keyframe_119.png`
  - FPV:
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/route_following_waypoints/ep000/route_leftcamera.gif`
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/route_following_waypoints/ep000/route_leftcamera_keyframe_119.png`
- Depth profile tracking (third-person GIF + FPV GIF):
  - Context: run=`h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437`, scenario=`OpenWater-HoveringCamera`, task=`depth_profile_tracking`, diff=`easy`, N=`1`, episode=`ep000`
  - Third-person:
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/depth_profile_tracking/ep000/depth_viewport.gif`
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/depth_profile_tracking/ep000/depth_viewport_keyframe_080.png`
  - FPV:
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/depth_profile_tracking/ep000/depth_leftcamera.gif`
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/depth_profile_tracking/ep000/depth_leftcamera_keyframe_080.png`
- Formation transit (dynamic; third-person GIF + FPV GIF):
  - Context: run=`h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437`, scenario=`OpenWater-HoveringCamera`, task=`formation_transit_multiagent`, diff=`easy`, N=`8`, episode=`ep000`
  - Third-person:
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/formation_transit_multiagent/ep000/formation_viewport.gif`
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/formation_transit_multiagent/ep000/formation_viewport.mp4`
  - FPV:
    - `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/OpenWater-HoveringCamera/formation_transit_multiagent/ep000/formation_leftcamera.gif`

Notes:
- For every selected item, there is also a matching FPV (`*_leftcamera.*`) in the same folder.
- Full media inventory for the run is indexed by:
  - `.../media_manifest.json`
  - `.../postprocess_media_manifest.json`

---

## 6) Repro commands (minimal)

All commands assume you are at the repo root: `oneocean(iros-2026-code)/`.

### 6.0 Repro capsule (versions + hashes; for supplement sanity)

- Code:
  - repo: `oneocean(iros-2026-code)/`
  - git commit (local): `eea30a92a30c3304238c0868d8e66f7504f51e2a`
- Runtime (H3 venv):
  - python: `3.9.25`
  - `holoocean==2.0.1`
  - `numpy==2.0.2`
- Data grounding artifact used by official H3 runs:
  - file: `oneocean(iros-2026-code)/runs/_cache/data_grounding/currents/cmems_center_uovo.npz`
  - size: `7.4K`
  - sha256: `1fe8ca53b2d42ed547af4b4b4819474724c032b07429d2cd1e80586e65482d7f`
- Storage (run dir sizes on this machine):
  - hero suite: `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_hero_easy_ep10_N8_20260303_103437/` ≈ `627M`
  - scaling sweep: `oneocean(iros-2026-code)/runs/oceangym_h3/h3_FINALFIX_COMSOCKET_scaling_formation_easy_ep10_20260303_114936/` ≈ `572K`
  - PierHarbor-only suite: `oneocean(iros-2026-code)/runs/oceangym_h3/FINAL_suite_multidiff_pierharbor_N8_ocpnet_20260303_062633/` ≈ `203M`
  - scene-only bundle: `oneocean(iros-2026-code)/runs/oceangym_h3/FINAL_scene_media_pierharbor_20260303_062633/` ≈ `103M`

### Re-run a small smoke (1 task, 1 episode)

```bash
rm -f /dev/shm/HOLODECK_MEM* /dev/shm/sem.HOLODECK_* 2>/dev/null || true
./.venv_h3_oceangym/bin/python tracks/h3_oceangym/run_task_suite.py \
  --scenarios PierHarbor-HoveringCamera \
  --tasks station_keeping \
  --episodes 1 \
  --difficulty easy \
  --n_multiagent 1 \
  --pollution_model ocpnet_3d \
  --out_dir runs/oceangym_h3/_smoke_h3_summarymode
```

### Re-run the FINALFIX hero suite (media + metrics)

```bash
TAG="$(date +%Y%m%d_%H%M%S)"
OUT="runs/oceangym_h3/hero_${TAG}"
rm -f /dev/shm/HOLODECK_MEM* /dev/shm/sem.HOLODECK_* 2>/dev/null || true
./.venv_h3_oceangym/bin/python tracks/h3_oceangym/run_task_suite.py \
  --scenarios PierHarbor-HoveringCamera OpenWater-HoveringCamera \
  --tasks go_to_goal_current station_keeping route_following_waypoints depth_profile_tracking formation_transit_multiagent \
         surface_pollution_cleanup_multiagent__localization surface_pollution_cleanup_multiagent__containment \
  --episodes 10 --difficulty easy --n_multiagent 8 \
  --pollution_model ocpnet_3d \
  --plume_localization_steps 400 --plume_containment_steps 200 \
  --current_npz runs/_cache/data_grounding/currents/cmems_center_uovo.npz \
  --dataset_days_per_sim_second 0.1 \
  --out_dir "${OUT}"
```

### Postprocess media (GIF + keyframes)

Run with a Python that has `opencv-python`, `imageio`, and `Pillow`.

On this machine, the H3 venv works:

```bash
./.venv_h3_oceangym/bin/python tracks/h3_oceangym/postprocess_media.py --roots "${OUT}"
```

### Scaling sweep (quant-only)

```bash
TAG="$(date +%Y%m%d_%H%M%S)"
OUT="runs/oceangym_h3/scaling_${TAG}"
rm -f /dev/shm/HOLODECK_MEM* /dev/shm/sem.HOLODECK_* 2>/dev/null || true
./.venv_h3_oceangym/bin/python tracks/h3_oceangym/run_scaling_sweep.py \
  --scenario PierHarbor-HoveringCamera \
  --task formation_transit_multiagent \
  --ns 2 4 8 10 \
  --episodes 10 --difficulty easy \
  --pollution_model ocpnet_3d \
  --current_npz runs/_cache/data_grounding/currents/cmems_center_uovo.npz \
  --dataset_days_per_sim_second 0.1 \
  --no_media \
  --out_dir "${OUT}"
```

---

## 7) Known limitations / caveats (important for paper claims)

- **Frame correctness was the root-cause of “tumbling”**:
  - `PoseSensor` in `IMUSocket` is **NED**, but our task/control logic is **NWU**.
  - FINALFIX forces `PoseSensor/VelocitySensor` to `COMSocket` (NWU) during scenario patching.
- **Currents forcing is an approximation**:
  - drift is applied via an explicit “current drift hook”, not a full water/vehicle interaction model.
- **No tide in H3 official runs**:
  - tides are not included in the official FINALFIX H3 suite; do not claim tide realism from H3.
- **Formation task success is currently strict / unstable**:
  - the FINALFIX hero and scaling sweep executed correctly, but `formation_transit_multiagent` has low/zero success under current thresholds/controller.
  - Use `rms_formation_error_m` as a continuous indicator; consider re-tuning only if the leader re-opens experiments.
- **Metrics coverage mismatch for plume tasks**:
  - `summary.csv` includes strong pollution metrics (error/mass/leakage), but not all episodes expose `energy_proxy` / `collisions` for the plume tasks (v1 harness limitation).
- **Task coverage is a subset of the canonical 10-task list**:
  - tasks requiring scene-native fish/pipeline assets are intentionally not attempted here due to asset/EULA constraints; only procedural tasks are allowed as future optional work.
