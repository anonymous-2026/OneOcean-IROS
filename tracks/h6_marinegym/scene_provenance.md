# Track H6 — MarineGym provenance (draft)

Upstream:
- MarineGym repo: `Marine-RL/MarineGym` (MIT License)
- MarineGym paper: Chu et al., “MarineGym: A High-Performance Reinforcement Learning Platform for Underwater Robotics”, arXiv:2503.09203 (IROS 2025).

Local cache policy:
- MarineGym sources are fetched into `runs/_cache/external_scenes/marinegym/` (gitignored).
- We do **not** commit upstream code or USD assets into this repo.

Local compatibility patches (this machine):
- Isaac Sim installed: `/home/shuaijun/isaacsim` (5.1.x)
- TorchRL/TensorSpec API mismatch vs upstream (Isaac Sim 4.1 era):
  - We patch spec imports and a small number of spec class usages in the cached source tree.
  - Patch script: `tracks/h6_marinegym/patch_marinegym_for_isaacsim51.py`
- Isaac Sim startup hang on this host due to GPU validation when IOMMU is enabled:
  - We run with extra kit args:
    - `--/validate/p2p/enabled=false`
    - `--/validate/p2p/memoryCheck/enabled=false`
    - `--/validate/iommu/enabled=false`
    - `--/validate/wait=0`

Fallback “Isaac Sim–native” scene/tasks (current deliverable):
- Files:
  - `tracks/h6_marinegym/run_fallback_multiagent_scene.py`
  - `tracks/h6_marinegym/run_fallback_plume_tasks.py`
- Upstream assets:
  - None (procedural USD geometry only; no third-party meshes/textures committed or required).
- “Looks underwater” strategy:
  - bluish lighting + particles (USD spheres) + seabed mesh with vertex colors,
  - plus a screen-space underwater postprocess applied to rendered frames using Replicator `distance_to_camera`
    (attenuation + haze tint + suspended particulate noise).

What makes the fallback runs “ours”:
- Must-use-data policy (currents):
  - Both fallback runners load `combined_environment.nc` (default: the `tiny` variant) and sample `uo/vo` (nearest-neighbor)
    to drive current-driven drift/advection in the scene/tasks.
- Pollution task signal:
  - A synthetic “plume concentration” field is advected by our dataset currents and used in observations/metrics:
    - Task 1: plume localization (success when estimated center is within `success_radius`)
    - Task 2: plume containment (ring formation; leakage proxy computed from plume concentration outside the containment radius)
- Multi-agent scaling:
  - Supports `NUM_AGENTS=2–10`; hero qualitative run targets `NUM_AGENTS=8/10`.

Known limitations (explicit):
- The fallback tasks are intentionally lightweight and deterministic (high-throughput), not a full rigid-body hydrodynamics model.
- Current sampling is nearest-neighbor at `time=0`; future work should add time-stepping and bilinear sampling for realism.
- MarineGym itself is not yet fully booted under Isaac Sim 5.1; the fallback exists so this track can still produce
  usable media + tasks + metrics without waiting on a full upstream port.
