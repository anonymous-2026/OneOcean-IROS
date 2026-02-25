# Track H4 — UNav-Sim / AirSim scene provenance (workspace + reproducibility note)

This file documents the upstream scene/simulator provenance for the H4 track.

## Status (2026-02-26)

This track is **abandoned on this machine** due to lack of a runnable Unreal underwater world build:
- No local Unreal Engine / `UnrealEditor` installation is present.
- UNav-Sim does not publish a downloadable packaged underwater world.
- MIMIR-UW Zenodo downloads are offline datasets (recorded sensor dumps), not runnable UE projects/packages.

If a packaged build (Linux preferred) of an underwater UNav-Sim/AirSim world becomes available, we can revive H4 by reusing `oneocean_sim_unavsim/` without changing the OneOcean task logic.

## Upstream

- Name: UNav-Sim (UE-based underwater robotics simulator; AirSim-derived)
- Repo: `https://github.com/open-airlab/UNav-Sim`
- Local cache (workspace-only; not committed to git):
  - `/data/private/user2/workspace/ocean/project_mgmt/sync/_external_scene_cache/unavsim_src_20260225/`
- Commit pinned for this stage:
  - `593386c06850a88f8afc7fb0bec983fb52dda665`
- License: MIT (see upstream `LICENSE`)

## What we use it for in OneOcean

- External high-quality underwater scene and renderer (UE).
- AirSim RPC interface for multi-vehicle control and sensors.

## What is “ours” (OneOcean additions)

We do not modify upstream Unreal assets in git.

Our “ours” layer lives in:
- `oneocean_sim_unavsim/` (Python task runner + dataset-driven drift & plume proxy)

## How to reproduce (high level)

1) Obtain UNav-Sim source (or a packaged world build) outside git.
2) Launch the Unreal world (Editor or packaged).
3) Generate an AirSim `settings.json` with multi-vehicle support:
   - `python -m oneocean_sim_unavsim.cli.write_settings --vehicle-count 8`
4) Run tasks + media export:
   - `python -m oneocean_sim_unavsim.cli.run_plume_tasks --vehicle-count 8 --output-dir runs/h4_unavsim_plume_hero_v1`
