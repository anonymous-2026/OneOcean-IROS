# OceanGym track (H3) — scene provenance (draft)

This document is part of the **H3 OceanGym candidate track**.

## Upstream sources

- OceanGym repo (task/benchmark code + configs + docs):
  - Local snapshot for reference (workspace-only): `feedback/ref_underwater_scenes/_src/OceanGym-main/`
  - Paper: arXiv:2509.26536 (see `feedback/ref_underwater_scenes/papers/REF_SCOUT_ALL.bib`)
- OceanGym environment binaries/world files:
  - Upstream distributes `OceanGym_small.zip` / `OceanGym_large.zip` via Baidu/Google Drive links in their README.
  - These zips are **large** (multi-GB) and must be kept **out of git**.

## License / redistribution constraints (important)

- OceanGym is built on Unreal Engine (UE 5.3) and references HoloOcean-inspired components.
- Unreal/Epic EULA constraints may apply to binaries and world assets.
- For our public release:
  - we should avoid redistributing third-party Unreal binaries/assets unless explicitly permitted,
  - prefer releasing our **scripts**, **configs**, **task definitions**, and **dataset-derived fields**,
  - document how users can fetch upstream assets themselves.

## What makes “our scene/tasks” (planned modifications)

Once the OceanGym world is runnable in our pipeline, we will make it “ours” by:
- driving drift/forces from our `combined_environment.nc` current fields,
- driving pollution/microplastics concentration from our diffusion model,
- defining our own tasks/metrics (including multi-agent N=2–10; hero N=8/10).

## Local cache policy

All third-party zips/extracted binaries go under:
- `runs/_cache/external_scenes/oceangym/`

and are ignored by git (see `.gitignore`).

