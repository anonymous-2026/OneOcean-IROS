# H3 (OceanGym / HoloOcean) — Scene Provenance & Licensing Notes

This track uses **HoloOcean 2.0.1** packaged Unreal worlds as the base “external underwater scene”.

## What we run today (Linux, packaged worlds)

- HoloOcean client: `holoocean==2.0.1`
- Packaged Ocean worlds installed at:
  - `~/.local/share/holoocean/2.0.1/worlds/Ocean/`
- Scenarios used in our H3 runs (examples):
  - `PierHarbor-HoveringCamera`
  - `OpenWater-HoveringCamera`
  - `Dam-HoveringCamera`
  - `SimpleUnderwater-Hovering`

We render both:
- **third-person**: `ViewportCapture`
- **first-person**: `LeftCamera`

## External source(s)

Workspace cache (not committed to git):
- `/data/private/user2/workspace/ocean/oneocean(iros-2026-code)/runs/_cache/external_scenes/oceangym/`
  - `OceanGym_small.zip`
  - `OceanGym_large_v2.zip`
  - `HoloOcean2.zip`

Upstream reference mentioned in requirements:
- `OceanGPT/OceanGym` (benchmark/framework reference)

## Large package status (OceanGym_large_v2.zip)

`OceanGym_large_v2.zip` contains an **Unreal Engine project** (EngineAssociation `5.3`) with Win64 editor artifacts,
and does **not** include Linux packaged `*.pak` worlds.

In this Linux environment, we cannot directly “switch” HoloOcean runtime to this large package without:
- building/cooking packaged worlds from UE5.3 (on a UE-capable build machine), then
- installing the resulting world package into HoloOcean’s worlds directory.

So far, H3 outputs are produced using the already-installed packaged Ocean worlds (see above).

## What we modified to make it “ours”

Code changes live in:
- `oneocean(iros-2026-code)/tracks/h3_oceangym/`

Key modifications:
- recording policies (camera/orbit/chase) to ensure clear underwater media,
- task implementations + metrics + manifests,
- data grounding hooks: currents exported from `combined_environment.nc` (NPZ) used in drift/advection.

## Asset inventory (what exists in the packaged Ocean worlds)

The packaged Ocean worlds ship with a large Unreal asset set. For quick “does it exist?” checks we scan:
- `~/.local/share/holoocean/2.0.1/worlds/Ocean/Linux/Manifest_UFSFiles_Linux.txt`

Examples of assets we confirmed are present (by manifest path):
- **Ships / boats (surface)**:
  - `Holodeck/Content/Worlds/ContainerShips/*` (container ship meshes/materials/textures)
  - `Holodeck/Content/Worlds/Boats/*` and `Holodeck/Content/Worlds/BoatPack/*` (multiple small boats + props)
  - `Holodeck/Content/HolodeckContent/Agents/SurfaceVessel/*` (surface vessel agent BP/mesh)
- **Pipes (structures/props)**:
  - `Holodeck/Content/Worlds/DamEnvironment/Meshes/*Pipe*`
  - `Holodeck/Content/Worlds/ModularHarborSet/Meshes/*Pipe*`
- **Aircraft (agent)**:
  - `Holodeck/Content/HolodeckContent/Agents/FixedWing/*` (FixedWing / Archer plane geometry)

What we did *not* find by name (manifest string scan):
- explicit `oil*`, `platform*`, `rig*`-named world assets (the only “oil” substring hit was unrelated: `Sm_Toiliet`).

## Redistribution / compliance note (to be finalized by Lane F)

HoloOcean / Unreal-derived assets typically have redistribution constraints (Epic UE EULA-related).
We treat all external binaries/assets as **non-redistributable** unless explicitly confirmed.

This repo contains only our **scripts** and **metadata**; heavy third-party binaries/assets remain in local caches.
