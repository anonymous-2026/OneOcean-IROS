# H2 scene provenance — HoloOcean packaged worlds (Ocean)

## What we use

- Simulator/client: HoloOcean Python client (v0.5.0 API surface).
- World package: `Ocean` (Linux packaged build).

Local install (workspace machine):
- Worlds root: `/home/shuaijun/.local/share/holoocean/0.5.0/worlds/Ocean/`

Local cache (workspace machine):
- Zip: `/data/private/user2/workspace/ocean/project_mgmt/sync/_external_scene_cache/holocean_worlds/Ocean_v0.5.0_Linux.zip`
- SHA256: `1f7ad5f829ae6c9a82daa43b6d21fdfaca07711eb09c1d578f5249b13f040618`

## Upstream sources

Because the official HoloOcean GitHub repository is not accessible here (Epic↔GitHub linking is required),
we use a public mirror for the Python client code while still relying on the official world-package backend:

- Client mirror repo (public): `wellscrosby/MRGHoloOcean`
- Official package backend: `https://robots.et.byu.edu/holo/`

## Licensing / EULA notes (important)

HoloOcean worlds are packaged Unreal assets. Treat them as **third-party binaries**:
- do not commit the world binaries/assets to GitHub,
- record exact fetch/install steps in `media_manifest.json` for each run,
- for public release, we will publish *scripts/configs/manifests* and only derived results where allowed.

## Citation

- “HoloOcean: An Underwater Robotics Simulator.” ICRA 2022. DOI: `10.1109/ICRA46639.2022.9812353`

