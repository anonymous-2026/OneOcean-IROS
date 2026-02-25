## H3 (OceanGym) track — integration notes

This folder is the **H3 OceanGym candidate track** implementation area.

Key constraint:
- OceanGym decision tasks are documented as **Windows-only** in the upstream README (UE5 editor workflow).
- Our goal is to reach a **packaged (non-editor) runnable** underwater world on Linux if possible, then inject our data (currents + pollution) and run our tasks.

Do **not** commit any third-party binaries or large assets. Use `runs/_cache/external_scenes/oceangym/` for local cache.

### What we need locally (external, not committed)

1) OceanGym environment zip (small or large):
- Upstream links are in `feedback/ref_underwater_scenes/_src/OceanGym-main/README.md` (workspace repo).
- Preferred local location (inside this repo; ignored by git):
  - `runs/_cache/external_scenes/oceangym/OceanGym_small.zip`
  - or `runs/_cache/external_scenes/oceangym/OceanGym_large.zip`

2) Install the OceanGym-provided `holoocean` client (not the PyPI placeholder):
- Use `tools/external_scenes/oceangym_extract_client.py` to extract only the `client/` folder from the zip.
- Then install it into a dedicated Python environment.

### Next implementation steps (once client is installed)

- Discover the installed `holoocean` API (package manager functions, scenario loader).
- Run a baseline scenario to export a **3D underwater** PNG + MP4/GIF.
- Add our data-driven hooks:
  - `combined_environment.nc` currents → drift/force
  - pollution diffusion field → observation/reward/success
- Implement our tasks with **N=2–10** vehicles (hero attempt N=8/10).

