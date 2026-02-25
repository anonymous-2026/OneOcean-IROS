# H2 — HoloOcean Track

This track uses **HoloOcean** packaged Unreal worlds as an external-scene candidate.

Constraints:
- Do **not** commit large world binaries/assets to GitHub.
- Keep all evidence in `oneocean(iros-2026-code)/runs/h2_holoocean/...` with `media_manifest.json`.

## Runtime env

- Venv: `oneocean(iros-2026-code)/.venv_h2_holoocean_mrgh/`
- Required pin: `numpy<2` (this track uses `numpy==1.26.4`)
- Extra deps for dataset-grounded runs:
  - `xarray`, `netCDF4`
  - If your default `pip` index has SSL issues, install via:
    - `pip -i https://pypi.org/simple --trusted-host pypi.org --trusted-host files.pythonhosted.org xarray netCDF4`

## Gate media (required first)

Render:
- one orbit/third-person MP4 that clearly looks underwater, and
- one MP4 showing vehicle motion in 3D.

Script:
```bash
cd oneocean(iros-2026-code)
.venv_h2_holoocean_mrgh/bin/python tracks/h2_holoocean/render_gate_media.py
```

## Plume tasks (multi-agent; must-use-data)

Runs 2 tasks and writes per-task MP4 + metrics + `media_manifest.json`:
- plume localization (multi-agent)
- plume containment+cleanup (multi-agent; supports N=2–10; demo uses N=10)

Pollution field options:
- `--pollution-model ocpnet_3d` (default): **OCPNet PollutionModel3D** advection-diffusion field driven by `combined_environment.nc` currents.
- `--pollution-model gaussian`: lightweight fallback (for debugging only; not valid for “official” must-use-data pollution runs).

```bash
cd oneocean(iros-2026-code)
.venv_h2_holoocean_mrgh/bin/python tracks/h2_holoocean/run_plume_tasks.py --scenario PierHarbor-HoveringCamera --num-agents 10 --seed 0
```

Optional tuning knobs:
```bash
cd oneocean(iros-2026-code)
.venv_h2_holoocean_mrgh/bin/python tracks/h2_holoocean/run_plume_tasks.py \\
  --scenario PierHarbor-HoveringCamera --num-agents 10 --seed 0 \\
  --pollution-model ocpnet_3d --pollution-domain-xy-m 160 \\
  --pollution-warmup-s 20 --pollution-update-period-s 2
```
