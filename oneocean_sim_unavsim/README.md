# OneOcean H4 — UNav-Sim / AirSim Track

This folder contains the **Python-side task runner** for Track **H4**.

Scope:
- We do **not** commit Unreal assets or packaged binaries to git.
- We assume an AirSim/UNav-Sim instance is already running (UE Editor or packaged build).

## 0) External dependency (workspace-only)

UNav-Sim source (local cache, not committed):
- `/data/private/user2/workspace/ocean/project_mgmt/sync/_external_scene_cache/unavsim_src_20260225`

Upstream:
- Repo: `open-airlab/UNav-Sim`
- Commit used for provenance: `593386c06850a88f8afc7fb0bec983fb52dda665`

## 1) Python deps

From code repo root:

```bash
python -m pip install -r requirements.txt
```

## 2) Write `settings.json` (multi-vehicle, external physics)

```bash
cd /data/private/user2/workspace/ocean/oneocean(iros-2026-code)
PYTHONPATH=. python -m oneocean_sim_unavsim.cli.write_settings --vehicle-count 8
```

This writes the default AirSim config to:
- `~/Documents/AirSim/settings.json`

## 3) Run tasks (requires a running UE world)

Start UNav-Sim / AirSim world first, then:

```bash
cd /data/private/user2/workspace/ocean/oneocean(iros-2026-code)
PYTHONPATH=. python -m oneocean_sim_unavsim.cli.run_plume_tasks \\
  --vehicle-count 8 \\
  --output-dir runs/h4_unavsim_plume_hero_v1
```

Outputs:
- `runs/h4_unavsim_plume_hero_v1/rollout.mp4`
- `runs/h4_unavsim_plume_hero_v1/metrics.json`
- `runs/h4_unavsim_plume_hero_v1/media_manifest.json`
- `runs/h4_unavsim_plume_hero_v1/results_manifest.json`

