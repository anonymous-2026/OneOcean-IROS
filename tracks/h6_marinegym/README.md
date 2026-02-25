# Track H6 — MarineGym (Isaac Sim) integration

Goal (per `project_mgmt/agent_packets/AGENT_H6_MARINEGYM_TRACK.md`):
- Boot MarineGym in Isaac Sim **headless**, render **underwater-looking** media, then integrate our dataset
  (`combined_environment.nc` / variants) for currents + pollution tasks, including multi-agent **N=2–10** (hero **N=8/10**).

Status (2026-02-26):
- MarineGym upstream targets Isaac Sim 4.1 / Py3.10; full port to this host’s Isaac Sim 5.1 / Py3.11 is still **in progress**.
- To avoid blocking deliverables, this track includes a **fallback Isaac Sim–native** underwater scene + tasks runner that:
  - renders 3D underwater-looking media (PNG + GIF),
  - supports **N=2–10** agents (hero `NUM_AGENTS=10`),
  - uses our combined dataset (`combined_environment.nc` variants) for **current-driven drift** (`uo/vo`).

## 0) Hard constraints (project rules)

- Do **not** commit large third-party binaries/assets. Keep external sources in `runs/_cache/` (gitignored).
- No secrets in repo.
- Commit messages must be English.

## 1) Fast path (fallback): run multi-agent scene (PNG + GIF)

Uses our dataset currents (if `COMBINED_NC` exists) and applies a simple underwater postprocess (attenuation + haze + particulates).

```bash
cd "/data/private/user2/workspace/ocean/oneocean(iros-2026-code)"
OUT="runs/h6_marinegym/fallback_scene_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT" NUM_AGENTS=10 /home/shuaijun/isaacsim/python.sh tracks/h6_marinegym/run_fallback_multiagent_scene.py
ls -la "$OUT"
```

## 2) Fast path (fallback): run 2 plume tasks (PNG + GIF + metrics + manifests)

Task 1: plume localization (multi-agent search baseline)  
Task 2: plume containment (formation/ring coordination; hero `NUM_AGENTS=8/10`)

```bash
cd "/data/private/user2/workspace/ocean/oneocean(iros-2026-code)"
OUT="runs/h6_marinegym/fallback_tasks_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT" NUM_AGENTS=10 /home/shuaijun/isaacsim/python.sh tracks/h6_marinegym/run_fallback_plume_tasks.py
ls -la "$OUT"
```

Outputs (per run):
- `media_manifest.json`
- `results_manifest.json`
- `task_plume_localize/{frame_000.png,rollout.gif,metrics.json}`
- `task_plume_contain/{frame_000.png,rollout.gif,metrics.json}`

## 1) Fetch MarineGym source into local cache

```bash
cd "/data/private/user2/workspace/ocean/oneocean(iros-2026-code)"
python3 tracks/h6_marinegym/fetch_marinegym_source.py
```

Outputs a cached source tree at:
- `runs/_cache/external_scenes/marinegym/MarineGym-main/`

## 2) Patch MarineGym for Isaac Sim 5.1 + TorchRL>=0.11

MarineGym upstream assumes older TorchRL/TensorSpec APIs (Isaac Sim 4.1 era). We patch a *local copy* under `runs/_cache/`.

```bash
cd "/data/private/user2/workspace/ocean/oneocean(iros-2026-code)"
python3 tracks/h6_marinegym/patch_marinegym_for_isaacsim51.py \
  --src runs/_cache/external_scenes/marinegym/MarineGym-main
```

## 3) Install minimal Python deps into Isaac Sim Python

We run using Isaac Sim’s Python:

```bash
/home/shuaijun/isaacsim/python.sh -m pip install -U -i https://pypi.org/simple hydra-core omegaconf
/home/shuaijun/isaacsim/python.sh -m pip install -U --no-deps -i https://pypi.org/simple tensordict torchrl
/home/shuaijun/isaacsim/python.sh -m pip install -U -i https://pypi.org/simple pyvers cloudpickle importlib_metadata orjson zipp
```

## 4) Run headless smoke render (PNG + GIF)

Important: Isaac Sim startup hangs on this machine unless we disable GPU validation checks.
The runner passes:
- `--/validate/p2p/enabled=false`
- `--/validate/iommu/enabled=false`
- `--/validate/wait=0`

```bash
cd "/data/private/user2/workspace/ocean/oneocean(iros-2026-code)"
OUT="runs/h6_marinegym/hover_smoke_$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$OUT" /home/shuaijun/isaacsim/python.sh tracks/h6_marinegym/run_hover_smoke.py
ls -la "$OUT"
```

## 5) Provenance

See `tracks/h6_marinegym/scene_provenance.md` for upstream + license notes.
