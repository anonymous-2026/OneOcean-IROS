# H2 — HoloOcean Track

This track uses **HoloOcean** packaged Unreal worlds as an external-scene candidate.

Constraints:
- Do **not** commit large world binaries/assets to GitHub.
- Keep all evidence in `oneocean(iros-2026-code)/runs/h2_holoocean/...` with `media_manifest.json`.

## Runtime env

- Venv: `oneocean(iros-2026-code)/.venv_h2_holoocean_mrgh/`
- Required pin: `numpy<2` (this track uses `numpy==1.26.4`)

## Gate media (required first)

Render:
- one orbit/third-person MP4 that clearly looks underwater, and
- one MP4 showing vehicle motion in 3D.

Script:
```bash
cd oneocean(iros-2026-code)
.venv_h2_holoocean_mrgh/bin/python tracks/h2_holoocean/render_gate_media.py
```

