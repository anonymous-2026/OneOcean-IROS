# S3 Handoff (Agent A2, S3 Underwater 3D Quality Gate)

Date: 2026-02-23  
Backend: `oneocean_sim_s3` (SAPIEN physics; CPU software renderer on this machine)

## What changed (reset directive)

- Prior non-qualifying S3 outputs (2D-only / toy / not-underwater-looking) were deleted.
- The active A2 track targets the **3D underwater quality gate**:
  - terrain mesh (dataset-grounded `elevation`) + 3D obstacles,
  - recognizably underwater look (haze/fog + suspended particles + textured seafloor),
  - ≥2 tasks, including ≥1 **multi-agent** task,
  - per-task **3D execution media** (GIF) + screenshots.
- Storage hygiene: keep only one best-so-far run directory:
  - `runs/s3_3d_underwater_hero_v1/`

## Commands (current, 3D quality gate)

Run the quality-gate suite (both tasks × tide on/off):

```bash
/data/private/user2/workspace/embodied_platforms/maniskill2_env/.venv/bin/python \
  -m oneocean_sim_s3.experiments.run_s3_quality_gate \
  --output-root runs/s3_3d_underwater_hero_v1 \
  --variants scene \
  --tasks reef_navigation,formation_navigation \
  --tide-modes on,off \
  --episodes 2 \
  --max-steps 360
```

Suite outputs:
- `runs/s3_3d_underwater_hero_v1/suite_manifest.json`
- `runs/s3_3d_underwater_hero_v1/suite_summary.md`

Export canonical manifests for paper/web handoff:

```bash
/data/private/user2/workspace/embodied_platforms/maniskill2_env/.venv/bin/python \
  -m oneocean_sim_s3.experiments.build_s3_results_manifest \
  --suite-root runs/s3_3d_underwater_hero_v1 \
  --output-json project/results_manifest.json \
  --output-md project/results_summary.md \
  --output-csv project/results_summary.csv

/data/private/user2/workspace/embodied_platforms/maniskill2_env/.venv/bin/python \
  -m oneocean_sim_s3.experiments.build_s3_media_manifest \
  --suite-root runs/s3_3d_underwater_hero_v1 \
  --output-json project/media_manifest.json \
  --output-md project/media_summary.md
```

## Known limitations

- SAPIEN Vulkan rendering is not available on this machine; we render with a CPU software renderer.
- This suite is sized to satisfy the quality gate and produce shareable media; extend episodes/seeds for paper-grade statistics.
