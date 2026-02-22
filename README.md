# OneOcean (IROS 2026) — Code

This repository contains the **ocean environment data pipeline** used to generate `combined_environment.nc` for simulation, experiments, and (eventually) public releases.

## What to run

Prerequisites:
- Set CMEMS credentials in your environment:
  - `export COPERNICUSMARINE_USERNAME=...`
  - `export COPERNICUSMARINE_PASSWORD=...`

### Canonical combined dataset (default)
```bash
python OceanEnv/Data_pipeline/run_pipeline.py --overwrite
```

### Multi-size datasets (tiny / scene / public)
```bash
python OceanEnv/Data_pipeline/generate_variants.py --which tiny,scene,public --overwrite
```

See `DATA_PIPELINE_LOG.md` for the full rationale, assumptions (tides), and reproducibility notes.

