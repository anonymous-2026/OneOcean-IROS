# PollutionModel3D (refactored runtime notes)

This module provides a 3D Eulerian pollution transport model used by OCPNet.

## Key changes

- Package imports are now relative and module-safe.
- Runtime crash in mass-balance reporting is fixed.
- A stable CLI exists at `OCPNet/pollution/cli.py` for end-to-end execution.
- Old notebook-export scripts were removed from the runtime path.

## Quick run

From repo root:

```bash
python -m OCPNet.pollution.cli run-synthetic --output-dir OCPNet/output/pollution_refactor/synthetic_run
```

The command writes:

- `run_summary.json`
- `run_report.json`
- diffusion panel PNG and GIF under `.../media/`

## Dataset-driven current visualization

```bash
python -m OCPNet.pollution.cli render-currents \
  --nc-path /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/combined_environment.nc \
  --output-dir OCPNet/output/pollution_refactor/current_viz
```

This generates:

- `base_current_3d.png`
- `total_current_3d.png`

## Dataset-driven diffusion proxy visualization

```bash
python -m OCPNet.pollution.cli run-dataset-driven \
  --nc-path /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc \
  --output-dir OCPNet/output/pollution_refactor/dataset_diffusion
```

This generates:

- `dataset_diffusion_panel.png`
- `dataset_diffusion.gif`

## NetCDF stats summary

```bash
python -m OCPNet.pollution.cli analyze-nc \
  --nc-path /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/combined_environment.nc
```
