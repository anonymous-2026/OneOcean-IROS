# OCPNet Pollution Utilities

This folder contains the stabilized pollution tools used by Agent J:

- `runner.py`: synthetic 3D diffusion run with `PollutionModel3D`.
- `viz.py`: dataset analysis and figure/GIF exporters.
- `cli.py`: unified command-line entrypoint.

## Commands

From repository root:

```bash
python -m OCPNet.pollution.cli run-synthetic \
  --output-dir OCPNet/output/pollution_refactor/synthetic_run
```

```bash
python -m OCPNet.pollution.cli render-currents \
  --nc-path /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/combined_environment.nc \
  --output-dir OCPNet/output/pollution_refactor/current_viz
```

```bash
python -m OCPNet.pollution.cli analyze-nc \
  --nc-path /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/combined_environment.nc
```

```bash
python -m OCPNet.pollution.cli run-dataset-driven \
  --nc-path /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc \
  --output-dir OCPNet/output/pollution_refactor/dataset_diffusion
```

## Output layout

- `.../synthetic_run/run_summary.json`: simulation summary stats.
- `.../synthetic_run/run_report.json`: run + media manifest.
- `.../synthetic_run/media/microplastic_diffusion_panel.png`: snapshot panel (PNG).
- `.../synthetic_run/media/microplastic_diffusion_panel.eps`: snapshot panel (EPS).
- `.../synthetic_run/media/microplastic_diffusion.gif`: diffusion animation.
- `.../current_viz_v2/*_el*_az*.png`: multi-view current-over-bathymetry renders (PNG).
- `.../current_viz_v2/*_el*_az*.eps`: multi-view current-over-bathymetry renders (EPS).
- `.../dataset_diffusion/dataset_diffusion_panel.png`: dataset-driven diffusion panel.
- `.../dataset_diffusion/dataset_diffusion_panel.eps`: dataset-driven diffusion panel (EPS).
- `.../dataset_diffusion/dataset_diffusion.gif`: dataset-driven diffusion animation.
