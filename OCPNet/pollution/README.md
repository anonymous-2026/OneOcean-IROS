# OCPNet Pollution Utilities

This folder contains the stabilized pollution tools used by Agent J:

- `runner.py`: synthetic 3D diffusion run with `PollutionModel3D`.
- `viz.py`: dataset analysis and figure/GIF exporters.
- `cli.py`: unified command-line entrypoint.

## Commands

From repository root:

```bash
# Activate an environment that has xarray/netCDF4/matplotlib installed.
. /data/private/user2/workspace/ocean/.venv_ocean/bin/activate
PYTHONPATH=. python -m OCPNet.pollution.cli run-synthetic
```

```bash
PYTHONPATH=. python -m OCPNet.pollution.cli render-currents \
  --nc-path /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc
```

```bash
PYTHONPATH=. python -m OCPNet.pollution.cli analyze-nc \
  --nc-path /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/scene/combined/combined_environment.nc
```

```bash
PYTHONPATH=. python -m OCPNet.pollution.cli run-dataset-suite \
  --nc-path /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/public25_japan_surface/combined/combined_environment.nc
```

```bash
PYTHONPATH=. python -m OCPNet.pollution.cli run-dataset-multi \
  --nc-path /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/public25_japan_surface/combined/combined_environment.nc
```

```bash
PYTHONPATH=. python -m OCPNet.pollution.cli run-dataset-multi-suite \
  --nc-path /data/private/user2/workspace/ocean/OceanEnv/Data_pipeline/Data/Combined/variants/public25_japan_surface/combined/combined_environment.nc
```

## Output layout

- Synthetic diffusion (pure synthetic field; basemap only):
  - `.../synthetic_latest/run_report.json`: run + media manifest.
  - `.../synthetic_latest/media/microplastic_diffusion_panel.png` + `.eps`
  - `.../synthetic_latest/media/microplastic_diffusion.gif`
- Dataset-driven diffusion proxy (multi-seed; uses real `uo/vo` + `land_mask`):
  - `.../dataset_latest_suite/dataset_latest_suite_suite_manifest.json`: seed list + output index.
  - `.../dataset_latest_suite/dataset_latest_suite_seed*_panel.png` + `.eps`
  - `.../dataset_latest_suite/dataset_latest_suite_seed*.gif`
- Dataset-driven multi-pollutant overlay (Microplastics + Oil + Aggregation/Weathering reactions):
  - `.../dataset_latest_multispecies/dataset_latest_multispecies_panel.png` + `.eps`
  - `.../dataset_latest_multispecies/dataset_latest_multispecies.gif`
  - `.../dataset_latest_multispecies/dataset_latest_multispecies_manifest.json`
  - Multi-seed suite:
    - `.../dataset_latest_multispecies_suite/dataset_latest_multispecies_suite_suite_manifest.json`
    - `.../dataset_latest_multispecies_suite/dataset_latest_multispecies_suite_seed*_panel.png` + `.eps`
    - `.../dataset_latest_multispecies_suite/dataset_latest_multispecies_suite_seed*.gif`
- 3D currents (final diffusion-like colormap; direction-colored arrows; multi-view):
  - `.../current_viz_final/*_diffusion_el*_az*.png` + `.eps`
