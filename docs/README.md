# Web Documentation

`docs/` contains the public website sources served by GitHub Pages.

- `index.html`: primary project website for the paper and public release.
- `platform/index.html`: platform/data usage page with a static GitHub-Pages explorer for a lightweight current-field subset.
- `oneocean_paper.pdf`: paper PDF linked from the project website.
- `static/`: website assets (logo, copied paper figures, demo media, CSS, JavaScript, and the exported web-data subset under `static/data/`).

Local preview:

```bash
cd docs
python -m http.server 8000
```

To rebuild the platform explorer dataset from a local `combined_environment.nc`:

```bash
python tools/export_platform_web_data.py \
  --input /path/to/combined_environment.nc
```
