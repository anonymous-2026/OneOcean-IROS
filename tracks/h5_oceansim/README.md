# Track H5 — OceanSim (Isaac Sim) candidate

This track targets **underwater sensing realism** (UW camera + imaging sonar) via OceanSim (Isaac Sim extension).

## Setup (local-only)

Fetch a minimal OceanSim extension snapshot into Isaac Sim `extsUser`:

```bash
cd "/data/private/user2/workspace/ocean/oneocean(iros-2026-code)"
python3 tracks/h5_oceansim/fetch_oceansim_extension.py --overwrite
```

## Run (headless)

```bash
cd "/data/private/user2/workspace/ocean/oneocean(iros-2026-code)"
OUT="runs/h5_oceansim/demo_$(date +%Y%m%d_%H%M%S)"
/home/shuaijun/isaacsim/python.sh tracks/h5_oceansim/run_headless_oceansim_demo.py \
  --out "$OUT" --frames 120 --warmup_frames 20 --gpu 0
```

Make MP4/GIF from camera frames:

```bash
bash tracks/h5_oceansim/make_video.sh "$OUT/uw_camera" "$OUT/uw_camera.mp4" "$OUT/uw_camera.gif"
```

Notes:
- This demo adds Kit flags to disable P2P/IOMMU validation and force single-GPU rendering (avoids multi-GPU startup delays).
- If `ffmpeg` is not installed, `make_video.sh` will write **GIF only** (no MP4).

## Provenance

See `tracks/h5_oceansim/scene_provenance.md`.
