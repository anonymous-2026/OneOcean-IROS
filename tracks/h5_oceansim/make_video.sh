#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <png_dir> <out.mp4> <out.gif>"
  exit 2
fi

PNG_DIR="$1"
OUT_MP4="$2"
OUT_GIF="$3"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

python3 - "$PNG_DIR" "$TMP_DIR" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.mkdir(parents=True, exist_ok=True)

pat = re.compile(r"UW_image_(\d+)\.png$")
items = []
for p in src.glob("UW_image_*.png"):
    m = pat.search(p.name)
    if m:
        items.append((int(m.group(1)), p))
items.sort(key=lambda x: x[0])
if not items:
    raise SystemExit(f"No UW_image_*.png found in {src}")
for i, (_, p) in enumerate(items):
    (dst / f"frame_{i:05d}.png").symlink_to(p.resolve())
print(f"linked {len(items)} frames")
PY

if command -v ffmpeg >/dev/null 2>&1; then
  ffmpeg -y -hide_banner -loglevel error -framerate 30 -i "$TMP_DIR/frame_%05d.png" \
    -c:v libx264 -pix_fmt yuv420p -crf 20 "$OUT_MP4"
  ffmpeg -y -hide_banner -loglevel error -i "$OUT_MP4" -vf "fps=15,scale=960:-1:flags=lanczos" "$OUT_GIF"
  echo "[ok] wrote: $OUT_MP4"
  echo "[ok] wrote: $OUT_GIF"
  exit 0
fi

echo "[warn] ffmpeg not found; writing GIF only (no MP4)."

python3 - "$TMP_DIR" "$OUT_GIF" <<'PY'
import sys
from pathlib import Path

from PIL import Image

src = Path(sys.argv[1])
out = Path(sys.argv[2])

frames = sorted(src.glob("frame_*.png"))
if not frames:
    raise SystemExit(f"No frame_*.png found in {src}")

imgs = []
for p in frames:
    im = Image.open(p).convert("RGB")
    # Downscale to keep GIF size reasonable.
    im.thumbnail((960, 960), Image.Resampling.LANCZOS)
    imgs.append(im)

duration_ms = int(1000 / 15)  # ~15 fps
imgs[0].save(out, save_all=True, append_images=imgs[1:], duration=duration_ms, loop=0, optimize=False)
print(f"wrote GIF: {out} ({len(imgs)} frames)")
PY

echo "[ok] wrote: $OUT_GIF"
