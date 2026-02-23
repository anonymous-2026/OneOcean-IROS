from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..external_assets.polyhaven import ensure_underwater_asset_pack


def main() -> int:
    parser = argparse.ArgumentParser(description="Fetch small CC0 underwater asset pack from Poly Haven (local cache).")
    parser.add_argument(
        "--output-dir",
        default="runs/_cache/polyhaven",
        help="Cache directory (kept local; do not commit). Default: runs/_cache/polyhaven",
    )
    parser.add_argument("--resolution", default="1k", choices=("1k", "2k", "4k", "8k"))
    parser.add_argument("--sand-texture-id", default="aerial_sand")
    parser.add_argument("--rock-model-id", default="rock_07")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    pack = ensure_underwater_asset_pack(
        out_dir=Path(args.output_dir),
        resolution=str(args.resolution),
        sand_texture_id=str(args.sand_texture_id),
        rock_model_id=str(args.rock_model_id),
        overwrite=bool(args.overwrite),
    )

    out = {
        "sand_diffuse": str(pack["sand_texture"].primary_path),
        "sand_manifest": str(pack["sand_texture"].manifest_path),
        "rock_gltf": str(pack["rock_model"].primary_path),
        "rock_manifest": str(pack["rock_model"].manifest_path),
        "license_note": "All downloaded assets are CC0 via Poly Haven; see per-asset manifest JSONs.",
    }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

