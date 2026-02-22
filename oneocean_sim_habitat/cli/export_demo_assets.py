from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..export_demo_assets import ExportConfig, export_demo_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export S2 Habitat run to demo_ref-compatible JSON assets"
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--scale", type=float, default=12.0)
    parser.add_argument("--terrain-grid-size", type=int, default=56)
    parser.add_argument("--terrain-margin", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260223)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ExportConfig(
        run_dir=Path(args.run_dir),
        episode=args.episode,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        scale=args.scale,
        terrain_grid_size=args.terrain_grid_size,
        terrain_margin=args.terrain_margin,
        seed=args.seed,
    )
    outputs = export_demo_assets(cfg)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
