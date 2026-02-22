from __future__ import annotations

import argparse
import json

from ..publish_e2_demo_assets import publish_e2_demo_assets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publish S2 demo export JSON into demo/assets/data-compatible filenames"
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--target-dir", type=str, default=None)
    parser.add_argument("--map-name", type=str, default="ocean_map_data.json")
    parser.add_argument("--path-name", type=str, default="ocean_path_data.json")
    parser.add_argument("--no-manifest", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = publish_e2_demo_assets(
        run_dir=args.run_dir,
        target_dir=args.target_dir,
        map_name=args.map_name,
        path_name=args.path_name,
        write_manifest=not args.no_manifest,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
