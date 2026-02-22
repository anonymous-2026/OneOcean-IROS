from __future__ import annotations

import argparse
import json

from ..prepare_drift_cache import prepare_drift_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare lightweight drift cache npz from combined_environment.nc"
    )
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--depth-index", type=int, default=0)
    parser.add_argument("--disable-tides", action="store_true")
    parser.add_argument("--disable-bathymetry", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = prepare_drift_cache(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        time_index=args.time_index,
        depth_index=args.depth_index,
        include_tides=not args.disable_tides,
        include_bathymetry=not args.disable_bathymetry,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
