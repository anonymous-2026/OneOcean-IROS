from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..export_s1_compatible_metrics import CompatConfig, export_s1_compatible_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export S2 Habitat run metrics to S1-compatible schema"
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--dt-sec", type=float, default=1.0)
    parser.add_argument("--output-dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = CompatConfig(
        run_dir=Path(args.run_dir),
        episode=args.episode,
        dt_sec=args.dt_sec,
        output_dir=Path(args.output_dir) if args.output_dir else None,
    )
    outputs = export_s1_compatible_metrics(cfg)
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
