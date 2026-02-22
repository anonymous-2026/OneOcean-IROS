from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..build_compact_bundle import build_compact_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build compact media bundle from an S2 Habitat run directory"
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--screenshot-count", type=int, default=4)
    parser.add_argument("--topdown-count", type=int, default=6)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_compact_bundle(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        screenshot_count=args.screenshot_count,
        topdown_count=args.topdown_count,
        include_video=not args.no_video,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
