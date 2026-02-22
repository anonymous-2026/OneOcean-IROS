from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..build_media_package import build_media_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build conference-grade S2 media package (screenshots/video/gif/manifest)"
    )
    parser.add_argument("--run-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--scene-count", type=int, default=3)
    parser.add_argument("--progress-count", type=int, default=4)
    parser.add_argument("--topdown-count", type=int, default=3)
    parser.add_argument("--gif-fps", type=float, default=8.0)
    parser.add_argument("--gif-max-frames", type=int, default=80)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    outputs = build_media_package(
        run_dir=Path(args.run_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        scene_count=args.scene_count,
        progress_count=args.progress_count,
        topdown_count=args.topdown_count,
        gif_fps=args.gif_fps,
        gif_max_frames=args.gif_max_frames,
    )
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
