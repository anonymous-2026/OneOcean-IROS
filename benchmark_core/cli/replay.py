from __future__ import annotations

import argparse
import json

from ..replay import replay_run


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Replay a headless recording and perform integrity checks.")
    ap.add_argument("--run-dir", type=str, required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    summary = replay_run(args.run_dir)
    print(json.dumps(summary.to_dict(), indent=2))
    return 0 if summary.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())

