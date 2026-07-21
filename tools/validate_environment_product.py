#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from benchmark_core.environment_contract import validate_environment_product, write_environment_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a OneOcean NetCDF environment product and optionally verify its drift NPZ export.")
    parser.add_argument("product", help="Path to the NetCDF environment product.")
    parser.add_argument("--u-var", default=None, help="Explicit eastward-current variable name.")
    parser.add_argument("--v-var", default=None, help="Explicit northward-current variable name.")
    parser.add_argument("--json-out", default=None, help="Write the complete machine-readable report to this path.")
    parser.add_argument("--strict-units", action="store_true", help="Treat unexpected units as errors instead of warnings.")
    parser.add_argument("--max-sample-points", type=int, default=100_000)
    parser.add_argument("--drift-npz", default=None, help="Optional exported drift cache for a NetCDF-to-NPZ round-trip check.")
    parser.add_argument("--time-index", type=int, default=0)
    parser.add_argument("--depth-index", type=int, default=0)
    parser.add_argument("--roundtrip-atol", type=float, default=1e-6)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate_environment_product(
        args.product,
        u_var=args.u_var,
        v_var=args.v_var,
        strict_units=bool(args.strict_units),
        max_sample_points=max(100, int(args.max_sample_points)),
        drift_npz=args.drift_npz,
        time_index=int(args.time_index),
        depth_index=int(args.depth_index),
        roundtrip_atol=float(args.roundtrip_atol),
    )
    if args.json_out:
        write_environment_report(report, Path(args.json_out))
    summary = {
        "status": report.status,
        "path": report.path,
        "mapping": report.canonical_mapping,
        "dimensions": report.dimensions,
        "warnings": sum(issue.level == "warning" for issue in report.issues),
        "errors": sum(issue.level == "error" for issue in report.issues),
        "issues": [
            {"level": issue.level, "code": issue.code, "message": issue.message}
            for issue in report.issues
        ],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if report.ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
