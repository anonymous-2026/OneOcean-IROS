#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

from oneocean_sim_headless.tasks import CANONICAL_TASKS_10

from .bc_dataset import build_bc_dataset, save_dataset


def main() -> int:
    ap = argparse.ArgumentParser(description="Build a BC dataset from headless run directories (requires goal semantics).")
    ap.add_argument("--runs-root", required=True, help="Directory containing run subfolders (each with run_meta.json).")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--max-samples", type=int, default=0, help="Optional cap on samples (0 = no cap).")
    args = ap.parse_args()

    runs_root = Path(args.runs_root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = [p for p in sorted(runs_root.rglob("*")) if p.is_dir() and (p / "run_meta.json").exists()]
    ds = build_bc_dataset(run_dirs, task_vocab=list(CANONICAL_TASKS_10), max_samples=int(args.max_samples))
    out_npz = out_dir / "bc_dataset_v1.npz"
    out_meta = out_dir / "bc_dataset_v1_meta.json"
    save_dataset(ds, out_npz=out_npz, out_meta_json=out_meta)
    print(json.dumps({"samples": int(ds.x.shape[0]), "x_dim": int(ds.x.shape[1]), "y_dim": int(ds.y.shape[1])}, indent=2))
    print(f"Wrote: {out_npz}")
    print(f"Wrote: {out_meta}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

