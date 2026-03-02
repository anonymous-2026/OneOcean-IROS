#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _parse_int_list(spec: str) -> list[int]:
    s = str(spec or "").strip()
    if not s:
        return []
    out: list[int] = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _merge_csvs(out_path: Path, inputs: list[Path]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header: list[str] | None = None
    rows: list[dict[str, Any]] = []
    for p in inputs:
        if not p.exists():
            continue
        with p.open("r", encoding="utf-8", newline="") as f:
            r = csv.DictReader(f)
            if header is None:
                header = list(r.fieldnames or [])
            for row in r:
                rows.append(row)
    if header is None:
        out_path.write_text("", encoding="utf-8")
        return
    with out_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        for row in rows:
            w.writerow(row)


@dataclass(frozen=True)
class Job:
    shard_index: int
    out_dir: Path
    gpu_id: int | None
    cmd: list[str]


def main() -> int:
    ap = argparse.ArgumentParser(description="Launch multiple run_matrix shards concurrently and merge outputs.")
    ap.add_argument("--out-dir", required=True, help="Parent output directory. Shards write to out_dir/shard_XX.")
    ap.add_argument("--shards", type=int, default=8, help="Number of shards (parallel workers).")
    ap.add_argument("--gpu-ids", type=str, default="0,1,2,3,4,5,6,7", help="Comma list of GPU ids for CUDA_VISIBLE_DEVICES mapping.")
    ap.add_argument("--max-parallel", type=int, default=0, help="Optional cap on concurrent processes (0 = use shards).")
    ap.add_argument("--poll-s", type=float, default=1.0, help="Polling interval for job completion.")
    args, rest = ap.parse_known_args()
    # Allow passing run_matrix args after a `--` separator; argparse keeps it in `rest`.
    if rest and rest[0] == "--":
        rest = rest[1:]

    shards = int(args.shards)
    if shards < 1:
        raise SystemExit("--shards must be >= 1")
    gpu_ids = _parse_int_list(str(args.gpu_ids))
    if not gpu_ids:
        gpu_ids = []
    max_parallel = int(args.max_parallel) if int(args.max_parallel) > 0 else int(shards)
    poll_s = float(max(0.05, float(args.poll_s)))

    parent = Path(args.out_dir).expanduser().resolve()
    parent.mkdir(parents=True, exist_ok=True)

    # Record farm meta (for reproducibility).
    meta = {
        "time_local": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "cwd": str(Path.cwd().resolve()),
        "argv": list(sys.argv),
        "shards": int(shards),
        "gpu_ids": gpu_ids,
        "max_parallel": int(max_parallel),
        "child_run_matrix_args": rest,
    }
    (parent / "farm_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    jobs: list[Job] = []
    for si in range(int(shards)):
        shard_out = parent / f"shard_{si:02d}"
        shard_out.mkdir(parents=True, exist_ok=True)
        gpu_id = gpu_ids[si % len(gpu_ids)] if gpu_ids else None
        cmd = [
            sys.executable,
            "-m",
            "oneocean_sim_headless.cli.run_matrix",
            "--out-dir",
            str(shard_out),
            "--shard-index",
            str(int(si)),
            "--shard-count",
            str(int(shards)),
            *rest,
        ]
        jobs.append(Job(shard_index=int(si), out_dir=shard_out, gpu_id=gpu_id, cmd=cmd))

    running: dict[int, subprocess.Popen[str]] = {}
    completed: dict[int, int] = {}
    queue = jobs.copy()

    def start_job(j: Job) -> None:
        env = dict(os.environ)
        if j.gpu_id is None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(int(j.gpu_id))
        # Keep stdout/stderr in per-shard logs.
        log_path = j.out_dir / "farm_child.log"
        fp = log_path.open("w", encoding="utf-8")
        p = subprocess.Popen(j.cmd, stdout=fp, stderr=subprocess.STDOUT, text=True, env=env)
        running[j.shard_index] = p

    t0 = time.time()
    while queue or running:
        while queue and len(running) < int(max_parallel):
            start_job(queue.pop(0))
        done = []
        for si, p in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            completed[si] = int(rc)
            done.append(si)
        for si in done:
            running.pop(si, None)
        if queue or running:
            time.sleep(poll_s)

    # Fail fast if any shard failed.
    bad = {k: v for k, v in completed.items() if int(v) != 0}
    if bad:
        (parent / "farm_failures.json").write_text(json.dumps(bad, indent=2), encoding="utf-8")
        print(json.dumps({"ok": False, "failed_shards": bad, "out_dir": str(parent)}, indent=2))
        return 2

    # Merge shard CSV outputs.
    shard_summaries = [parent / f"shard_{si:02d}" / "summary.csv" for si in range(int(shards))]
    shard_matrix = [parent / f"shard_{si:02d}" / "matrix_results.csv" for si in range(int(shards))]
    _merge_csvs(parent / "summary.csv", shard_summaries)
    _merge_csvs(parent / "matrix_results.csv", shard_matrix)
    (parent / "farm_done.json").write_text(json.dumps({"ok": True, "elapsed_s": float(time.time() - t0)}, indent=2), encoding="utf-8")

    print(json.dumps({"ok": True, "out_dir": str(parent), "summary_csv": str((parent / 'summary.csv').resolve())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
