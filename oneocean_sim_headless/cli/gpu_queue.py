#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GpuStat:
    index: int
    mem_total_mb: int
    mem_used_mb: int
    util_gpu_pct: int


@dataclass(frozen=True)
class Job:
    name: str
    cmd: list[str]
    cwd: str | None = None
    env: dict[str, str] | None = None


def _query_gpus() -> list[GpuStat]:
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        )
    except Exception:
        return []

    stats: list[GpuStat] = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 4:
            continue
        try:
            stats.append(
                GpuStat(
                    index=int(parts[0]),
                    mem_total_mb=int(parts[1]),
                    mem_used_mb=int(parts[2]),
                    util_gpu_pct=int(parts[3]),
                )
            )
        except Exception:
            continue
    return stats


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


def _load_jobs(path: Path) -> list[Job]:
    jobs: list[Job] = []
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        d = json.loads(s)
        name = str(d.get("name") or f"job_{i:04d}")
        cmd_raw = d.get("cmd")
        if isinstance(cmd_raw, str):
            cmd = shlex.split(cmd_raw)
        elif isinstance(cmd_raw, list) and all(isinstance(x, str) for x in cmd_raw):
            cmd = [str(x) for x in cmd_raw]
        else:
            raise ValueError(f"Invalid cmd for job line {i+1}: expected string or list[str]")
        cwd = d.get("cwd")
        env = d.get("env")
        if cwd is not None:
            cwd = str(cwd)
        if env is not None:
            env = {str(k): str(v) for k, v in dict(env).items()}
        jobs.append(Job(name=name, cmd=cmd, cwd=cwd, env=env))
    return jobs


def _choose_gpu(
    *,
    allow: list[int],
    running_count: dict[int, int],
    per_gpu: int,
) -> int | None:
    stats = _query_gpus()
    if not stats:
        return None
    if allow:
        stats = [s for s in stats if s.index in allow]
    # Filter by per-gpu concurrency.
    cand = [s for s in stats if int(running_count.get(int(s.index), 0)) < int(per_gpu)]
    if not cand:
        return None
    # Prefer: lowest running count -> lowest mem_used -> lowest util.
    cand.sort(key=lambda s: (int(running_count.get(int(s.index), 0)), int(s.mem_used_mb), int(s.util_gpu_pct)))
    return int(cand[0].index)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="GPU-aware local job queue (one process per GPU by default).")
    ap.add_argument("--jobs-jsonl", type=str, required=True, help="JSONL file: {name, cmd, cwd?, env?}")
    ap.add_argument("--gpu-ids", type=str, default="", help="Optional comma list of allowed GPU indices.")
    ap.add_argument("--per-gpu", type=int, default=1, help="Max concurrent jobs per GPU.")
    ap.add_argument("--poll-s", type=float, default=2.0)
    ap.add_argument("--log-dir", type=str, default="", help="Where to write per-job logs and queue meta.")
    ap.add_argument("--fail-fast", action="store_true", help="Stop launching new jobs after the first failure.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    jobs_path = Path(args.jobs_jsonl).expanduser().resolve()
    if not jobs_path.exists():
        raise SystemExit(f"jobs file not found: {jobs_path}")
    jobs = _load_jobs(jobs_path)
    if not jobs:
        raise SystemExit("no jobs found")

    allow = _parse_int_list(str(args.gpu_ids))
    per_gpu = int(max(1, int(args.per_gpu)))
    poll_s = float(max(0.2, float(args.poll_s)))

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir).expanduser().resolve() if str(args.log_dir).strip() else (Path("runs") / "_queue_logs" / f"gpu_queue_{stamp}").resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "time_local": datetime.now().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "cwd": str(Path.cwd().resolve()),
        "argv": list(sys.argv),
        "jobs_jsonl": str(jobs_path),
        "job_count": int(len(jobs)),
        "allow_gpu_ids": allow,
        "per_gpu": int(per_gpu),
    }
    (log_dir / "queue_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    pending = jobs.copy()
    running: dict[int, dict[str, Any]] = {}  # pid -> {job,gpu,proc,log_path,start_s}
    running_count: dict[int, int] = {}
    completed: list[dict[str, Any]] = []
    failed = False

    def start_job(job: Job, gpu_id: int | None) -> None:
        env = dict(os.environ)
        if gpu_id is None:
            env.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            env["CUDA_VISIBLE_DEVICES"] = str(int(gpu_id))
        if job.env:
            env.update({str(k): str(v) for k, v in job.env.items()})

        log_path = log_dir / f"{job.name}.log"
        fp = log_path.open("w", encoding="utf-8")
        proc = subprocess.Popen(job.cmd, cwd=job.cwd, stdout=fp, stderr=subprocess.STDOUT, text=True, env=env)
        running[int(proc.pid)] = {"job": job, "gpu": gpu_id, "proc": proc, "log_path": str(log_path), "start_s": time.time()}
        if gpu_id is not None:
            running_count[int(gpu_id)] = int(running_count.get(int(gpu_id), 0)) + 1

    t0 = time.time()
    while pending or running:
        while pending and not (failed and bool(args.fail_fast)):
            gpu_id = _choose_gpu(allow=allow, running_count=running_count, per_gpu=per_gpu)
            if gpu_id is None:
                # If no GPUs are available/visible, run sequentially on CPU (or current CUDA_VISIBLE_DEVICES).
                if _query_gpus():
                    break
                job = pending.pop(0)
                start_job(job, None)
            else:
                job = pending.pop(0)
                start_job(job, gpu_id)

        # Poll completions
        done_pids: list[int] = []
        for pid, rec in list(running.items()):
            proc: subprocess.Popen[str] = rec["proc"]
            rc = proc.poll()
            if rc is None:
                continue
            done_pids.append(pid)
            gpu_id = rec["gpu"]
            if gpu_id is not None:
                running_count[int(gpu_id)] = max(0, int(running_count.get(int(gpu_id), 0)) - 1)
            completed.append(
                {
                    "name": rec["job"].name,
                    "cmd": rec["job"].cmd,
                    "cwd": rec["job"].cwd,
                    "gpu": gpu_id,
                    "returncode": int(rc),
                    "elapsed_s": float(time.time() - float(rec["start_s"])),
                    "log_path": rec["log_path"],
                }
            )
            if int(rc) != 0:
                failed = True
        for pid in done_pids:
            running.pop(pid, None)

        if pending or running:
            time.sleep(poll_s)

    out = {"ok": not failed, "elapsed_s": float(time.time() - t0), "completed": completed}
    (log_dir / "queue_results.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    if failed:
        print(json.dumps({"ok": False, "log_dir": str(log_dir), "failures": [c for c in completed if int(c["returncode"]) != 0]}, indent=2))
        return 2
    print(json.dumps({"ok": True, "log_dir": str(log_dir), "jobs": int(len(completed))}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

