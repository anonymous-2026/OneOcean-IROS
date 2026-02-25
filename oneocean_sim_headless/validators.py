from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    reason: str


def _count_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as fp:
        r = csv.reader(fp)
        header = next(r, None)
        if header is None:
            return 0
        return sum(1 for _ in r)


def _check_monotonic_t(path: Path, t_col: int = 0) -> bool:
    prev = None
    with path.open("r", encoding="utf-8", newline="") as fp:
        r = csv.reader(fp)
        header = next(r, None)
        if header is None:
            return False
        for row in r:
            t = float(row[t_col])
            if prev is not None and t + 1e-12 < prev:
                return False
            prev = t
    return True


def validate_run_dir(run_dir: str | Path) -> ValidationResult:
    root = Path(run_dir).expanduser().resolve()
    meta_path = root / "run_meta.json"
    if not meta_path.exists():
        return ValidationResult(False, "missing run_meta.json")
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception as e:
        return ValidationResult(False, f"invalid run_meta.json: {e}")

    n_agents = int(meta.get("n_agents", 0))
    if n_agents <= 0:
        return ValidationResult(False, "run_meta.json missing n_agents")

    expected_rows = None
    for i in range(n_agents):
        agent_dir = root / "agents" / f"agent_{i:03d}"
        pose = agent_dir / "pose_groundtruth" / "data.csv"
        act = agent_dir / "actions" / "data.csv"
        cur = agent_dir / "obs" / "local_current" / "data.csv"
        probe = agent_dir / "obs" / "pollution_probe" / "data.csv"
        for p in (pose, act, cur, probe):
            if not p.exists():
                return ValidationResult(False, f"missing stream: {p}")
            if not _check_monotonic_t(p, t_col=0):
                return ValidationResult(False, f"non-monotonic timestamps: {p}")

        rows = {
            "pose": _count_rows(pose),
            "actions": _count_rows(act),
            "current": _count_rows(cur),
            "probe": _count_rows(probe),
        }
        if len(set(rows.values())) != 1:
            return ValidationResult(False, f"row count mismatch for agent_{i:03d}: {rows}")
        if expected_rows is None:
            expected_rows = rows["pose"]
        elif rows["pose"] != expected_rows:
            return ValidationResult(False, f"row count mismatch across agents: agent_{i:03d} has {rows['pose']} vs {expected_rows}")

    for rel in ("metrics.json", "metrics.csv"):
        if not (root / rel).exists():
            return ValidationResult(False, f"missing {rel}")

    return ValidationResult(True, "ok")

