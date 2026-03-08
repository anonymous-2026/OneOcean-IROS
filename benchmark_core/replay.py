from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .validators import validate_run_dir


@dataclass(frozen=True)
class ReplaySummary:
    run_dir: str
    ok: bool
    reason: str
    n_agents: int | None
    steps: int | None
    start_xyz: list[list[float]] | None
    end_xyz: list[list[float]] | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _read_pose_xyz(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as fp:
        r = csv.reader(fp)
        header = next(r, None)
        if header is None:
            return np.zeros((0, 3), dtype=np.float64)
        idx_x = header.index("x")
        idx_y = header.index("y")
        idx_z = header.index("z")
        rows = []
        for row in r:
            rows.append([float(row[idx_x]), float(row[idx_y]), float(row[idx_z])])
        return np.asarray(rows, dtype=np.float64)


def replay_run(run_dir: str | Path) -> ReplaySummary:
    root = Path(run_dir).expanduser().resolve()
    v = validate_run_dir(root)
    if not v.ok:
        return ReplaySummary(
            run_dir=str(root),
            ok=False,
            reason=v.reason,
            n_agents=None,
            steps=None,
            start_xyz=None,
            end_xyz=None,
        )

    meta = json.loads((root / "run_meta.json").read_text(encoding="utf-8"))
    n_agents = int(meta["n_agents"])
    starts: list[list[float]] = []
    ends: list[list[float]] = []
    steps = None
    for i in range(n_agents):
        pose_path = root / "agents" / f"agent_{i:03d}" / "pose_groundtruth" / "data.csv"
        xyz = _read_pose_xyz(pose_path)
        if xyz.shape[0] == 0:
            return ReplaySummary(
                run_dir=str(root),
                ok=False,
                reason=f"empty pose stream: agent_{i:03d}",
                n_agents=n_agents,
                steps=0,
                start_xyz=None,
                end_xyz=None,
            )
        if steps is None:
            steps = int(xyz.shape[0])
        starts.append([float(x) for x in xyz[0].tolist()])
        ends.append([float(x) for x in xyz[-1].tolist()])

    return ReplaySummary(
        run_dir=str(root),
        ok=True,
        reason="ok",
        n_agents=n_agents,
        steps=int(steps or 0),
        start_xyz=starts,
        end_xyz=ends,
    )

