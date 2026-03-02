from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Dataset:
    x: np.ndarray  # (M, D)
    y: np.ndarray  # (M, 3)
    meta: dict[str, Any]


def _read_goal_semantics(path: Path) -> dict[float, Any]:
    out: dict[float, Any] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        if "t" not in obj:
            continue
        t = float(obj["t"])
        goal = obj.get("goal_for_action_xyz", None)
        if goal is None:
            continue
        out[t] = goal
    return out


def _read_rows(path: Path) -> Iterable[list[str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        r = csv.reader(fp)
        header = next(r, None)
        if header is None:
            return
        for row in r:
            yield row


def _onehot(task: str, vocab: list[str]) -> np.ndarray:
    v = np.zeros((len(vocab),), dtype=np.float32)
    try:
        v[vocab.index(task)] = 1.0
    except ValueError:
        pass
    return v


def build_bc_dataset(
    run_dirs: list[Path],
    *,
    task_vocab: list[str],
    semantics_stride_s: float = 5.0,
    max_samples: int = 0,
) -> Dataset:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    used_runs: list[str] = []
    used_rows = 0

    for run_dir in run_dirs:
        run_dir = run_dir.expanduser().resolve()
        meta_path = run_dir / "run_meta.json"
        if not meta_path.exists():
            continue
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        dt_s = float((meta.get("env_config") or {}).get("dt_s", 1.0))
        n_agents = int(meta.get("n_agents", 0))
        task = str((meta.get("task") or {}).get("kind") or meta.get("task", {}).get("kind") or meta.get("task_kind") or meta.get("task", {}).get("kind") or meta.get("task", {}).get("kind") or meta.get("task", {}).get("kind") or "")
        # In our run_meta.json layout, task kind is stored at meta["task"]["kind"].
        if isinstance(meta.get("task"), dict):
            task = str(meta["task"].get("kind", task))
        task_oh = _onehot(task, task_vocab)

        sem_path = run_dir / "environment_samples" / "semantics.jsonl"
        goals_by_t = _read_goal_semantics(sem_path)
        if not goals_by_t:
            continue

        # Build per-agent iterators.
        for ai in range(n_agents):
            agent_dir = run_dir / "agents" / f"agent_{ai:03d}"
            pose_p = agent_dir / "pose_groundtruth" / "data.csv"
            act_p = agent_dir / "actions" / "data.csv"
            cur_p = agent_dir / "obs" / "local_current" / "data.csv"
            probe_p = agent_dir / "obs" / "pollution_probe" / "data.csv"
            if not (pose_p.exists() and act_p.exists() and cur_p.exists() and probe_p.exists()):
                continue

            pose_it = _read_rows(pose_p)
            act_it = _read_rows(act_p)
            cur_it = _read_rows(cur_p)
            probe_it = _read_rows(probe_p)
            for prow, arow, crow, rrow in zip(pose_it, act_it, cur_it, probe_it):
                # t alignment is validated elsewhere; use pose t.
                t = float(prow[0])
                # Only train on downsampled steps where goal is recorded.
                if t not in goals_by_t:
                    continue
                goal_any = goals_by_t[t]
                if isinstance(goal_any, list) and goal_any and isinstance(goal_any[0], list):
                    # per-agent
                    if ai >= len(goal_any):
                        continue
                    gx, gy, gz = [float(x) for x in goal_any[ai]]
                else:
                    gx, gy, gz = [float(x) for x in goal_any]

                # Recorder stores pose AFTER applying (action + current) for this step, but timestamps it at t.
                # For BC we want the PRE-step pose that produced the logged action.
                x_post, y_post, z_post = float(prow[1]), float(prow[2]), float(prow[3])
                ax, ay, az = float(arow[1]), float(arow[2]), float(arow[3])
                cx, cy, cz = float(crow[1]), float(crow[2]), float(crow[3])
                x = x_post - (ax + cx) * dt_s
                y = y_post - (ay + cy) * dt_s
                z = z_post - (az + cz) * dt_s
                probe = float(rrow[1])

                dx, dy, dz = (gx - x), (gy - y), (gz - z)
                feat = np.concatenate(
                    [
                        np.array([dx, dy, dz, y, probe], dtype=np.float32),
                        task_oh.astype(np.float32),
                    ],
                    axis=0,
                )
                xs.append(feat)
                ys.append(np.array([ax, ay, az], dtype=np.float32))
                used_rows += 1
                if max_samples and used_rows >= int(max_samples):
                    break
            if max_samples and used_rows >= int(max_samples):
                break

        used_runs.append(str(run_dir))
        if max_samples and used_rows >= int(max_samples):
            break

    x = np.stack(xs, axis=0) if xs else np.zeros((0, 5 + len(task_vocab)), dtype=np.float32)
    y = np.stack(ys, axis=0) if ys else np.zeros((0, 3), dtype=np.float32)
    meta_out = {
        "schema": "bc_dataset_v1",
        "run_dirs": used_runs,
        "task_vocab": task_vocab,
        "n_samples": int(x.shape[0]),
        "note": "Features=[goal_delta(3) from PRE-step pose, depth_y(1), probe(1), task_onehot]; Targets=[action_xyz].",
        "semantics_goal_source": "environment_samples/semantics.jsonl:goal_for_action_xyz",
        "max_samples": int(max_samples),
    }
    return Dataset(x=x, y=y, meta=meta_out)


def save_dataset(ds: Dataset, *, out_npz: Path, out_meta_json: Path) -> None:
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, x=ds.x, y=ds.y)
    out_meta_json.parent.mkdir(parents=True, exist_ok=True)
    out_meta_json.write_text(json.dumps(ds.meta, indent=2), encoding="utf-8")
