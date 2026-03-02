from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RecorderConfig:
    write_csv: bool = True


class _CsvStream:
    def __init__(self, path: Path, header: list[str]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._fp = path.open("w", encoding="utf-8", newline="")
        self._writer = csv.writer(self._fp)
        self._writer.writerow(header)
        self._rows = 0

    def write_row(self, row: list[Any]) -> None:
        self._writer.writerow(row)
        self._rows += 1

    @property
    def rows(self) -> int:
        return self._rows

    def close(self) -> None:
        try:
            self._fp.flush()
        finally:
            self._fp.close()


class HeadlessRecorder:
    def __init__(self, out_dir: str | Path, *, n_agents: int, config: RecorderConfig | None = None) -> None:
        self.root = Path(out_dir).expanduser().resolve()
        self.n_agents = int(n_agents)
        self.cfg = config or RecorderConfig()

        self._pose: list[_CsvStream] = []
        self._actions: list[_CsvStream] = []
        self._current: list[_CsvStream] = []
        self._probe: list[_CsvStream] = []
        self._latlon: list[_CsvStream] = []
        self._bathy: list[_CsvStream] = []
        self._run_meta: dict[str, Any] = {}
        self._semantics_path = self.root / "environment_samples" / "semantics.jsonl"

        for i in range(self.n_agents):
            agent_dir = self.root / "agents" / f"agent_{i:03d}"
            self._pose.append(
                _CsvStream(
                    agent_dir / "pose_groundtruth" / "data.csv",
                    header=["t", "x", "y", "z", "qx", "qy", "qz", "qw"],
                )
            )
            self._actions.append(_CsvStream(agent_dir / "actions" / "data.csv", header=["t", "ax", "ay", "az"]))
            self._current.append(_CsvStream(agent_dir / "obs" / "local_current" / "data.csv", header=["t", "cx", "cy", "cz"]))
            self._probe.append(_CsvStream(agent_dir / "obs" / "pollution_probe" / "data.csv", header=["t", "concentration"]))
            self._latlon.append(_CsvStream(agent_dir / "obs" / "latlon" / "data.csv", header=["t", "lat", "lon"]))
            self._bathy.append(_CsvStream(agent_dir / "obs" / "bathymetry" / "data.csv", header=["t", "elevation", "land_mask"]))

        (self.root / "environment_samples").mkdir(parents=True, exist_ok=True)

    def write_run_meta(self, payload: dict[str, Any]) -> None:
        self._run_meta = payload
        (self.root / "run_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def write_metrics(self, metrics: dict[str, Any]) -> None:
        (self.root / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        # Flatten to a single-row CSV for quick aggregation.
        keys = sorted(metrics.keys())
        with (self.root / "metrics.csv").open("w", encoding="utf-8", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(keys)
            w.writerow([metrics.get(k) for k in keys])

    def write_environment_sample(self, t: float, *, dataset_time_index: int | None, dataset_depth_index: int | None) -> None:
        path = self.root / "environment_samples" / "global_time_index.csv"
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as fp:
            w = csv.writer(fp)
            if not exists:
                w.writerow(["t", "time_index", "depth_index"])
            w.writerow([float(t), dataset_time_index, dataset_depth_index])

    def write_semantics(self, t: float, payload: dict[str, Any]) -> None:
        """Write per-step semantic objects/events in an append-only JSONL stream.

        This is optional (not all tasks populate it), but it enables replay export and auditing
        without inflating per-agent CSV streams.
        """
        entry = {"t": float(t), **payload}
        with self._semantics_path.open("a", encoding="utf-8") as fp:
            fp.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def step(
        self,
        t: float,
        *,
        positions_xyz: np.ndarray,
        yaws_rad: np.ndarray,
        actions_xyz: np.ndarray,
        currents_xyz: np.ndarray,
        pollution_probe: np.ndarray,
        latitude: np.ndarray,
        longitude: np.ndarray,
        elevation: np.ndarray,
        land_mask: np.ndarray,
    ) -> None:
        pos = np.asarray(positions_xyz, dtype=np.float64).reshape(self.n_agents, 3)
        yaw = np.asarray(yaws_rad, dtype=np.float64).reshape(self.n_agents)
        act = np.asarray(actions_xyz, dtype=np.float64).reshape(self.n_agents, 3)
        cur = np.asarray(currents_xyz, dtype=np.float64).reshape(self.n_agents, 3)
        probe = np.asarray(pollution_probe, dtype=np.float64).reshape(self.n_agents)
        lat = np.asarray(latitude, dtype=np.float64).reshape(self.n_agents)
        lon = np.asarray(longitude, dtype=np.float64).reshape(self.n_agents)
        elev = np.asarray(elevation, dtype=np.float64).reshape(self.n_agents)
        mask = np.asarray(land_mask, dtype=np.float64).reshape(self.n_agents)

        for i in range(self.n_agents):
            qy = float(np.sin(0.5 * yaw[i]))
            qw = float(np.cos(0.5 * yaw[i]))
            self._pose[i].write_row([float(t), float(pos[i, 0]), float(pos[i, 1]), float(pos[i, 2]), 0.0, qy, 0.0, qw])
            self._actions[i].write_row([float(t), float(act[i, 0]), float(act[i, 1]), float(act[i, 2])])
            self._current[i].write_row([float(t), float(cur[i, 0]), float(cur[i, 1]), float(cur[i, 2])])
            self._probe[i].write_row([float(t), float(probe[i])])
            self._latlon[i].write_row([float(t), float(lat[i]), float(lon[i])])
            self._bathy[i].write_row([float(t), float(elev[i]), float(mask[i])])

    def close(self) -> None:
        for streams in (self._pose, self._actions, self._current, self._probe, self._latlon, self._bathy):
            for s in streams:
                s.close()


def required_streams_exist(run_dir: str | Path, *, n_agents: int) -> bool:
    root = Path(run_dir).expanduser().resolve()
    for i in range(int(n_agents)):
        agent_dir = root / "agents" / f"agent_{i:03d}"
        for rel in (
            "pose_groundtruth/data.csv",
            "actions/data.csv",
            "obs/local_current/data.csv",
            "obs/pollution_probe/data.csv",
            "obs/latlon/data.csv",
            "obs/bathymetry/data.csv",
        ):
            if not (agent_dir / rel).exists():
                return False
    for rel in ("run_meta.json", "metrics.json", "metrics.csv"):
        if not (root / rel).exists():
            return False
    return True
