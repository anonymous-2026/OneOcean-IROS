from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


TaskKind = Literal[
    "go_to_goal_current",
    "station_keeping",
    "pollution_localization",
    "pollution_containment_multiagent",
]

DifficultyKind = Literal["easy", "medium", "hard"]


@dataclass(frozen=True)
class TaskConfig:
    kind: TaskKind
    difficulty: DifficultyKind = "medium"
    success_radius_m: float = 6.0
    max_steps: int = 240
    hold_steps: int = 30  # for station-keeping
    leakage_radius_m: float = 35.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskState:
    goal_xyz: np.ndarray
    hold_counter: int = 0


def preset_task(kind: TaskKind, difficulty: DifficultyKind) -> TaskConfig:
    d = str(difficulty)
    if kind == "go_to_goal_current":
        return TaskConfig(
            kind=kind,
            difficulty=difficulty,
            success_radius_m=10.0 if d == "easy" else 6.0 if d == "medium" else 3.5,
            max_steps=160 if d == "easy" else 240 if d == "medium" else 320,
        )
    if kind == "station_keeping":
        return TaskConfig(
            kind=kind,
            difficulty=difficulty,
            success_radius_m=8.0 if d == "easy" else 5.0 if d == "medium" else 3.0,
            max_steps=200 if d == "easy" else 260 if d == "medium" else 340,
            hold_steps=20 if d == "easy" else 40 if d == "medium" else 60,
        )
    if kind == "pollution_localization":
        return TaskConfig(
            kind=kind,
            difficulty=difficulty,
            success_radius_m=10.0 if d == "easy" else 6.0 if d == "medium" else 3.0,
            max_steps=240 if d == "easy" else 320 if d == "medium" else 420,
        )
    if kind == "pollution_containment_multiagent":
        return TaskConfig(
            kind=kind,
            difficulty=difficulty,
            success_radius_m=6.0,
            max_steps=220 if d == "easy" else 280 if d == "medium" else 360,
            leakage_radius_m=55.0 if d == "easy" else 40.0 if d == "medium" else 28.0,
        )
    raise ValueError(f"Unknown task kind: {kind}")


def reset_task(rng: np.random.Generator, bounds_xyz: tuple[np.ndarray, np.ndarray], cfg: TaskConfig) -> TaskState:
    lo, hi = bounds_xyz
    goal = rng.uniform(lo, hi).astype(np.float64)
    return TaskState(goal_xyz=goal, hold_counter=0)


def compute_success(
    cfg: TaskConfig,
    *,
    step_index: int,
    positions_xyz: np.ndarray,
    task_state: TaskState,
    pollution_source_xyz: np.ndarray | None,
    pollution_total_mass: float | None,
) -> tuple[bool, dict[str, Any]]:
    pos = np.asarray(positions_xyz, dtype=np.float64).reshape(-1, 3)
    n_agents = pos.shape[0]
    goal = np.asarray(task_state.goal_xyz, dtype=np.float64).reshape(3)

    if cfg.kind in ("go_to_goal_current", "station_keeping"):
        d = np.linalg.norm(pos - goal[None, :], axis=1)
        best = float(np.min(d))
        if cfg.kind == "go_to_goal_current":
            return (best <= float(cfg.success_radius_m)), {"best_dist_to_goal_m": best}

        # station keeping: must hold within radius for hold_steps.
        if best <= float(cfg.success_radius_m):
            task_state.hold_counter += 1
        else:
            task_state.hold_counter = 0
        return (task_state.hold_counter >= int(cfg.hold_steps)), {"best_dist_to_goal_m": best, "hold_counter": int(task_state.hold_counter)}

    if cfg.kind == "pollution_localization":
        if pollution_source_xyz is None:
            return False, {"source_error_m": None}
        src = np.asarray(pollution_source_xyz, dtype=np.float64).reshape(3)
        d = np.linalg.norm(pos - src[None, :], axis=1)
        best = float(np.min(d))
        return (best <= float(cfg.success_radius_m)), {"source_error_m": best}

    if cfg.kind == "pollution_containment_multiagent":
        # Success if mass reduced enough.
        if pollution_total_mass is None:
            return False, {"mass": None}
        # target mass reduction fraction scales with agent count
        target = 0.75 if n_agents <= 3 else 0.6 if n_agents <= 6 else 0.5
        # We interpret pollution_total_mass as fraction vs initial (the env tracks it).
        return (float(pollution_total_mass) <= float(target)), {"mass_frac": float(pollution_total_mass), "target": float(target)}

    raise ValueError(f"Unknown task kind: {cfg.kind}")
