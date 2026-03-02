from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


ControllerKind = Literal["go_to_goal", "station_keep", "plume_gradient", "containment_ring"]


@dataclass(frozen=True)
class ControllerConfig:
    kind: ControllerKind
    max_speed_mps: float = 1.2
    kp: float = 0.8
    ring_radius_m: float = 20.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def preset_controller(kind: ControllerKind, *, max_speed_mps: float) -> ControllerConfig:
    if kind == "go_to_goal":
        return ControllerConfig(kind=kind, max_speed_mps=float(max_speed_mps), kp=0.9)
    if kind == "station_keep":
        return ControllerConfig(kind=kind, max_speed_mps=float(max_speed_mps), kp=0.8)
    if kind == "plume_gradient":
        return ControllerConfig(kind=kind, max_speed_mps=float(max_speed_mps), kp=0.7, ring_radius_m=16.0)
    if kind == "containment_ring":
        return ControllerConfig(kind=kind, max_speed_mps=float(max_speed_mps), kp=0.7, ring_radius_m=22.0)
    raise ValueError(f"Unknown controller kind: {kind}")


def _clip_speed(v: np.ndarray, max_speed: float) -> np.ndarray:
    sp = float(np.linalg.norm(v))
    if sp <= max(1e-9, float(max_speed)):
        return v
    return v * (float(max_speed) / sp)


def compute_actions(
    cfg: ControllerConfig,
    *,
    step_index: int,
    positions_xyz: np.ndarray,
    goal_xyz: np.ndarray,
    pollution_probe: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    pos = np.asarray(positions_xyz, dtype=np.float64).reshape(-1, 3)
    n = pos.shape[0]
    goal_arr = np.asarray(goal_xyz, dtype=np.float64)
    if goal_arr.shape == (3,):
        goal = np.repeat(goal_arr.reshape(1, 3), n, axis=0)
        goal_center = goal_arr.reshape(3)
    elif goal_arr.shape == (n, 3):
        goal = goal_arr
        goal_center = np.mean(goal_arr, axis=0)
    else:
        raise ValueError(f"goal_xyz must be shape (3,) or (N,3); got {goal_arr.shape} (N={n})")
    probe = np.asarray(pollution_probe, dtype=np.float64).reshape(n)

    actions = np.zeros((n, 3), dtype=np.float64)
    if cfg.kind in ("go_to_goal", "station_keep"):
        for i in range(n):
            d = goal[i] - pos[i]
            actions[i] = _clip_speed(float(cfg.kp) * d, cfg.max_speed_mps)
        return actions

    if cfg.kind == "plume_gradient":
        # Estimate local gradient of log(concentration) from multi-agent probes, then ascend.
        leader = int(np.argmax(probe))
        p0 = pos[leader].copy()

        eps = 1e-12
        y = np.log(np.maximum(probe, eps))
        X = np.stack([pos[:, 0], pos[:, 2], np.ones((n,), dtype=np.float64)], axis=1)
        # Weight higher-probe agents more (better SNR).
        w = np.clip(probe / (float(np.max(probe)) + 1e-12), 0.05, 1.0)
        W = np.diag(w)
        try:
            beta, *_ = np.linalg.lstsq(W @ X, W @ y, rcond=None)
            grad_x = float(beta[0])
            grad_z = float(beta[1])
            g = np.array([grad_x, 0.0, grad_z], dtype=np.float64)
        except Exception:
            g = np.zeros((3,), dtype=np.float64)

        gn = float(np.linalg.norm(g))
        if not np.isfinite(gn) or gn < 1e-10:
            # Fallback exploration when probes are flat.
            ang = float(rng.uniform(0, 2 * math.pi))
            g = np.array([math.cos(ang), 0.0, math.sin(ang)], dtype=np.float64)
            gn = float(np.linalg.norm(g))

        g = g / max(1e-12, gn)
        actions[leader] = _clip_speed(float(cfg.max_speed_mps) * g, cfg.max_speed_mps)
        ring_r = 0.6 * float(cfg.ring_radius_m) / max(1.0, float(n))
        for i in range(n):
            if i == leader:
                continue
            a = 2.0 * math.pi * (i / max(1, n))
            target = p0 + np.array([ring_r * math.cos(a), 0.0, ring_r * math.sin(a)], dtype=np.float64)
            actions[i] = _clip_speed(float(cfg.kp) * (target - pos[i]), cfg.max_speed_mps)
        return actions

    if cfg.kind == "containment_ring":
        # Ring around goal (interpreted as plume center estimate).
        center = goal_center
        spin = 0.25 * float(step_index)
        for i in range(n):
            ang = 2.0 * math.pi * (i / max(1, n)) + spin
            target = center + np.array(
                [float(cfg.ring_radius_m) * math.cos(ang), 0.0, float(cfg.ring_radius_m) * math.sin(ang)],
                dtype=np.float64,
            )
            actions[i] = _clip_speed(float(cfg.kp) * (target - pos[i]), cfg.max_speed_mps)
        return actions

    raise ValueError(f"Unknown controller kind: {cfg.kind}")
