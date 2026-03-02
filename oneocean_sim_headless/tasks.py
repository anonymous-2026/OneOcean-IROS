from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np


TaskKind = Literal[
    # Canonical 10-task list (project/h_track_requirements.md).
    "go_to_goal_current",
    "station_keeping",
    "surface_pollution_cleanup_multiagent",
    "underwater_pollution_lift_5uuv",
    "fish_herding_8uuv",
    "area_scan_terrain_recon",
    "pipeline_inspection_leak_detection",
    "route_following_waypoints",
    "depth_profile_tracking",
    "formation_transit_multiagent",
    # Legacy/internal ids (kept for backward-compat).
    "pollution_localization",
    "pollution_containment_multiagent",
]

DifficultyKind = Literal["easy", "medium", "hard"]

CANONICAL_TASKS_10: tuple[str, ...] = (
    "go_to_goal_current",
    "station_keeping",
    "surface_pollution_cleanup_multiagent",
    "underwater_pollution_lift_5uuv",
    "fish_herding_8uuv",
    "area_scan_terrain_recon",
    "pipeline_inspection_leak_detection",
    "route_following_waypoints",
    "depth_profile_tracking",
    "formation_transit_multiagent",
)


def required_n_agents(kind: TaskKind) -> int | None:
    if kind == "underwater_pollution_lift_5uuv":
        return 5
    if kind == "fish_herding_8uuv":
        return 8
    return None


@dataclass(frozen=True)
class TaskConfig:
    kind: TaskKind
    difficulty: DifficultyKind = "medium"
    success_radius_m: float = 6.0
    max_steps: int = 240

    # station keeping
    hold_steps: int = 30

    # legacy pollution containment (mass fraction threshold scales w/ N)
    leakage_radius_m: float = 35.0

    # waypoint-like tasks
    waypoints_n: int = 6

    # formation transit
    formation_radius_m: float = 20.0

    # scan
    scan_cell_size_m: float = 25.0
    scan_target_coverage: float = 0.8
    scan_radius_m: float = 28.0

    # surface cleanup
    cleanup_sources_n: int = 4
    cleanup_dwell_s: float = 3.5

    # pipeline inspection
    pipeline_leaks_n: int = 3
    pipeline_detect_radius_m: float = 10.0

    # fish herding
    fish_count: int = 60

    # lift
    lift_attach_radius_m: float = 6.0
    lift_hold_s: float = 2.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class TaskState:
    # controller uses this as the default "center" goal; env may override with per-agent goals
    goal_xyz: np.ndarray

    # station keeping
    hold_counter: int = 0

    # waypoint tasks
    waypoints_xyz: np.ndarray | None = None  # (K,3)
    waypoint_index: int = 0

    # formation
    formation_offsets_xyz: np.ndarray | None = None  # (N,3)

    # surface cleanup
    cleanup_sources_xyz: np.ndarray | None = None  # (S,3)
    cleanup_done: np.ndarray | None = None  # (S,) bool
    cleanup_progress_s: np.ndarray | None = None  # (S,) seconds near source
    cleanup_assigned_source: np.ndarray | None = None  # (N,) int source index

    # underwater lift
    lift_barrel_xyz: np.ndarray | None = None  # (3,)
    lift_phase: str = "approach"  # approach -> lift_off -> join5 -> to_surface
    lift_attached: np.ndarray | None = None  # (N,) bool
    lift_off_counter_s: float = 0.0

    # fish
    fish_xyz: np.ndarray | None = None  # (F,3)
    fish_stage: int = 0
    fish_init_dist_to_goal_xz_m: float | None = None

    # scan
    scan_visited: np.ndarray | None = None  # (H,W) bool
    scan_grid_origin_xz: np.ndarray | None = None  # (2,) origin (x0,z0)
    scan_grid_hw: tuple[int, int] | None = None

    # pipeline inspection
    pipeline_xyz: np.ndarray | None = None  # (P,3)
    leak_xyz: np.ndarray | None = None  # (L,3)
    leak_detected: np.ndarray | None = None  # (L,) bool
    leak_first_detect_t: np.ndarray | None = None  # (L,) seconds


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
    if kind == "route_following_waypoints":
        return TaskConfig(
            kind=kind,
            difficulty=difficulty,
            success_radius_m=10.0 if d == "easy" else 6.0 if d == "medium" else 3.5,
            max_steps=220 if d == "easy" else 320 if d == "medium" else 560,
            waypoints_n=5 if d == "easy" else 7 if d == "medium" else 9,
        )
    if kind == "depth_profile_tracking":
        return TaskConfig(
            kind=kind,
            difficulty=difficulty,
            success_radius_m=10.0 if d == "easy" else 7.0 if d == "medium" else 5.0,
            max_steps=240 if d == "easy" else 360 if d == "medium" else 520,
            waypoints_n=5 if d == "easy" else 7 if d == "medium" else 9,
        )
    if kind == "formation_transit_multiagent":
        return TaskConfig(
            kind=kind,
            difficulty=difficulty,
            success_radius_m=14.0 if d == "easy" else 10.0 if d == "medium" else 7.0,
            max_steps=260 if d == "easy" else 380 if d == "medium" else 520,
            formation_radius_m=16.0 if d == "easy" else 22.0 if d == "medium" else 28.0,
        )
    if kind == "surface_pollution_cleanup_multiagent":
        return TaskConfig(
            kind=kind,
            difficulty=difficulty,
            success_radius_m=8.0 if d == "easy" else 6.0 if d == "medium" else 4.0,
            max_steps=260 if d == "easy" else 380 if d == "medium" else 520,
            cleanup_sources_n=3 if d == "easy" else 4 if d == "medium" else 5,
            cleanup_dwell_s=3.0 if d == "easy" else 3.5 if d == "medium" else 4.0,
        )
    if kind == "underwater_pollution_lift_5uuv":
        return TaskConfig(
            kind=kind,
            difficulty=difficulty,
            success_radius_m=6.0,
            max_steps=360 if d == "easy" else 460 if d == "medium" else 620,
            lift_attach_radius_m=8.0 if d == "easy" else 7.0 if d == "medium" else 6.0,
            lift_hold_s=1.5 if d == "easy" else 2.0 if d == "medium" else 2.5,
        )
    if kind == "fish_herding_8uuv":
        return TaskConfig(
            kind=kind,
            difficulty=difficulty,
            success_radius_m=28.0 if d == "easy" else 24.0 if d == "medium" else 18.0,
            max_steps=360 if d == "easy" else 520 if d == "medium" else 760,
            fish_count=50 if d == "easy" else 70 if d == "medium" else 90,
        )
    if kind == "area_scan_terrain_recon":
        return TaskConfig(
            kind=kind,
            difficulty=difficulty,
            success_radius_m=8.0,
            # Needs to be large enough to traverse a lawnmower grid; keep generous for headless replay.
            max_steps=900 if d == "easy" else 1400 if d == "medium" else 2800,
            scan_cell_size_m=30.0 if d == "easy" else 22.0 if d == "medium" else 16.0,
            scan_target_coverage=0.65 if d == "easy" else 0.8 if d == "medium" else 0.90,
            scan_radius_m=35.0 if d == "easy" else 30.0 if d == "medium" else 24.0,
        )
    if kind == "pipeline_inspection_leak_detection":
        return TaskConfig(
            kind=kind,
            difficulty=difficulty,
            success_radius_m=9.0 if d == "easy" else 7.0 if d == "medium" else 5.0,
            max_steps=520 if d == "easy" else 880 if d == "medium" else 1400,
            pipeline_leaks_n=2 if d == "easy" else 3 if d == "medium" else 4,
            pipeline_detect_radius_m=16.0 if d == "easy" else 12.0 if d == "medium" else 9.0,
        )
    raise ValueError(f"Unknown task kind: {kind}")


def reset_task(rng: np.random.Generator, bounds_xyz: tuple[np.ndarray, np.ndarray], cfg: TaskConfig, *, n_agents: int) -> TaskState:
    lo, hi = bounds_xyz
    goal = rng.uniform(lo, hi).astype(np.float64)
    st = TaskState(goal_xyz=goal, hold_counter=0)

    # Waypoint-like tasks: generate a polyline within bounds.
    if cfg.kind in ("route_following_waypoints", "depth_profile_tracking", "pipeline_inspection_leak_detection"):
        k = int(max(2, cfg.waypoints_n))
        p0 = rng.uniform(lo, hi).astype(np.float64)
        p1 = rng.uniform(lo, hi).astype(np.float64)
        ts = np.linspace(0.0, 1.0, k, dtype=np.float64)
        wps = (1.0 - ts[:, None]) * p0[None, :] + ts[:, None] * p1[None, :]
        # Add small lateral wiggle in XZ, keep depth stable-ish.
        span = float(np.linalg.norm((hi - lo)[[0, 2]]))
        wig = rng.normal(scale=0.06 * span, size=(k, 2))
        wps[:, 0] = np.clip(wps[:, 0] + wig[:, 0], lo[0], hi[0])
        wps[:, 2] = np.clip(wps[:, 2] + wig[:, 1], lo[2], hi[2])
        wps[:, 1] = np.clip(wps[:, 1], lo[1], hi[1])
        st.waypoints_xyz = wps.astype(np.float64)
        st.waypoint_index = 0
        st.goal_xyz = wps[0].copy()
        if cfg.kind == "pipeline_inspection_leak_detection":
            st.pipeline_xyz = wps.astype(np.float64)
            l = int(max(1, cfg.pipeline_leaks_n))
            # Place leaks *on the pipeline* (on random waypoint segments) so a waypoint-following policy can detect them.
            leak = np.zeros((l, 3), dtype=np.float64)
            for li in range(l):
                # Prefer placing leaks close to waypoints to make detection robust under drift.
                if k >= 3:
                    wi = int(rng.integers(1, k - 1))
                    leak[li] = wps[wi]
                else:
                    seg = int(rng.integers(0, max(1, k - 1)))
                    a = wps[seg]
                    b = wps[min(k - 1, seg + 1)]
                    tt = float(rng.uniform(0.2, 0.8))
                    leak[li] = (1.0 - tt) * a + tt * b
            st.leak_xyz = leak.astype(np.float64)
            st.leak_detected = np.zeros((l,), dtype=bool)
            st.leak_first_detect_t = np.full((l,), np.nan, dtype=np.float64)

    if cfg.kind == "formation_transit_multiagent":
        st.formation_offsets_xyz = np.zeros((int(n_agents), 3), dtype=np.float64)
        r = float(cfg.formation_radius_m)
        for i in range(int(n_agents)):
            ang = 2.0 * np.pi * (i / max(1, int(n_agents)))
            st.formation_offsets_xyz[i] = np.array([r * float(np.cos(ang)), 0.0, r * float(np.sin(ang))], dtype=np.float64)

    if cfg.kind == "surface_pollution_cleanup_multiagent":
        s = int(max(1, cfg.cleanup_sources_n))
        sources = rng.uniform(lo, hi, size=(s, 3)).astype(np.float64)
        sources[:, 1] = lo[1] + 0.6  # near-surface
        st.cleanup_sources_xyz = sources
        st.cleanup_done = np.zeros((s,), dtype=bool)
        st.cleanup_progress_s = np.zeros((s,), dtype=np.float64)
        st.cleanup_assigned_source = np.full((int(n_agents),), -1, dtype=np.int64)
        st.goal_xyz = sources[0].copy()

    if cfg.kind == "underwater_pollution_lift_5uuv":
        barrel = rng.uniform(lo, hi).astype(np.float64)
        barrel[1] = min(hi[1] - 0.5, lo[1] + 0.70 * float(hi[1] - lo[1]))  # deep-ish but reachable
        st.lift_barrel_xyz = barrel
        st.lift_phase = "approach"
        st.lift_attached = np.zeros((int(n_agents),), dtype=bool)
        st.lift_off_counter_s = 0.0
        st.goal_xyz = barrel.copy()

    if cfg.kind == "fish_herding_8uuv":
        f = int(max(1, cfg.fish_count))
        fish = rng.uniform(lo, hi, size=(f, 3)).astype(np.float64)
        # Spawn near one corner (shore-ish), shallow.
        fish[:, 0] = np.clip(lo[0] + 0.2 * float(hi[0] - lo[0]) + rng.normal(scale=8.0, size=(f,)), lo[0], hi[0])
        fish[:, 2] = np.clip(lo[2] + 0.2 * float(hi[2] - lo[2]) + rng.normal(scale=8.0, size=(f,)), lo[2], hi[2])
        fish[:, 1] = lo[1] + 0.8
        st.fish_xyz = fish
        st.fish_stage = 0

    if cfg.kind == "area_scan_terrain_recon":
        width = float(hi[0] - lo[0])
        depth = float(hi[2] - lo[2])
        cell = float(max(4.0, cfg.scan_cell_size_m))
        w = int(max(3, math.ceil(width / cell)))
        h = int(max(3, math.ceil(depth / cell)))
        st.scan_grid_hw = (h, w)
        st.scan_grid_origin_xz = np.array([lo[0], lo[2]], dtype=np.float64)
        st.scan_visited = np.zeros((h, w), dtype=bool)

    return st


def compute_success(
    cfg: TaskConfig,
    *,
    step_index: int,
    dt_s: float,
    positions_xyz: np.ndarray,
    task_state: TaskState,
    pollution_source_xyz: np.ndarray | None,
    pollution_total_mass: float | None,
) -> tuple[bool, dict[str, Any]]:
    pos = np.asarray(positions_xyz, dtype=np.float64).reshape(-1, 3)
    n_agents = int(pos.shape[0])
    goal = np.asarray(task_state.goal_xyz, dtype=np.float64).reshape(3)
    dt_s = float(dt_s)

    if cfg.kind in ("go_to_goal_current", "station_keeping"):
        d = np.linalg.norm(pos - goal[None, :], axis=1)
        best = float(np.min(d))
        if cfg.kind == "go_to_goal_current":
            return (best <= float(cfg.success_radius_m)), {"best_dist_to_goal_m": best}

        if best <= float(cfg.success_radius_m):
            task_state.hold_counter += 1
        else:
            task_state.hold_counter = 0
        return (
            task_state.hold_counter >= int(cfg.hold_steps),
            {"best_dist_to_goal_m": best, "hold_counter": int(task_state.hold_counter)},
        )

    if cfg.kind == "pollution_localization":
        if pollution_source_xyz is None:
            return False, {"source_error_m": None}
        src = np.asarray(pollution_source_xyz, dtype=np.float64).reshape(3)
        d = np.linalg.norm(pos - src[None, :], axis=1)
        best = float(np.min(d))
        return (best <= float(cfg.success_radius_m)), {"source_error_m": best}

    if cfg.kind == "pollution_containment_multiagent":
        if pollution_total_mass is None:
            return False, {"mass": None}
        target = 0.75 if n_agents <= 3 else 0.6 if n_agents <= 6 else 0.5
        return (float(pollution_total_mass) <= float(target)), {"mass_frac": float(pollution_total_mass), "target": float(target)}

    if cfg.kind in ("route_following_waypoints", "depth_profile_tracking"):
        if task_state.waypoints_xyz is None:
            return False, {"waypoint_index": None}
        wps = np.asarray(task_state.waypoints_xyz, dtype=np.float64)
        i = int(np.clip(int(task_state.waypoint_index), 0, wps.shape[0] - 1))
        wp = wps[i]
        d = np.linalg.norm(pos - wp[None, :], axis=1)
        best = float(np.min(d))
        if best <= float(cfg.success_radius_m) and i < (wps.shape[0] - 1):
            task_state.waypoint_index = i + 1
            task_state.goal_xyz = wps[task_state.waypoint_index].copy()
        done = task_state.waypoint_index >= (wps.shape[0] - 1) and best <= float(cfg.success_radius_m)
        out: dict[str, Any] = {"waypoint_index": int(task_state.waypoint_index), "best_dist_to_waypoint_m": best, "waypoints_total": int(wps.shape[0])}
        if cfg.kind == "depth_profile_tracking":
            out["mean_depth_abs_error_m"] = float(np.mean(np.abs(pos[:, 1] - float(wp[1]))))
        return done, out

    if cfg.kind == "formation_transit_multiagent":
        if task_state.formation_offsets_xyz is None:
            return False, {"formation_err_m": None}
        offsets = np.asarray(task_state.formation_offsets_xyz, dtype=np.float64).reshape(n_agents, 3)
        goals = goal[None, :] + offsets
        d = np.linalg.norm(pos - goals, axis=1)
        return bool(np.all(d <= float(cfg.success_radius_m))), {"formation_err_m": float(np.mean(d)), "formation_max_err_m": float(np.max(d))}

    if cfg.kind == "surface_pollution_cleanup_multiagent":
        if task_state.cleanup_sources_xyz is None or task_state.cleanup_done is None or task_state.cleanup_progress_s is None:
            return False, {"sources_done": None}
        srcs = np.asarray(task_state.cleanup_sources_xyz, dtype=np.float64)
        done = np.asarray(task_state.cleanup_done, dtype=bool)
        prog = np.asarray(task_state.cleanup_progress_s, dtype=np.float64)

        # Assign each agent to a nearest unfinished source.
        if task_state.cleanup_assigned_source is None:
            task_state.cleanup_assigned_source = np.full((n_agents,), -1, dtype=np.int64)
        for ai in range(n_agents):
            cur = int(task_state.cleanup_assigned_source[ai])
            if cur >= 0 and cur < int(done.size) and not bool(done[cur]):
                continue
            if np.all(done):
                task_state.cleanup_assigned_source[ai] = -1
                continue
            cand = np.where(~done)[0]
            dists = np.linalg.norm(srcs[cand] - pos[ai][None, :], axis=1)
            task_state.cleanup_assigned_source[ai] = int(cand[int(np.argmin(dists))])

        radius = float(cfg.success_radius_m)
        for si in range(int(srcs.shape[0])):
            if bool(done[si]):
                continue
            near = False
            for ai in range(n_agents):
                if int(task_state.cleanup_assigned_source[ai]) != int(si):
                    continue
                if float(np.linalg.norm(pos[ai] - srcs[si])) <= radius:
                    near = True
                    break
            if near:
                prog[si] += dt_s
            if float(prog[si]) >= float(cfg.cleanup_dwell_s):
                done[si] = True

        task_state.cleanup_done = done
        task_state.cleanup_progress_s = prog
        sources_done = int(np.count_nonzero(done))
        return bool(sources_done == int(done.size)), {
            "sources_done": sources_done,
            "sources_total": int(done.size),
            "mean_source_progress_s": float(np.mean(prog)),
        }

    if cfg.kind == "underwater_pollution_lift_5uuv":
        if task_state.lift_barrel_xyz is None or task_state.lift_attached is None:
            return False, {"lift_phase": None}
        barrel = np.asarray(task_state.lift_barrel_xyz, dtype=np.float64).reshape(3)
        attached = np.asarray(task_state.lift_attached, dtype=bool).reshape(n_agents)
        phase = str(task_state.lift_phase)

        r = float(cfg.lift_attach_radius_m)
        for i in range(n_agents):
            if attached[i]:
                continue
            if float(np.linalg.norm(pos[i] - barrel)) <= r:
                attached[i] = True

        if phase == "approach" and int(np.count_nonzero(attached[:4])) >= 4:
            phase = "lift_off"
        if phase == "lift_off":
            task_state.lift_off_counter_s += dt_s
            if float(task_state.lift_off_counter_s) >= float(cfg.lift_hold_s):
                phase = "join5"
        if phase == "join5":
            if n_agents >= 5 and bool(attached[4]):
                phase = "to_surface"

        task_state.lift_attached = attached
        task_state.lift_phase = phase
        # "surface" proxy: small depth
        success = phase == "to_surface" and float(barrel[1]) <= 3.0
        return success, {"lift_phase": phase, "lift_attached_count": int(np.count_nonzero(attached))}

    if cfg.kind == "fish_herding_8uuv":
        if task_state.fish_xyz is None:
            return False, {"fish_progress": None}
        fish = np.asarray(task_state.fish_xyz, dtype=np.float64).reshape(-1, 3)
        centroid = np.mean(fish, axis=0)
        dist = float(np.linalg.norm(centroid[[0, 2]] - goal[[0, 2]]))
        init = float(task_state.fish_init_dist_to_goal_xz_m or float("nan"))
        if not np.isfinite(init) or init <= 1e-6:
            init = max(1e-6, float(dist))
            task_state.fish_init_dist_to_goal_xz_m = init
        prog = float(np.clip(1.0 - dist / init, 0.0, 1.0))
        stage = int(np.clip(int(math.floor(prog * 4.0)), 0, 3))
        task_state.fish_stage = stage
        return dist <= float(cfg.success_radius_m), {
            "fish_progress": prog,
            "fish_stage": int(stage),
            "fish_dist_to_goal_xz_m": dist,
            "fish_init_dist_to_goal_xz_m": float(init),
        }

    if cfg.kind == "area_scan_terrain_recon":
        if task_state.scan_visited is None or task_state.scan_grid_origin_xz is None or task_state.scan_grid_hw is None:
            return False, {"coverage": None}
        visited = np.asarray(task_state.scan_visited, dtype=bool)
        ox, oz = [float(x) for x in np.asarray(task_state.scan_grid_origin_xz, dtype=np.float64).reshape(2)]
        h, w = task_state.scan_grid_hw
        cell = float(cfg.scan_cell_size_m)
        # A scan covers an area around each vehicle (proxy for a sonar/FOV footprint).
        rr = int(max(0, math.ceil(float(cfg.scan_radius_m) / max(1e-9, cell))))
        for ai in range(n_agents):
            x = float(pos[ai, 0])
            z = float(pos[ai, 2])
            j0 = int(np.clip(math.floor((x - ox) / cell), 0, w - 1))
            i0 = int(np.clip(math.floor((z - oz) / cell), 0, h - 1))
            i_lo = max(0, i0 - rr)
            i_hi = min(h, i0 + rr + 1)
            j_lo = max(0, j0 - rr)
            j_hi = min(w, j0 + rr + 1)
            visited[i_lo:i_hi, j_lo:j_hi] = True
        task_state.scan_visited = visited
        coverage = float(np.count_nonzero(visited) / float(visited.size))
        return coverage >= float(cfg.scan_target_coverage), {
            "coverage": coverage,
            "cells_visited": int(np.count_nonzero(visited)),
            "cells_total": int(visited.size),
        }

    if cfg.kind == "pipeline_inspection_leak_detection":
        if task_state.leak_xyz is None or task_state.leak_detected is None or task_state.leak_first_detect_t is None:
            return False, {"leaks_detected": None}
        leak = np.asarray(task_state.leak_xyz, dtype=np.float64).reshape(-1, 3)
        det = np.asarray(task_state.leak_detected, dtype=bool).reshape(-1)
        first_t = np.asarray(task_state.leak_first_detect_t, dtype=np.float64).reshape(-1)

        # Navigation: follow the pipeline polyline (waypoints) if available.
        wp_info: dict[str, Any] = {}
        if task_state.waypoints_xyz is not None:
            wps = np.asarray(task_state.waypoints_xyz, dtype=np.float64)
            i = int(np.clip(int(task_state.waypoint_index), 0, wps.shape[0] - 1))
            wp = wps[i]
            d_wp = np.linalg.norm(pos - wp[None, :], axis=1)
            best_wp = float(np.min(d_wp))
            if best_wp <= float(cfg.success_radius_m) and i < (wps.shape[0] - 1):
                task_state.waypoint_index = i + 1
                task_state.goal_xyz = wps[task_state.waypoint_index].copy()
            wp_info = {
                "waypoint_index": int(task_state.waypoint_index),
                "waypoints_total": int(wps.shape[0]),
                "best_dist_to_waypoint_m": best_wp,
            }

        r = float(cfg.pipeline_detect_radius_m)
        for li in range(int(leak.shape[0])):
            if bool(det[li]):
                continue
            d = np.linalg.norm(pos - leak[li][None, :], axis=1)
            if float(np.min(d)) <= r:
                det[li] = True
                first_t[li] = float(step_index) * dt_s
        task_state.leak_detected = det
        task_state.leak_first_detect_t = first_t
        count = int(np.count_nonzero(det))
        t_first = float(np.nanmin(first_t)) if np.any(np.isfinite(first_t)) else None
        return count == int(det.size), {"leaks_detected": count, "leaks_total": int(det.size), "time_to_first_detection_s": t_first, **wp_info}

    raise ValueError(f"Unknown task kind: {cfg.kind}")
