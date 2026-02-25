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


def _iter_rows(path: Path) -> tuple[list[str], Any]:
    fp = path.open("r", encoding="utf-8", newline="")
    r = csv.reader(fp)
    header = next(r, None)
    if header is None:
        fp.close()
        raise ValueError(f"empty csv: {path}")
    return header, (fp, r)


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
    env_cfg = meta.get("env_config", {}) if isinstance(meta, dict) else {}
    constraint_mode = str(env_cfg.get("constraint_mode", "hard"))
    bathy_mode = str(env_cfg.get("bathy_mode", "off"))
    land_thr = float(env_cfg.get("land_mask_threshold", 0.5))
    clearance = float(env_cfg.get("seafloor_clearance_m", 1.0))
    for i in range(n_agents):
        agent_dir = root / "agents" / f"agent_{i:03d}"
        pose = agent_dir / "pose_groundtruth" / "data.csv"
        act = agent_dir / "actions" / "data.csv"
        cur = agent_dir / "obs" / "local_current" / "data.csv"
        probe = agent_dir / "obs" / "pollution_probe" / "data.csv"
        latlon = agent_dir / "obs" / "latlon" / "data.csv"
        bathy = agent_dir / "obs" / "bathymetry" / "data.csv"
        for p in (pose, act, cur, probe, latlon, bathy):
            if not p.exists():
                return ValidationResult(False, f"missing stream: {p}")
            if not _check_monotonic_t(p, t_col=0):
                return ValidationResult(False, f"non-monotonic timestamps: {p}")

        rows = {
            "pose": _count_rows(pose),
            "actions": _count_rows(act),
            "current": _count_rows(cur),
            "probe": _count_rows(probe),
            "latlon": _count_rows(latlon),
            "bathymetry": _count_rows(bathy),
        }
        if len(set(rows.values())) != 1:
            return ValidationResult(False, f"row count mismatch for agent_{i:03d}: {rows}")
        if expected_rows is None:
            expected_rows = rows["pose"]
        elif rows["pose"] != expected_rows:
            return ValidationResult(False, f"row count mismatch across agents: agent_{i:03d} has {rows['pose']} vs {expected_rows}")

        # Optional physics-ish constraints (when enabled): ensure we never record invalid-region / touchdown samples.
        if constraint_mode != "off":
            p_fp = None
            b_fp = None
            try:
                # pose: t,x,y,z,...
                p_header, (p_fp, p_r) = _iter_rows(pose)
                b_header, (b_fp, b_r) = _iter_rows(bathy)
                # indices
                tpi = p_header.index("t") if "t" in p_header else 0
                ypi = p_header.index("y") if "y" in p_header else 2
                tbi = b_header.index("t") if "t" in b_header else 0
                ei = b_header.index("elevation") if "elevation" in b_header else 1
                mi = b_header.index("land_mask") if "land_mask" in b_header else 2

                j = 0
                for prow, brow in zip(p_r, b_r):
                    tp = float(prow[tpi])
                    tb = float(brow[tbi])
                    if abs(tp - tb) > 1e-6:
                        p_fp.close()
                        b_fp.close()
                        return ValidationResult(False, f"timestamp mismatch pose vs bathy: agent_{i:03d} row_{j}")

                    land = float(brow[mi])
                    if land == land and land >= (land_thr - 1e-12):  # not-NaN and violates
                        p_fp.close()
                        b_fp.close()
                        return ValidationResult(False, f"land_mask violates hard constraint: agent_{i:03d} row_{j} land_mask={land}")

                    if bathy_mode == "hard":
                        elev = float(brow[ei])
                        if not (elev == elev):  # NaN
                            p_fp.close()
                            b_fp.close()
                            return ValidationResult(False, f"missing elevation under bathy_mode=hard: agent_{i:03d} row_{j}")
                        if elev >= 0.0:
                            p_fp.close()
                            b_fp.close()
                            return ValidationResult(False, f"non-underwater elevation under bathy_mode=hard: agent_{i:03d} row_{j} elevation={elev}")
                        water_depth = -float(elev)
                        y_depth = float(prow[ypi])
                        if (y_depth + clearance) > (water_depth + 1e-6):
                            p_fp.close()
                            b_fp.close()
                            return ValidationResult(
                                False,
                                f"touchdown/too-shallow under bathy_mode=hard: agent_{i:03d} row_{j} y={y_depth} elev={elev} clearance={clearance}",
                            )
                    j += 1
            except Exception as e:
                return ValidationResult(False, f"constraint validation failed: agent_{i:03d}: {type(e).__name__}: {e}")
            finally:
                if p_fp is not None:
                    try:
                        p_fp.close()
                    except Exception:
                        pass
                if b_fp is not None:
                    try:
                        b_fp.close()
                    except Exception:
                        pass

    for rel in ("metrics.json", "metrics.csv"):
        if not (root / rel).exists():
            return ValidationResult(False, f"missing {rel}")

    return ValidationResult(True, "ok")
