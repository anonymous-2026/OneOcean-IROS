from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def _tag_now_local() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _ensure_ssl_cert_file() -> None:
    try:
        import certifi  # type: ignore

        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    except Exception:
        pass


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fp:
        r = csv.DictReader(fp)
        return [dict(row) for row in r]


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: set[str] = set()
    for row in rows:
        keys.update(row.keys())
    cols = sorted(keys)
    with path.open("w", encoding="utf-8", newline="") as fp:
        w = csv.DictWriter(fp, fieldnames=cols)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in cols})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="PierHarbor-HoveringCamera")
    ap.add_argument("--task", default="formation_transit_multiagent")
    ap.add_argument("--ns", nargs="*", type=int, default=[2, 4, 8, 10])
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--difficulty", default="easy", choices=("easy", "medium", "hard"))
    ap.add_argument("--current_npz", default=None)
    ap.add_argument("--dataset_days_per_sim_second", type=float, default=0.1)
    ap.add_argument("--pollution_model", default="ocpnet_3d", choices=("analytic", "ocpnet_3d"))
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--no_media", action="store_true", help="Disable MP4 generation in child suites (recommended for sweeps).")
    args = ap.parse_args()

    _ensure_ssl_cert_file()

    out_root = Path(args.out_dir) if args.out_dir else Path("runs") / "oceangym_h3" / f"scaling_sweep_{_tag_now_local()}"
    out_root = out_root.resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    suite_py = sys.executable
    suite_script = (Path(__file__).parent / "run_task_suite.py").resolve()
    env = dict(os.environ)

    children: list[dict[str, object]] = []
    all_rows: list[dict[str, str]] = []
    for n in list(args.ns):
        child_out = out_root / f"n{int(n):02d}"
        cmd = [
            suite_py,
            str(suite_script),
            "--scenarios",
            str(args.scenario),
            "--tasks",
            str(args.task),
            "--episodes",
            str(int(args.episodes)),
            "--difficulty",
            str(args.difficulty),
            "--n_multiagent",
            str(int(n)),
            "--pollution_model",
            str(args.pollution_model),
            "--dataset_days_per_sim_second",
            str(float(args.dataset_days_per_sim_second)),
            "--out_dir",
            str(child_out),
        ]
        if args.current_npz:
            cmd += ["--current_npz", str(args.current_npz)]
        if args.resume:
            cmd += ["--resume"]
        if args.no_media:
            cmd += ["--no_media"]

        subprocess.run(cmd, check=True, env=env)

        child_manifest = child_out / "results_manifest.json"
        child_summary = child_out / "summary.csv"
        children.append({"n_agents": int(n), "out_dir": str(child_out), "results_manifest": str(child_manifest), "summary_csv": str(child_summary)})

        if child_summary.exists():
            all_rows.extend(_read_csv(child_summary))

    if all_rows:
        _write_csv(out_root / "summary.csv", all_rows)

    root_manifest = {
        "track": "h3_oceangym",
        "type": "scaling_sweep",
        "script": str(Path(__file__).resolve()),
        "python": sys.executable,
        "command": list(sys.argv),
        "out_dir": str(out_root),
        "scenario": str(args.scenario),
        "task": str(args.task),
        "difficulty": str(args.difficulty),
        "episodes": int(args.episodes),
        "ns": [int(x) for x in list(args.ns)],
        "no_media": bool(args.no_media),
        "children": children,
    }
    (out_root / "results_manifest.json").write_text(json.dumps(root_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[h3] wrote:", out_root / "results_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
