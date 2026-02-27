from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Task suite output root (contains results_manifest.json).")
    args = ap.parse_args()

    out_root = Path(args.out_dir).resolve()
    root_manifest = out_root / "results_manifest.json"
    if not root_manifest.exists():
        raise FileNotFoundError(root_manifest)

    data = json.loads(root_manifest.read_text(encoding="utf-8"))
    scenarios = data.get("scenarios", {})
    for scenario_name, per in scenarios.items():
        for per_task in per.get("episodes", []):
            task_name = per_task.get("task")
            if not task_name:
                continue
            task_dir = out_root / scenario_name.replace("/", "_") / task_name
            results_path = task_dir / "results_manifest.json"
            if not results_path.exists():
                continue
            r = json.loads(results_path.read_text(encoding="utf-8"))
            media: dict[str, str] = {}
            for i, ep in enumerate(r.get("episodes", [])):
                m = ep.get("media")
                if not isinstance(m, dict):
                    continue
                for k, v in m.items():
                    media[f"ep{i:03d}_{k}"] = str(v)
            (task_dir / "media_manifest.json").write_text(json.dumps(media, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[h3] wrote per-task media_manifest.json files under:", out_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

