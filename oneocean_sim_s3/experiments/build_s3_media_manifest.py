from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build consolidated S3 (A2) media manifest from a suite run")
    parser.add_argument("--suite-root", type=str, required=True, help="Path to runs/s3_3d_* suite folder")
    parser.add_argument("--output-json", type=str, default="project/media_manifest.json")
    parser.add_argument("--output-md", type=str, default="project/media_summary.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    suite_root = Path(args.suite_root).expanduser().resolve()
    suite_manifest_path = suite_root / "suite_manifest.json"
    if not suite_manifest_path.exists():
        raise FileNotFoundError(f"Missing suite manifest: {suite_manifest_path}")

    with suite_manifest_path.open("r", encoding="utf-8") as file:
        suite = json.load(file)
    cases = suite.get("cases", [])
    if not isinstance(cases, list) or not cases:
        raise ValueError("suite_manifest.json has no cases")

    media_rows: list[dict[str, object]] = []
    missing_media_cases: list[int] = []

    for case in cases:
        case_id = int(case["case_id"])
        task = str(case["task"])
        variant = str(case["variant"])
        tide_mode = str(case.get("tide_mode", "unknown"))
        output_dir = Path(str(case.get("output_dir", ""))).expanduser().resolve()

        manifest_path = output_dir / "media" / "media_manifest.json"
        if not manifest_path.exists():
            missing_media_cases.append(case_id)
            continue

        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        for item in manifest.get("media", []):
            media_rows.append(
                {
                    "case_id": case_id,
                    "task": task,
                    "variant": variant,
                    "tide_mode": tide_mode,
                    "episode": int(item.get("episode", 0)),
                    "scene_png": str(item.get("scene_png", "")),
                    "final_png": str(item.get("final_png", "")),
                    "rollout_gif": str(item.get("rollout_gif", "")),
                    "width": int(item.get("width", 0)),
                    "height": int(item.get("height", 0)),
                    "fps": int(item.get("fps", 0)),
                    "frame_count": int(item.get("frame_count", 0)),
                    "output_dir": str(output_dir),
                    "media_manifest": str(manifest_path),
                }
            )

    by_task: dict[str, int] = {}
    gif_count = 0
    for row in media_rows:
        by_task[str(row["task"])] = by_task.get(str(row["task"]), 0) + 1
        if str(row.get("rollout_gif", "")).endswith(".gif"):
            gif_count += 1

    output_json = Path(args.output_json).expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md = Path(args.output_md).expanduser().resolve()
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by": "oneocean_sim_s3.experiments.build_s3_media_manifest",
        "suite_root": str(suite_root),
        "suite_manifest": str(suite_manifest_path),
        "media_count": int(len(media_rows)),
        "gif_count": int(gif_count),
        "missing_media_case_ids": missing_media_cases,
        "task_counts": [{"task": k, "items": int(v)} for k, v in sorted(by_task.items())],
        "media": media_rows,
    }

    with output_json.open("w", encoding="utf-8") as file:
        json.dump(output_data, file, indent=2)

    with output_md.open("w", encoding="utf-8") as file:
        file.write("# S3 (A2) Media Summary\n\n")
        file.write(f"- Suite root: `{suite_root}`  \n")
        file.write(f"- Media entries: `{len(media_rows)}`  \n")
        file.write(f"- GIF count: `{gif_count}`  \n")
        file.write(f"- Missing media cases: `{len(missing_media_cases)}`\n\n")

        file.write("## By Task\n\n")
        file.write("| task | items |\n")
        file.write("|---|---:|\n")
        for item in output_data["task_counts"]:
            file.write(f"| {item['task']} | {item['items']} |\n")

        file.write("\n## Media List\n\n")
        file.write("| case_id | task | variant | tide_mode | episode | rollout_gif | scene_png | final_png |\n")
        file.write("|---:|---|---|---|---:|---|---|---|\n")
        for row in media_rows:
            file.write(
                f"| {row['case_id']} | {row['task']} | {row['variant']} | {row['tide_mode']} | {row['episode']} | "
                f"{row['rollout_gif']} | {row['scene_png']} | {row['final_png']} |\n"
            )

    print(
        json.dumps(
            {
                "media_manifest_json": str(output_json),
                "media_summary_md": str(output_md),
                "media_count": len(media_rows),
                "gif_count": gif_count,
                "missing_media_cases": len(missing_media_cases),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
