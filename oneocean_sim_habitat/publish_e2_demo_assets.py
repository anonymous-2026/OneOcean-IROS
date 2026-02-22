from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil

MAP_REQUIRED_KEYS = {
    "seed",
    "cityBuildings",
    "mountainBuildings",
    "buildingColliders",
    "cabinPositions",
    "finalUsers",
    "terrainMap",
}

PATH_REQUIRED_KEYS = {
    "mapSeed",
    "waypoints",
    "userHoverMarkers",
    "experiments",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        payload = json.load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected object JSON at: {path}")
    return payload


def _validate_required_keys(payload: dict, required: set[str], label: str) -> None:
    missing = sorted(required - set(payload.keys()))
    if missing:
        raise ValueError(f"{label} missing required keys: {missing}")


def _default_target_dir() -> Path:
    project_root = Path(__file__).resolve().parents[2]
    candidate = project_root / "demo" / "assets" / "data"
    if candidate.exists():
        return candidate
    return Path.cwd() / "demo" / "assets" / "data"


def _copy_with_backup(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        backup_name = f"{destination.stem}.bak{destination.suffix}"
        backup_path = destination.with_name(backup_name)
        shutil.copy2(destination, backup_path)
    shutil.copy2(source, destination)


def _resolve_demo_export_paths(run_dir: Path) -> tuple[Path, Path]:
    run_dir = run_dir.resolve()
    demo_export_dir = run_dir / "demo_export"
    map_path = demo_export_dir / "drone_map_data.json"
    path_path = demo_export_dir / "drone_path_data.json"
    if map_path.exists() and path_path.exists():
        return map_path, path_path

    manifest_path = run_dir / "run_and_package_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)
        demo_export = manifest.get("demo_export", {})
        map_path = Path(demo_export.get("map_json", ""))
        path_path = Path(demo_export.get("path_json", ""))
        if map_path.exists() and path_path.exists():
            return map_path, path_path

    raise FileNotFoundError(
        f"Cannot resolve demo export JSON files under run dir: {run_dir}"
    )


def publish_e2_demo_assets(
    run_dir: str | Path,
    target_dir: str | Path | None = None,
    map_name: str = "ocean_map_data.json",
    path_name: str = "ocean_path_data.json",
    write_manifest: bool = True,
) -> dict[str, str]:
    run_path = Path(run_dir).expanduser().resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    source_map, source_path = _resolve_demo_export_paths(run_path)
    map_payload = _load_json(source_map)
    path_payload = _load_json(source_path)
    _validate_required_keys(map_payload, MAP_REQUIRED_KEYS, "map")
    _validate_required_keys(path_payload, PATH_REQUIRED_KEYS, "path")

    target = (
        Path(target_dir).expanduser().resolve()
        if target_dir is not None
        else _default_target_dir().resolve()
    )
    target.mkdir(parents=True, exist_ok=True)

    destination_map = target / map_name
    destination_path = target / path_name
    _copy_with_backup(source_map, destination_map)
    _copy_with_backup(source_path, destination_path)

    result = {
        "run_dir": str(run_path),
        "target_dir": str(target),
        "source_map_json": str(source_map.resolve()),
        "source_path_json": str(source_path.resolve()),
        "published_map_json": str(destination_map.resolve()),
        "published_path_json": str(destination_path.resolve()),
    }

    if write_manifest:
        manifest_path = target / "oneocean_e2_sync_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "synced_at": datetime.now().isoformat(timespec="seconds"),
                    **result,
                },
                file,
                indent=2,
            )
        result["sync_manifest_json"] = str(manifest_path.resolve())

    return result
