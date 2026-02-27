from __future__ import annotations

OCEAN_WORLD_CAMERA_SCENARIOS: list[str] = [
    "Dam-HoveringCamera",
    "PierHarbor-HoveringCamera",
    "OpenWater-HoveringCamera",
    "SimpleUnderwater-Hovering",
]


def scenario_preset(name: str) -> list[str]:
    name = name.strip()
    if name in {"ocean_worlds_camera", "ocean_worlds"}:
        return list(OCEAN_WORLD_CAMERA_SCENARIOS)
    raise ValueError(f"Unknown preset: {name!r}")

