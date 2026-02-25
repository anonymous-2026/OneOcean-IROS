from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


@dataclass(frozen=True)
class SettingsSpec:
    vehicle_count: int = 1
    base_name: str = "Rov"
    vehicle_type: str = "RovSimple"
    pawn_path: str = "DefaultRov"
    spacing_m: float = 2.0
    start_z: float = -2.0
    start_x: float = 0.0
    start_y: float = 0.0
    camera_name: str = "front_right_custom"
    camera_width: int = 1280
    camera_height: int = 720
    camera_fov_deg: float = 90.0
    physics_engine_name: Optional[str] = "ExternalPhysicsEngine"


def build_settings(spec: SettingsSpec) -> Dict:
    vehicles = {}
    for idx in range(spec.vehicle_count):
        name = f"{spec.base_name}{idx:02d}"
        vehicles[name] = {
            "VehicleType": spec.vehicle_type,
            "DefaultVehicleState": "Armed",
            "PawnPath": spec.pawn_path,
            "EnableCollisions": True,
            "AllowAPIAlways": True,
            "X": spec.start_x + idx * spec.spacing_m,
            "Y": spec.start_y,
            "Z": spec.start_z,
            "Cameras": {
                spec.camera_name: {
                    "CaptureSettings": [
                        {
                            "ImageType": 0,
                            "FOV_Degrees": spec.camera_fov_deg,
                            "Width": spec.camera_width,
                            "Height": spec.camera_height,
                        }
                    ],
                    "X": 0.50,
                    "Y": 0.06,
                    "Z": 0.10,
                    "Pitch": 0.0,
                    "Roll": 0.0,
                    "Yaw": 0.0,
                }
            },
        }

    settings = {
        "SeeDocsAt": "https://github.com/Microsoft/AirSim/blob/master/docs/settings.md",
        "SettingsVersion": 1.2,
        "SimMode": "Rov",
        "ClockSpeed": 1,
        "PawnPaths": {
            "DefaultQuadrotor": {"PawnBP": "Class'/AirSim/Blueprints/BP_FlyingPawn.BP_FlyingPawn_C'"},
            "DefaultRov": {"PawnBP": "Class'/AirSim/Blueprints/BP_RovPawn.BP_RovPawn_C'"},
            "DefaultComputerVision": {"PawnBP": "Class'/AirSim/Blueprints/BP_ComputerVisionPawn.BP_ComputerVisionPawn_C'"},
        },
        "Vehicles": vehicles,
    }
    if spec.physics_engine_name:
        settings["PhysicsEngineName"] = spec.physics_engine_name
    return settings


def write_settings_json(path: str | Path, settings: Dict) -> Path:
    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    return out_path

