from __future__ import annotations

import copy
from dataclasses import dataclass


@dataclass(frozen=True)
class HoloCfg:
    package_name: str = "Ocean"
    ticks_per_sec: int = 20
    fps: int = 20
    window_width: int = 1280
    window_height: int = 720
    render_quality: int = 3
    show_viewport: bool = False
    camera_width: int = 768
    camera_height: int = 768


def patch_scenario_for_recording(base: dict, cfg: HoloCfg, *, add_viewport_capture: bool) -> dict:
    scenario = copy.deepcopy(base)
    scenario["package_name"] = cfg.package_name
    scenario["ticks_per_sec"] = int(cfg.ticks_per_sec)
    scenario["frames_per_sec"] = int(cfg.fps)
    scenario["window_width"] = int(cfg.window_width)
    scenario["window_height"] = int(cfg.window_height)

    agents = scenario.get("agents", [])
    if not agents:
        raise ValueError("Scenario has no agents.")
    main = agents[0]

    sensors = list(main.get("sensors", []))
    if not sensors:
        raise ValueError("Scenario agent has no sensors.")

    def _clamp_hz(s: dict) -> None:
        hz = int(s.get("Hz", cfg.ticks_per_sec))
        s["Hz"] = min(hz, cfg.ticks_per_sec)
        if s.get("sensor_type") == "RGBCamera":
            s["Hz"] = int(cfg.fps)

    for s in sensors:
        _clamp_hz(s)

    # Some scenarios (e.g., SimpleUnderwater-Hovering) ship without cameras.
    # Add a minimal LeftCamera so every world can export an on-vehicle view.
    if not any(s.get("sensor_type") == "RGBCamera" for s in sensors):
        sensors.append(
            {
                "sensor_type": "RGBCamera",
                "sensor_name": "LeftCamera",
                "socket": "CameraLeftSocket",
                "Hz": int(cfg.fps),
                "configuration": {"CaptureWidth": int(cfg.camera_width), "CaptureHeight": int(cfg.camera_height)},
            }
        )
    else:
        # If cameras already exist, bump resolution slightly for clearer screenshots/GIFs.
        for s in sensors:
            if s.get("sensor_type") != "RGBCamera":
                continue
            conf = dict(s.get("configuration", {}) or {})
            w = int(conf.get("CaptureWidth", 512))
            h = int(conf.get("CaptureHeight", 512))
            if w < int(cfg.camera_width) or h < int(cfg.camera_height):
                conf["CaptureWidth"] = int(cfg.camera_width)
                conf["CaptureHeight"] = int(cfg.camera_height)
            s["configuration"] = conf

    if not any(s.get("sensor_type") == "CollisionSensor" for s in sensors):
        sensors.append({"sensor_type": "CollisionSensor", "sensor_name": "CollisionSensor", "Hz": cfg.ticks_per_sec})

    if add_viewport_capture and not any(s.get("sensor_type") == "ViewportCapture" for s in sensors):
        sensors.append(
            {
                "sensor_type": "ViewportCapture",
                "sensor_name": "ViewportCapture",
                "Hz": int(cfg.fps),
                "configuration": {"CaptureWidth": int(cfg.window_width), "CaptureHeight": int(cfg.window_height)},
            }
        )

    main["sensors"] = sensors
    agents[0] = main
    scenario["agents"] = agents
    return scenario


def add_hovering_auv_agents(base_scenario: dict, *, n_agents: int) -> dict:
    if n_agents < 1:
        raise ValueError("n_agents must be >= 1")
    scenario = copy.deepcopy(base_scenario)
    agents = scenario.get("agents", [])
    if not agents:
        raise ValueError("Scenario has no agents.")

    a0 = agents[0]
    if a0.get("agent_type") != "HoveringAUV":
        raise ValueError(f"Expected HoveringAUV as agent0; got {a0.get('agent_type')!r}")

    # Keep full sensors for auv0; for others keep a minimal set to reduce crash risk
    # when scaling to N=8/10.
    def _minimize_non_main_sensors(a: dict) -> dict:
        a = copy.deepcopy(a)
        keep_types = {"PoseSensor"}
        sensors = [s for s in a.get("sensors", []) if s.get("sensor_type") in keep_types]
        if not sensors:
            raise ValueError("Non-main agent would have zero sensors after minimization.")
        a["sensors"] = sensors
        return a

    out_agents: list[dict] = []
    for i in range(n_agents):
        ai = copy.deepcopy(a0)
        ai["agent_name"] = f"auv{i}"
        if i > 0:
            ai = _minimize_non_main_sensors(ai)
        out_agents.append(ai)

    scenario["agents"] = out_agents
    scenario["main_agent"] = "auv0"
    return scenario
