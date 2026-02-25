from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .rpc import AirSimRpc, RpcAddress


@dataclass(frozen=True)
class RovConfig:
    vehicle_name: str = "RovSimple"


class RovClient:
    """Python-side client for UNav-Sim's ROV RPC surface.

    UNav-Sim adds a dedicated ROV RPC client in C++ (RovRpcLibClient) that calls methods like:
    - getRovState
    - moveByMotorPWMs

    The official upstream PythonClient does not expose a RovClient class, but the RPC methods are
    still accessible via msgpack-rpc. This wrapper keeps the payloads as plain dicts for robustness.
    """

    def __init__(self, address: RpcAddress = RpcAddress(), timeout_s: float = 60.0, vehicle_name: str = "RovSimple"):
        self.rpc = AirSimRpc(address=address, timeout_s=timeout_s)
        self.vehicle_name = vehicle_name

    def confirm_connection(self) -> None:
        self.rpc.confirm_connection()

    def enable_api_control(self, enabled: bool) -> None:
        self.rpc.enable_api_control(enabled, self.vehicle_name)

    def arm_disarm(self, arm: bool) -> bool:
        return self.rpc.arm_disarm(arm, self.vehicle_name)

    def get_rov_state(self) -> Dict[str, Any]:
        return self.rpc.call("getRovState", self.vehicle_name)

    def move_by_motor_pwms_async(self, pwms: List[float], duration_s: float) -> Any:
        return self.rpc.call_async("moveByMotorPWMs", pwms, float(duration_s), self.vehicle_name)

    def sim_get_image(self, camera_name: str = "front_right_custom", image_type: int = 0, external: bool = False) -> bytes:
        return self.rpc.sim_get_image(camera_name=camera_name, image_type=image_type, vehicle_name=self.vehicle_name, external=external)

    def sim_get_vehicle_pose(self) -> Dict[str, Any]:
        return self.rpc.sim_get_vehicle_pose(self.vehicle_name)

    def sim_set_vehicle_pose(self, pose: Dict[str, Any], ignore_collision: bool = False) -> None:
        self.rpc.sim_set_vehicle_pose(pose=pose, ignore_collision=ignore_collision, vehicle_name=self.vehicle_name)

    def sim_get_collision_info(self) -> Dict[str, Any]:
        return self.rpc.sim_get_collision_info(self.vehicle_name)


def extract_position_xy(state: Dict[str, Any]) -> Optional[tuple[float, float, float]]:
    try:
        kin = state["kinematics_estimated"]
        pos = kin["position"]
        return float(pos["x_val"]), float(pos["y_val"]), float(pos["z_val"])
    except Exception:
        return None

