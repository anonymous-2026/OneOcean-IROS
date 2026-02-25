from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence, Tuple

import msgpackrpc


@dataclass(frozen=True)
class RpcAddress:
    ip: str = "127.0.0.1"
    port: int = 41451  # AirSim default RPC port


class AirSimRpc:
    def __init__(self, address: RpcAddress = RpcAddress(), timeout_s: float = 60.0):
        self._address = address
        self._client = msgpackrpc.Client(msgpackrpc.Address(address.ip, address.port), timeout=timeout_s)

    def call(self, method: str, *args: Any) -> Any:
        return self._client.call(method, *args)

    def call_async(self, method: str, *args: Any) -> Any:
        return self._client.call_async(method, *args)

    def ping(self) -> bool:
        return bool(self.call("ping"))

    def confirm_connection(self, retries: int = 120, sleep_s: float = 1.0) -> None:
        last_error: Optional[Exception] = None
        for _ in range(retries):
            try:
                self.ping()
                return
            except Exception as exc:  # noqa: BLE001 - surface error to caller after retries
                last_error = exc
                time.sleep(sleep_s)
        raise TimeoutError(f"AirSim RPC not reachable at {self._address}. Last error: {last_error!r}")

    # ----- common vehicle control -----
    def enable_api_control(self, enabled: bool, vehicle_name: str = "") -> None:
        self.call("enableApiControl", enabled, vehicle_name)

    def arm_disarm(self, arm: bool, vehicle_name: str = "") -> bool:
        return bool(self.call("armDisarm", arm, vehicle_name))

    # ----- pose / external physics -----
    def sim_get_vehicle_pose(self, vehicle_name: str = "") -> Dict[str, Any]:
        return self.call("simGetVehiclePose", vehicle_name)

    def sim_set_vehicle_pose(self, pose: Dict[str, Any], ignore_collision: bool, vehicle_name: str = "") -> None:
        self.call("simSetVehiclePose", pose, ignore_collision, vehicle_name)

    # ----- image capture -----
    def sim_get_image(
        self,
        camera_name: str,
        image_type: int,
        vehicle_name: str = "",
        external: bool = False,
    ) -> bytes:
        data = self.call("simGetImage", camera_name, image_type, vehicle_name, external)
        return bytes(data)

    # ----- collisions -----
    def sim_get_collision_info(self, vehicle_name: str = "") -> Dict[str, Any]:
        return self.call("simGetCollisionInfo", vehicle_name)


def make_pose_xyz_yaw(x: float, y: float, z: float, yaw_rad: float) -> Dict[str, Any]:
    import math

    half = 0.5 * yaw_rad
    qw = math.cos(half)
    qz = math.sin(half)
    return {
        "position": {"x_val": float(x), "y_val": float(y), "z_val": float(z)},
        "orientation": {"w_val": float(qw), "x_val": 0.0, "y_val": 0.0, "z_val": float(qz)},
    }


def pose_to_xyz_yaw(pose: Dict[str, Any]) -> Tuple[float, float, float, float]:
    import math

    p = pose["position"]
    q = pose["orientation"]
    x, y, z = float(p["x_val"]), float(p["y_val"]), float(p["z_val"])
    qw, qz = float(q["w_val"]), float(q["z_val"])
    yaw = 2.0 * math.atan2(qz, qw)
    return x, y, z, yaw

