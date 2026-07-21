from __future__ import annotations

import json
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Float32, String
from std_srvs.srv import Trigger

from benchmark_core.controllers import preset_controller
from benchmark_core.env import EnvConfig, HeadlessOceanEnv
from benchmark_core.tasks import preset_task

from .messages import commands_enu_to_core, snapshots_from_observation


def _json_value(value: Any) -> Any:
    if is_dataclass(value):
        return _json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    return value


class OneOceanBridge(Node):
    def __init__(self) -> None:
        super().__init__("oneocean_bridge")
        defaults: dict[str, Any] = {
            "drift_npz": "",
            "output_directory": "runs/ros2",
            "task": "go_to_goal_current",
            "difficulty": "medium",
            "controller": "go_to_goal",
            "pollution_model": "gaussian",
            "n_agents": 1,
            "seed": 0,
            "dt_s": 1.0,
            "dynamics_model": "6dof",
            "constraint_mode": "hard",
            "bathy_mode": "off",
            "control_mode": "external",
            "autostart": True,
            "auto_reset": False,
            "increment_seed_on_reset": False,
            "realtime_rate": 1.0,
            "command_timeout_s": 2.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)
        self._episode_index = 0
        self._environment: HeadlessOceanEnv | None = None
        self._last_info: dict[str, Any] = {}
        self._done = False
        self._n_agents = int(self.get_parameter("n_agents").value)
        self._commands_enu = np.zeros((self._n_agents, 3), dtype=np.float64)
        self._command_times = np.full((self._n_agents,), -np.inf, dtype=np.float64)
        self._clock_publisher = self.create_publisher(Clock, "/clock", 10)
        self._metrics_publisher = self.create_publisher(String, "/oneocean/episode/metrics", 10)
        self._odom_publishers = []
        self._current_publishers = []
        self._pollution_publishers = []
        self._bathymetry_publishers = []
        self._command_subscriptions = []
        for agent_index in range(self._n_agents):
            prefix = f"/oneocean/agents/agent_{agent_index:03d}"
            self._odom_publishers.append(self.create_publisher(Odometry, f"{prefix}/odom", 10))
            self._current_publishers.append(self.create_publisher(Vector3Stamped, f"{prefix}/current", 10))
            self._pollution_publishers.append(self.create_publisher(Float32, f"{prefix}/pollution", 10))
            self._bathymetry_publishers.append(self.create_publisher(Float32, f"{prefix}/bathymetry", 10))
            self._command_subscriptions.append(
                self.create_subscription(Twist, f"{prefix}/cmd_vel", lambda message, index=agent_index: self._receive_command(index, message), 10)
            )
        self.create_service(Trigger, "/oneocean/reset", self._reset_service)
        self.create_service(Trigger, "/oneocean/step", self._step_service)
        self.create_service(Trigger, "/oneocean/get_task_state", self._task_state_service)
        self._create_environment()
        dt_s = float(self.get_parameter("dt_s").value)
        realtime_rate = max(1e-6, float(self.get_parameter("realtime_rate").value))
        self._timer = self.create_timer(dt_s / realtime_rate, self._timer_callback) if bool(self.get_parameter("autostart").value) else None
        self._publish_state()

    def _create_environment(self) -> None:
        if self._environment is not None:
            self._environment.close()
        drift_npz = str(self.get_parameter("drift_npz").value).strip()
        if not drift_npz:
            raise ValueError("The drift_npz ROS parameter is required")
        output_root = Path(str(self.get_parameter("output_directory").value)).expanduser().resolve()
        output_directory = output_root / f"episode_{self._episode_index:03d}"
        output_directory.mkdir(parents=True, exist_ok=True)
        config = EnvConfig(
            drift_cache_npz=drift_npz,
            pollution_model=str(self.get_parameter("pollution_model").value),
            dt_s=float(self.get_parameter("dt_s").value),
            dynamics_model=str(self.get_parameter("dynamics_model").value),
            constraint_mode=str(self.get_parameter("constraint_mode").value),
            bathy_mode=str(self.get_parameter("bathy_mode").value),
        )
        task = preset_task(
            kind=str(self.get_parameter("task").value),
            difficulty=str(self.get_parameter("difficulty").value),
        )
        controller = preset_controller(
            kind=str(self.get_parameter("controller").value),
            max_speed_mps=config.max_speed_mps,
        )
        self._environment = HeadlessOceanEnv(
            config,
            out_dir=output_directory,
            seed=int(self.get_parameter("seed").value)
            + (self._episode_index if bool(self.get_parameter("increment_seed_on_reset").value) else 0),
            n_agents=self._n_agents,
        )
        self._environment.reset(task=task, controller=controller)
        self._commands_enu.fill(0.0)
        self._command_times.fill(-np.inf)
        self._last_info = {}
        self._done = False

    @staticmethod
    def _time_message(time_s: float) -> Time:
        seconds = int(np.floor(max(0.0, float(time_s))))
        nanoseconds = int(round((max(0.0, float(time_s)) - seconds) * 1_000_000_000.0))
        if nanoseconds >= 1_000_000_000:
            seconds += 1
            nanoseconds -= 1_000_000_000
        return Time(sec=seconds, nanosec=nanoseconds)

    def _receive_command(self, agent_index: int, message: Twist) -> None:
        self._commands_enu[agent_index] = np.array([message.linear.x, message.linear.y, message.linear.z], dtype=np.float64)
        self._command_times[agent_index] = time.monotonic()

    def _actions(self) -> np.ndarray | None:
        if str(self.get_parameter("control_mode").value) == "internal":
            return None
        commands = self._commands_enu.copy()
        timeout_s = float(self.get_parameter("command_timeout_s").value)
        if timeout_s >= 0.0:
            stale = (time.monotonic() - self._command_times) > timeout_s
            commands[stale] = 0.0
        return commands_enu_to_core(commands, self._n_agents)

    def _advance(self) -> None:
        if self._environment is None:
            return
        if self._done:
            if bool(self.get_parameter("auto_reset").value):
                self._episode_index += 1
                self._create_environment()
            else:
                return
        self._done, self._last_info = self._environment.step(self._actions())
        self._publish_state()

    def _publish_state(self) -> None:
        if self._environment is None:
            return
        observation = self._environment.observe()
        time_s = float(observation["time_s"])
        stamp = self._time_message(time_s)
        clock = Clock()
        clock.clock = stamp
        self._clock_publisher.publish(clock)
        for snapshot in snapshots_from_observation(observation):
            agent_index = snapshot.agent_index
            odometry = Odometry()
            odometry.header.stamp = stamp
            odometry.header.frame_id = "oneocean/map_enu"
            odometry.child_frame_id = f"oneocean/agent_{agent_index}/base_link"
            odometry.pose.pose.position.x, odometry.pose.pose.position.y, odometry.pose.pose.position.z = snapshot.position_enu_m.tolist()
            odometry.pose.pose.orientation.x, odometry.pose.pose.orientation.y, odometry.pose.pose.orientation.z, odometry.pose.pose.orientation.w = snapshot.orientation_enu_xyzw.tolist()
            odometry.twist.twist.linear.x, odometry.twist.twist.linear.y, odometry.twist.twist.linear.z = snapshot.linear_velocity_enu_mps.tolist()
            odometry.twist.twist.angular.x, odometry.twist.twist.angular.y, odometry.twist.twist.angular.z = snapshot.angular_velocity_enu_rps.tolist()
            self._odom_publishers[agent_index].publish(odometry)
            current = Vector3Stamped()
            current.header.stamp = stamp
            current.header.frame_id = "oneocean/map_enu"
            current.vector.x, current.vector.y, current.vector.z = snapshot.current_enu_mps.tolist()
            self._current_publishers[agent_index].publish(current)
            pollution = Float32()
            pollution.data = float(snapshot.pollution)
            self._pollution_publishers[agent_index].publish(pollution)
            bathymetry = Float32()
            bathymetry.data = float(snapshot.bathymetry_depth_m)
            self._bathymetry_publishers[agent_index].publish(bathymetry)
        metrics = String()
        metrics.data = json.dumps({"time_s": time_s, "done": bool(self._done), **self._last_info}, allow_nan=True)
        self._metrics_publisher.publish(metrics)

    def _timer_callback(self) -> None:
        self._advance()

    def _reset_service(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        try:
            self._episode_index += 1
            self._create_environment()
            self._publish_state()
            response.success = True
            response.message = f"reset to episode {self._episode_index}"
        except Exception as exception:
            response.success = False
            response.message = f"{type(exception).__name__}: {exception}"
        return response

    def _step_service(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        try:
            self._advance()
            response.success = True
            response.message = json.dumps({"done": bool(self._done), **self._last_info}, allow_nan=True)
        except Exception as exception:
            response.success = False
            response.message = f"{type(exception).__name__}: {exception}"
        return response

    def _task_state_service(self, request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        del request
        if self._environment is None:
            response.success = False
            response.message = "environment is not initialized"
            return response
        response.success = True
        response.message = json.dumps(
            _json_value(
                {
                    "episode_index": self._episode_index,
                    "done": self._done,
                    "task_config": self._environment.task_cfg,
                    "task_state": self._environment.task_state,
                    "last_step": self._last_info,
                }
            ),
            allow_nan=True,
        )
        return response

    def destroy_node(self) -> bool:
        if self._environment is not None:
            self._environment.close()
            self._environment = None
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node: OneOceanBridge | None = None
    try:
        node = OneOceanBridge()
        rclpy.spin(node)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
