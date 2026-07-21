from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    default_parameters = str(Path(get_package_share_directory("oneocean_ros")) / "config" / "default.yaml")
    parameters_file = LaunchConfiguration("params_file")
    drift_npz = LaunchConfiguration("drift_npz")
    return LaunchDescription(
        [
            DeclareLaunchArgument("params_file", default_value=default_parameters),
            DeclareLaunchArgument("drift_npz"),
            Node(
                package="oneocean_ros",
                executable="oneocean_bridge",
                name="oneocean_bridge",
                output="screen",
                parameters=[parameters_file, {"drift_npz": drift_npz}],
            ),
        ]
    )
