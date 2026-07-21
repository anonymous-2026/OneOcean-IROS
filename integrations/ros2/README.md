# ROS 2 Integration

`oneocean_ros` is an optional ROS 2 Jazzy bridge around the public `HeadlessOceanEnv` interface. It is an integration layer, not an additional simulator backend, and ROS packages are not required by the data pipeline or benchmark core.

## Frame and control contract

- OneOcean core: `[x, y, z] = [east, depth-down, north]` in meters.
- ROS map frame: ENU `[x, y, z] = [east, north, up]`.
- Vector conversion: `core [x, y, z] -> ROS [x, z, -y]`.
- `/cmd_vel` linear velocity is interpreted in ENU and converted to the core frame.
- External commands pass through the same speed clipping, dynamics, current, terrain constraints, recorder, task updates, and success logic as built-in controllers.
- Set `control_mode:=internal` to run the configured OneOcean controller while still publishing ROS observations.

## Topics and services

Published for each zero-based agent index:

- `/oneocean/agents/agent_<id>/odom` (`nav_msgs/Odometry`)
- `/oneocean/agents/agent_<id>/current` (`geometry_msgs/Vector3Stamped`)
- `/oneocean/agents/agent_<id>/pollution` (`std_msgs/Float32`)
- `/oneocean/agents/agent_<id>/bathymetry` (`std_msgs/Float32`, positive water depth; NaN for masked/unknown cells)

The bridge also publishes `/clock` and `/oneocean/episode/metrics` as a JSON `std_msgs/String`. It subscribes to `/oneocean/agents/agent_<id>/cmd_vel`, where IDs are zero-padded (for example, `agent_000`). `/oneocean/reset`, `/oneocean/step`, and `/oneocean/get_task_state` use `std_srvs/Trigger`; JSON state is returned in the response message. The step service is useful with `autostart:=false`. Reset preserves the configured seed by default; set `increment_seed_on_reset:=true` to advance it with the episode index.

## Build on Ubuntu 24.04 / ROS 2 Jazzy

Install the normal OneOcean Python requirements first. ROS remains outside `requirements.txt`.

```bash
source /opt/ros/jazzy/setup.bash
export ONEOCEAN_ROOT=/path/to/OneOcean
export PYTHONPATH="$ONEOCEAN_ROOT:$PYTHONPATH"

mkdir -p ~/oneocean_ros_ws/src
ln -s "$ONEOCEAN_ROOT/integrations/ros2/oneocean_ros" ~/oneocean_ros_ws/src/oneocean_ros
cd ~/oneocean_ros_ws
colcon build --symlink-install
source install/setup.bash
```

Export a drift cache using `docs/DATASET_CONTRACT.md`, then launch:

```bash
ros2 launch oneocean_ros oneocean_core.launch.py \
  drift_npz:=/absolute/path/to/drift_scene_t0_d0.npz
```

For manual stepping:

```bash
ros2 run oneocean_ros oneocean_bridge --ros-args \
  -p drift_npz:=/absolute/path/to/drift_scene_t0_d0.npz \
  -p autostart:=false
ros2 service call /oneocean/step std_srvs/srv/Trigger '{}'
```

The host used for the paper release does not install ROS system-wide, so repository regression tests cover the external-action boundary, frame conversion, observation conversion, clipping, and backward compatibility without making ROS a core dependency. The package was additionally built and launched in an `osrf/ros:jazzy-desktop` container: topic discovery and a `/oneocean/step` service call completed successfully.
