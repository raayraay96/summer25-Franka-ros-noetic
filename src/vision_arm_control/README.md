# vision_arm_control (ROS package)

Catkin package for vision-guided Franka end-effector position teleoperation.

See the repository root [README.md](../../README.md) for the full portfolio documentation.

## Layout

```text
config/     YAML parameters
launch/     mimicry, perception_only, simulation, rosbag_replay
scripts/    ROS nodes (+ legacy/)
src/vision_arm_control/   pure Python library (testable without ROS)
```

## Run

```bash
roslaunch vision_arm_control perception_only.launch
roslaunch vision_arm_control simulation.launch
roslaunch vision_arm_control mimicry.launch use_robot:=false
```
