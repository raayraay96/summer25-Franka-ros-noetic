#!/usr/bin/env bash
set -euo pipefail
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
set +u
source /opt/ros/noetic/setup.bash
source /ws/devel/setup.bash
set -u
python3 -m pytest -q /workspaces/franka-teleop/tests/
rospack find vision_arm_control
set +e
timeout 15s roslaunch vision_arm_control simulation.launch use_gazebo:=false use_moveit:=false
status=$?
set -e
if [[ "$status" -ne 0 && "$status" -ne 124 ]]; then
  echo "simulation.launch failed with status $status" >&2
  exit "$status"
fi
echo "simulation.launch dry-run smoke test passed"
