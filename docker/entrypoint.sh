#!/usr/bin/env bash
set -euo pipefail
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
set +u
source /opt/ros/noetic/setup.bash
if [[ -f /ws/devel/setup.bash ]]; then
  source /ws/devel/setup.bash
fi
set -u
exec "$@"
