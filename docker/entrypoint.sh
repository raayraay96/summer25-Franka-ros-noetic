#!/bin/bash
set -e
source /opt/ros/noetic/setup.bash
if [[ -f /ws/devel/setup.bash ]]; then
  # shellcheck disable=SC1091
  source /ws/devel/setup.bash
fi
exec "$@"
