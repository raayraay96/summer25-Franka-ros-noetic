#!/usr/bin/env bash
set -euo pipefail
# Prepare a catkin workspace checkout of this repository (ROS Noetic / Ubuntu 20.04).

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WS="${CATKIN_WS:-$HOME/catkin_ws}"

echo "Repository: $ROOT"
echo "Catkin WS:  $WS"

if [[ -z "${ROS_DISTRO:-}" ]]; then
  if [[ -f /opt/ros/noetic/setup.bash ]]; then
    # shellcheck disable=SC1091
    set +u
    source /opt/ros/noetic/setup.bash
    set -u
  else
    echo "ROS Noetic not found. Install ROS Noetic first." >&2
    exit 1
  fi
fi

mkdir -p "$WS/src"
# Link or copy package into workspace if not already there
if [[ ! -e "$WS/src/vision_arm_control" ]]; then
  ln -s "$ROOT/src/vision_arm_control" "$WS/src/vision_arm_control"
  echo "Linked vision_arm_control into $WS/src"
fi

python3 -m pip install --user -r "$ROOT/requirements.txt" || true

cd "$WS"
if command -v catkin_make >/dev/null 2>&1; then
  catkin_make
  # shellcheck disable=SC1091
  source devel/setup.bash
  echo "Build complete. Sourced $WS/devel/setup.bash"
else
  echo "catkin_make not found; ROS environment incomplete." >&2
  exit 1
fi

echo "Next: bash $ROOT/scripts/check_environment.sh"
