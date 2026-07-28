#!/usr/bin/env bash
set -euo pipefail
mkdir src
catkin_make
source devel/setup.bash
echo $ROS_PACKAGE_PATH
chmod +x ./devel/setup.bash
