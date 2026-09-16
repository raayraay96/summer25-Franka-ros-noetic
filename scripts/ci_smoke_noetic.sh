#!/usr/bin/env bash
# Local ROS Noetic end-to-end gate; the filename is retained for compatibility.
set -euo pipefail
export ROS_MASTER_URI="${ROS_MASTER_URI:-http://localhost:11311}"
set +u
source "${ROS_SETUP_FILE:-/opt/ros/noetic/setup.bash}"
source "${CATKIN_SETUP_FILE:-/ws/devel/setup.bash}"
set -u
rospack find vision_arm_control
# Refuse an existing master: unrelated nodes must not satisfy this test.
if timeout 3s rosparam get /run_id >/dev/null 2>&1; then
  echo "A ROS master is already running at $ROS_MASTER_URI; use an isolated master for the smoke test." >&2
  exit 1
fi
scratch_dir=$(mktemp -d)
launch_pid=""
cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "$launch_pid" ]]; then
    kill -TERM "$launch_pid" 2>/dev/null || true
    wait "$launch_pid" 2>/dev/null || true
  fi
  if [[ "$status" -ne 0 ]]; then
    cat "$scratch_dir/launch.log" >&2
    cat "$scratch_dir/topic.log" >&2
  fi
  rm -rf "$scratch_dir"
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
# Outer timeout bounds shutdown too if ROS gets stuck.
timeout --kill-after=5s 55s roslaunch vision_arm_control simulation.launch \
  use_gazebo:=false use_moveit:=false >"$scratch_dir/launch.log" 2>&1 &
launch_pid=$!
: >"$scratch_dir/topic.log"
master_ready=false
deadline=$((SECONDS + 15))
while (( SECONDS < deadline )); do
  if ! kill -0 "$launch_pid" 2>/dev/null; then
    break
  fi
  if timeout 1s rosparam get /run_id >/dev/null 2>&1; then
    master_ready=true
    break
  fi
  sleep 0.1
done
if [[ "$master_ready" != true ]]; then
  echo "Simulation failed: ROS master did not become ready." >&2
  exit 1
fi
if ! timeout --kill-after=3s 30s rostopic echo -n 3 \
  /robot_controller_node/executed_pose >"$scratch_dir/topic.log" 2>&1; then
  echo "Simulation failed: controller did not echo three poses within 30 seconds." >&2
  exit 1
fi
if ! kill -0 "$launch_pid" 2>/dev/null || \
   [[ $(grep -c '^---$' "$scratch_dir/topic.log" || true) -lt 3 ]]; then
  echo "Simulation failed: launch exited or controller output was incomplete." >&2
  exit 1
fi
cat "$scratch_dir/topic.log"
echo "simulation.launch dry-run smoke test passed: three controller poses observed (no hardware)."
