#!/usr/bin/env bash
# Record the three end-to-end simulation scenarios on a ROS 2 host (Scholar).
# NOT runnable in the no-ROS audit VM. See docs/research/scholar-completion-runbook.md.
#
#   A. shoulder_relative + cbf_qp        (baseline safe teleop)
#   B. sew_orientation  + cbf_qp         (SEW feature path; ik_backend=geometric_unverified for RViz)
#   C. obstacle/workspace intervention   (same pipeline; obstacle in path)
#
# Prereqs on the ROS 2 host:
#   - ROS 2 (Humble+), colcon-built ros2_ws, source install/setup.bash
#   - vision_arm_control on PYTHONPATH (the node inserts it from the repo, but
#     `pip install -e` of the core or PYTHONPATH export is recommended)
#   - a Panda fake-hardware / RViz bringup (panda_rviz_sim.launch.py or MoveIt)
#   - ros2 bag, and (optional) a screen recorder for the RViz MP4
set -euo pipefail

OUT="${1:-results/v1.1-hardening/ros2}"
DUR="${DUR:-20}"      # seconds per scenario
mkdir -p "$OUT"

run_scenario () {
  local name="$1" strategy="$2" safety="$3" ik="$4"
  local jsonl="$OUT/${name}.jsonl"
  echo "=== scenario ${name}: ${strategy} + ${safety} (ik=${ik}) ==="
  ros2 launch franka_teleop_bringup teleop_pipeline_demo.launch.py \
      strategy:="${strategy}" safety:="${safety}" ik_backend:="${ik}" \
      diagnostics_jsonl:="${jsonl}" &
  local lpid=$!
  # Record the diagnostics + pose/joint/safety topics for the window.
  timeout "${DUR}" ros2 bag record -o "$OUT/${name}_bag" \
      /teleop/target_pose /teleop/command_pose /joint_states \
      /teleop/safety_status /teleop/gate_status /teleop/diagnostics || true
  kill "${lpid}" 2>/dev/null || true
  sleep 1
  # Offline metrics from the diagnostics JSONL (pure-Python, CI-tested aggregator).
  python3 scripts/extract_run_metrics.py "${jsonl}" --output "$OUT/${name}_metrics.json" || true
}

run_scenario "scenario_a_shoulder_cbf"     shoulder_relative cbf_qp none
run_scenario "scenario_b_sew_cbf"          sew_orientation   cbf_qp geometric_unverified
run_scenario "scenario_c_obstacle_interv"  shoulder_relative cbf_qp none

echo "Recorded scenarios + metrics under: $OUT"
echo "Next: capture an RViz MP4/GIF with burned-in labels (see runbook) and commit evidence."
