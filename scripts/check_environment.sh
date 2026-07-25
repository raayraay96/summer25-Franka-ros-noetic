#!/usr/bin/env bash
# Environment sanity checks for vision_arm_control.
set -euo pipefail

ok=0
warn=0

check() {
  local label="$1"
  shift
  if "$@"; then
    echo "[OK]   $label"
  else
    echo "[FAIL] $label"
    ok=1
  fi
}

warn_check() {
  local label="$1"
  shift
  if "$@"; then
    echo "[OK]   $label"
  else
    echo "[WARN] $label"
    warn=1
  fi
}

echo "=== vision_arm_control environment check ==="
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host: $(hostname)"
echo "python: $(python3 --version 2>&1)"

check "ROS_DISTRO set" bash -c '[[ -n "${ROS_DISTRO:-}" ]]'
warn_check "ROS Noetic" bash -c '[[ "${ROS_DISTRO:-}" == "noetic" ]]'
warn_check "rospack vision_arm_control" bash -c 'command -v rospack >/dev/null && rospack find vision_arm_control >/dev/null'
warn_check "numpy" python3 -c "import numpy"
warn_check "cv2" python3 -c "import cv2"
warn_check "mediapipe" python3 -c "import mediapipe"
warn_check "torch" python3 -c "import torch"
warn_check "yaml" python3 -c "import yaml"

if [[ -n "${FRANKA_MODEL_DIR:-}" ]]; then
  check "FRANKA_MODEL_DIR exists" test -d "$FRANKA_MODEL_DIR"
  warn_check "encoder.pth" test -f "$FRANKA_MODEL_DIR/encoder.pth"
  warn_check "depth.pth" test -f "$FRANKA_MODEL_DIR/depth.pth"
else
  echo "[WARN] FRANKA_MODEL_DIR not set"
  warn=1
fi

echo
if [[ "$ok" -ne 0 ]]; then
  echo "Critical checks failed."
  exit 1
fi
if [[ "$warn" -ne 0 ]]; then
  echo "Completed with warnings (optional deps may be missing)."
  exit 0
fi
echo "All checks passed."
