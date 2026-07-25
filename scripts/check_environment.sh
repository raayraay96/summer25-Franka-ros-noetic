#!/usr/bin/env bash
# Environment sanity checks for vision_arm_control.
#
# Modes:
#   bash scripts/check_environment.sh           # ROS-oriented (default)
#   bash scripts/check_environment.sh --dev     # pure Python / portfolio tests only
#   bash scripts/check_environment.sh --strict  # fail on warnings too
set -euo pipefail

MODE="ros"
STRICT=0
for arg in "$@"; do
  case "$arg" in
    --dev) MODE="dev" ;;
    --strict) STRICT=1 ;;
    -h|--help)
      sed -n '2,10p' "$0"
      exit 0
      ;;
  esac
done

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

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== vision_arm_control environment check ==="
echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "host: $(hostname 2>/dev/null || echo unknown)"
echo "python: $(python3 --version 2>&1)"
echo "mode: $MODE"
echo "repo: $ROOT"

# Layout hygiene (always)
check "src/ only vision_arm_control" bash -c '
  mapfile -t entries < <(ls -1 "'"$ROOT"'/src" 2>/dev/null)
  [[ ${#entries[@]} -eq 1 && "${entries[0]}" == "vision_arm_control" ]]
'
check "no .gitmodules" test ! -f "$ROOT/.gitmodules"
check "no .pth in working tree" bash -c '
  ! find "'"$ROOT"'" -path "'"$ROOT"'/.git" -prune -o -type f \( -name "*.pth" -o -name "*.pt" -o -name "*.onnx" \) -print -quit | grep -q .
'

# Pure Python portfolio path
warn_check "numpy" python3 -c "import numpy"
warn_check "yaml" python3 -c "import yaml"
warn_check "pytest" python3 -c "import pytest"

if [[ "$MODE" == "dev" ]]; then
  check "unit tests import path" python3 -c "
import sys
from pathlib import Path
sys.path.insert(0, str(Path('$ROOT')/'src'/'vision_arm_control'/'src'))
import vision_arm_control
print(vision_arm_control.__version__)
"
  echo
  if [[ "$ok" -ne 0 ]]; then
    echo "Critical checks failed."
    exit 1
  fi
  if [[ "$warn" -ne 0 && "$STRICT" -eq 1 ]]; then
    echo "Completed with warnings (strict mode)."
    exit 1
  fi
  if [[ "$warn" -ne 0 ]]; then
    echo "Completed with warnings (optional deps may be missing)."
    exit 0
  fi
  echo "Dev checks passed. Run: python3 -m pytest -q tests/"
  exit 0
fi

# ROS-oriented checks
check "ROS_DISTRO set" bash -c '[[ -n "${ROS_DISTRO:-}" ]]'
warn_check "ROS Noetic" bash -c '[[ "${ROS_DISTRO:-}" == "noetic" ]]'
warn_check "rospack vision_arm_control" bash -c 'command -v rospack >/dev/null && rospack find vision_arm_control >/dev/null'
warn_check "cv2" python3 -c "import cv2"
warn_check "mediapipe" python3 -c "import mediapipe"
warn_check "torch" python3 -c "import torch"

if [[ -n "${FRANKA_MODEL_DIR:-}" ]]; then
  check "FRANKA_MODEL_DIR exists" test -d "$FRANKA_MODEL_DIR"
  warn_check "encoder.pth" test -f "$FRANKA_MODEL_DIR/encoder.pth"
  warn_check "depth.pth" test -f "$FRANKA_MODEL_DIR/depth.pth"
else
  echo "[WARN] FRANKA_MODEL_DIR not set (see docs/model-setup.md)"
  warn=1
fi

echo
if [[ "$ok" -ne 0 ]]; then
  echo "Critical checks failed."
  exit 1
fi
if [[ "$warn" -ne 0 && "$STRICT" -eq 1 ]]; then
  echo "Completed with warnings (strict mode)."
  exit 1
fi
if [[ "$warn" -ne 0 ]]; then
  echo "Completed with warnings (optional deps may be missing)."
  exit 0
fi
echo "All checks passed."
