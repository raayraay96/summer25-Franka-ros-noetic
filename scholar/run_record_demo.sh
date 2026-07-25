#!/bin/bash
# Record ROS 2 Panda teleop demo under Xvfb (run via SLURM).
set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/scholar/edraymon/franka-teleoperation}"
DATA="${RCAC_SCRATCH:-/scratch/scholar/edraymon}/franka-teleop-data"
SIF="${SIF:-$DATA/containers/ros2_humble_franka.sif}"
OUT="$DATA/raw_video/ros2_demo_${SLURM_JOB_ID:-manual}"
DEMO="$REPO_DIR/docs/demo"
RESULTS="$REPO_DIR/results/ros2"
mkdir -p "$OUT" "$DEMO" "$RESULTS"

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -lh "$SIF"

apptainer exec \
  -B "$REPO_DIR:/repo" \
  -B "$DATA:/data" \
  -B "$OUT:/out" \
  "$SIF" bash -lc '
set -e
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CC=/usr/bin/gcc CXX=/usr/bin/g++
source /opt/ros/humble/setup.bash
cd /repo/ros2_ws
# build if needed
if [[ ! -f install/setup.bash ]]; then
  colcon build --packages-select franka_teleop_ros2 franka_teleop_bringup
fi
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LIBGL_ALWAYS_SOFTWARE=1
export QT_QPA_PLATFORM=xcb
export MESA_GL_VERSION_OVERRIDE=3.3
export DISPLAY=:99

Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
XPID=$!
sleep 2

ros2 launch franka_teleop_bringup panda_rviz_sim.launch.py use_rviz:=true diagnostics_out:=/out/sim_metrics.json &
LPID=$!
# Give RViz time to start
sleep 12

# Record 16s
ffmpeg -y -video_size 1280x720 -framerate 12 -f x11grab -draw_mouse 0 -i :99.0 \
  -t 16 -c:v libx264 -pix_fmt yuv420p -crf 28 /out/raw_demo.mp4

ffmpeg -y -ss 00:00:08 -i /out/raw_demo.mp4 -frames:v 1 /out/thumbnail.png || true

# Label overlay
ffmpeg -y -i /out/raw_demo.mp4 \
  -vf "drawtext=text='\''ROS 2 Humble | RViz fake-hardware | No physical Franka | Purdue Scholar'\'':fontcolor=white:fontsize=20:box=1:boxcolor=black@0.55:x=16:y=16" \
  -c:v libx264 -pix_fmt yuv420p -crf 28 -t 16 /out/labeled_demo.mp4

# GIF
ffmpeg -y -i /out/labeled_demo.mp4 -vf "fps=8,scale=720:-1:flags=lanczos" -t 12 /out/demo.gif

# Stop launch cleanly
kill $LPID 2>/dev/null || true
sleep 2
kill $XPID 2>/dev/null || true
sleep 1
# best-effort cleanup of leftover children without pkill -f on self
for p in rviz2 robot_state_publisher mock_landmark human_to_robot safety_monitor robot_controller target_visualizer diagnostics Xvfb; do
  pkill -x "$p" 2>/dev/null || true
done
ls -lh /out/
'

# Copy into repo
cp -f "$OUT/labeled_demo.mp4" "$DEMO/ros2-franka-teleoperation.mp4" 2>/dev/null || cp -f "$OUT/raw_demo.mp4" "$DEMO/ros2-franka-teleoperation.mp4"
cp -f "$OUT/demo.gif" "$DEMO/ros2-franka-teleoperation.gif" 2>/dev/null || true
cp -f "$OUT/thumbnail.png" "$DEMO/ros2-franka-thumbnail.png" 2>/dev/null || true
cp -f "$OUT/sim_metrics.json" "$RESULTS/" 2>/dev/null || true

# Copy verification from earlier successful run if present
FIX=$(ls -d "$DATA"/outputs/fix2_* 2>/dev/null | tail -1 || true)
if [[ -n "$FIX" ]]; then
  cp -f "$FIX"/joint-verification.json "$RESULTS/" 2>/dev/null || true
  cp -f "$FIX"/build.log "$RESULTS/build-summary.txt" 2>/dev/null || true
fi

ls -lh "$DEMO" || true
python3 - <<PY
from pathlib import Path
demo = Path("$DEMO")
for p in sorted(demo.glob("ros2-franka*")):
    mb = p.stat().st_size / 1e6
    print(f"{p.name}: {mb:.2f} MB")
print("RECORD_DONE")
PY
