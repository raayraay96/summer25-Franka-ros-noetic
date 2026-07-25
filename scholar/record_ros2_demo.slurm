#!/bin/bash
#SBATCH --job-name=ros2-record
#SBATCH --account=scholar
#SBATCH --partition=scholar-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/scholar/edraymon/franka-teleop-data/logs/ros2-record-%j.out
#SBATCH --error=/scratch/scholar/edraymon/franka-teleop-data/logs/ros2-record-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/scholar/edraymon/franka-teleoperation}"
DATA="${RCAC_SCRATCH:-/scratch/scholar/edraymon}/franka-teleop-data"
SIF="${SIF:-$DATA/containers/ros2_humble_franka.sif}"
OUT="$DATA/raw_video/ros2_demo_${SLURM_JOB_ID}"
WS="$REPO_DIR/ros2_ws"
DEMO="$REPO_DIR/docs/demo"
mkdir -p "$OUT" "$DEMO" "$DATA/logs"

export APPTAINER_CACHEDIR="$DATA/ros2-cache/apptainer"
export APPTAINER_TMPDIR="$DATA/ros2-cache/tmp"
export APPTAINER_BINDPATH="$REPO_DIR:/repo,$DATA:/data,$OUT:/out"

echo "host=$(hostname) job=$SLURM_JOB_ID date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"

apptainer exec \
  -B "$REPO_DIR:/repo" -B "$DATA:/data" -B "$OUT:/out" \
  "$SIF" bash -lc '
set -e
source /opt/ros/humble/setup.bash
source /repo/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LIBGL_ALWAYS_SOFTWARE=1
export QT_QPA_PLATFORM=xcb
export MESA_GL_VERSION_OVERRIDE=3.3

# Start virtual display
Xvfb :99 -screen 0 1280x720x24 -ac +extension GLX +render -noreset &
XPID=$!
export DISPLAY=:99
sleep 2

# Launch simulation with RViz
ros2 launch franka_teleop_bringup panda_rviz_sim.launch.py use_rviz:=true diagnostics_out:=/out/sim_metrics.json &
LPID=$!
sleep 8

# Record 16 seconds of the display
ffmpeg -y -video_size 1280x720 -framerate 15 -f x11grab -i :99.0 \
  -t 16 -c:v libx264 -pix_fmt yuv420p -crf 28 /out/raw_demo.mp4

# Also grab a thumbnail
ffmpeg -y -ss 00:00:08 -i /out/raw_demo.mp4 -frames:v 1 /out/thumbnail.png

# Overlay label burn-in for portfolio clarity
ffmpeg -y -i /out/raw_demo.mp4 \
  -vf "drawtext=text='\''ROS 2 Humble | RViz fake-hardware | No physical Franka | Scholar'\'':fontcolor=white:fontsize=22:box=1:boxcolor=black@0.5:x=20:y=20" \
  -c:v libx264 -pix_fmt yuv420p -crf 28 -t 16 /out/labeled_demo.mp4

# GIF (lightweight)
ffmpeg -y -i /out/labeled_demo.mp4 -vf "fps=10,scale=720:-1:flags=lanczos" -t 12 /out/demo.gif

# Cleanup
kill $LPID 2>/dev/null || true
kill $XPID 2>/dev/null || true
sleep 2
pkill -f rviz2 || true
pkill -f franka_teleop || true
pkill -f robot_state_publisher || true
pkill -f Xvfb || true

ls -lh /out/
'

# Copy compressed media into git tree (size check)
cp -f "$OUT/labeled_demo.mp4" "$DEMO/ros2-franka-teleoperation.mp4"
cp -f "$OUT/demo.gif" "$DEMO/ros2-franka-teleoperation.gif"
cp -f "$OUT/thumbnail.png" "$DEMO/ros2-franka-thumbnail.png"
ls -lh "$DEMO"
# Fail if assets are huge
python3 - <<PY
from pathlib import Path
demo = Path("$DEMO")
for p in demo.glob("ros2-franka*"):
    mb = p.stat().st_size / 1e6
    print(p.name, f"{mb:.1f} MB")
    if p.suffix == ".gif" and mb > 20:
        raise SystemExit(f"GIF too large: {mb}")
    if p.suffix == ".mp4" and mb > 40:
        raise SystemExit(f"MP4 too large: {mb}")
print("media size OK")
PY

echo "RECORD_DONE raw=$OUT demo=$DEMO"
