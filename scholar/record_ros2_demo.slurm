#!/bin/bash
#SBATCH --job-name=ros2-record-v2
#SBATCH --account=scholar
#SBATCH --partition=scholar-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:50:00
#SBATCH --output=/scratch/scholar/edraymon/franka-teleop-data/logs/ros2-record-%j.out
#SBATCH --error=/scratch/scholar/edraymon/franka-teleop-data/logs/ros2-record-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/scholar/edraymon/franka-teleoperation}"
DATA="${RCAC_SCRATCH:-/scratch/scholar/edraymon}/franka-teleop-data"
SIF="${SIF:-$DATA/containers/ros2_humble_franka.sif}"
OUT="$DATA/raw_video/ros2_demo_v2_${SLURM_JOB_ID}"
DEMO="$REPO_DIR/docs/demo"
RESULTS="$REPO_DIR/results/ros2"
mkdir -p "$OUT" "$DEMO" "$RESULTS" "$DATA/logs"

export APPTAINER_CACHEDIR="$DATA/ros2-cache/apptainer"
export APPTAINER_TMPDIR="$DATA/ros2-cache/tmp"

echo "host=$(hostname) job=$SLURM_JOB_ID date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -lh "$SIF"

# 1280x720 fills RViz reliably under software GL (avoids black unused viewport).
WIDTH=1280
HEIGHT=720
FPS=24
RECORD_SEC=15
WARMUP_SEC=16
LOOP_SEC=12

apptainer exec \
  -B "$REPO_DIR:/repo" -B "$DATA:/data" -B "$OUT:/out" -B "$DEMO:/demo" -B "$RESULTS:/results" \
  --env SLURM_JOB_ID="$SLURM_JOB_ID" \
  "$SIF" bash -lc "
set -eo pipefail
# Note: do not enable nounset (-u); ROS setup.bash references optional unbound vars.
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export AMENT_TRACE_SETUP_FILES=\${AMENT_TRACE_SETUP_FILES:-}
export COLCON_TRACE=\${COLCON_TRACE:-}
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH 2>/dev/null || true
source /opt/ros/humble/setup.bash
cd /repo/ros2_ws
# Always rebuild so recording picks up marker/motion changes
colcon build --packages-select franka_teleop_ros2 franka_teleop_bringup
source install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LIBGL_ALWAYS_SOFTWARE=1
export QT_QPA_PLATFORM=xcb
export MESA_GL_VERSION_OVERRIDE=3.3
export DISPLAY=:99

command -v ffmpeg
command -v Xvfb

Xvfb :99 -screen 0 ${WIDTH}x${HEIGHT}x24 -ac +extension GLX +render -noreset &
XPID=\$!
sleep 2

ros2 launch franka_teleop_bringup panda_rviz_sim.launch.py \
  use_rviz:=true diagnostics_out:=/out/sim_metrics.json &
LPID=\$!
sleep ${WARMUP_SEC}

ffmpeg -y -video_size ${WIDTH}x${HEIGHT} -framerate ${FPS} -f x11grab -draw_mouse 0 -i :99.0 \
  -t ${RECORD_SEC} -c:v libx264 -pix_fmt yuv420p -crf 20 /out/raw_demo.mp4

kill \$LPID 2>/dev/null || true
wait \$LPID 2>/dev/null || true
sleep 1
kill \$XPID 2>/dev/null || true
wait \$XPID 2>/dev/null || true
sleep 1
for p in rviz2 robot_state_publisher mock_landmark_publisher human_to_robot_mapper \
         safety_monitor robot_controller target_visualizer diagnostics Xvfb; do
  pkill -x \"\$p\" 2>/dev/null || true
done
ls -lh /out/
# Continue post-process even if ROS children complain on shutdown
set +e
true


# --- Post-process: overlays, mock panel, GIF, contact sheet ---
python3 /repo/scholar/postprocess_demo_v2.py \
  --raw /out/raw_demo.mp4 \
  --out-dir /out \
  --demo-dir /demo \
  --results-dir /results \
  --job-id \"\${SLURM_JOB_ID:-local}\" \
  --repo /repo
"

echo "RECORD_V2_DONE raw=$OUT demo=$DEMO job=$SLURM_JOB_ID"
ls -lh "$DEMO"/ros2-franka-*-v2.* 2>/dev/null || ls -lh "$DEMO"
