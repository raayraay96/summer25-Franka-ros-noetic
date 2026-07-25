#!/bin/bash
#SBATCH --job-name=ros2-sim
#SBATCH --account=scholar
#SBATCH --partition=scholar-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/scholar/edraymon/franka-teleop-data/logs/ros2-sim-%j.out
#SBATCH --error=/scratch/scholar/edraymon/franka-teleop-data/logs/ros2-sim-%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/scholar/edraymon/franka-teleoperation}"
DATA="${RCAC_SCRATCH:-/scratch/scholar/edraymon}/franka-teleop-data"
SIF="${SIF:-$DATA/containers/ros2_humble_franka.sif}"
OUT="$DATA/outputs/ros2_sim_${SLURM_JOB_ID}"
WS="$REPO_DIR/ros2_ws"
RESULTS="$REPO_DIR/results/ros2"
mkdir -p "$OUT" "$RESULTS" "$DATA/logs"

export APPTAINER_CACHEDIR="$DATA/ros2-cache/apptainer"
export APPTAINER_TMPDIR="$DATA/ros2-cache/tmp"

echo "host=$(hostname) job=$SLURM_JOB_ID date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SIF=$SIF"
ls -lh "$SIF"

# Bind repo + data into container
BINDS=(-B "$REPO_DIR:/repo" -B "$DATA:/data" -B "$OUT:/out")

run() {
  apptainer exec "${BINDS[@]}" "$SIF" bash -lc "$*"
}

# Environment metadata
run 'source /opt/ros/humble/setup.bash
python3 - <<PY
import json, platform, os, subprocess
from datetime import datetime, timezone
meta = {
  "hostname": platform.node(),
  "platform": platform.platform(),
  "python": platform.python_version(),
  "date_utc": datetime.now(timezone.utc).isoformat(),
  "slurm_job_id": os.environ.get("SLURM_JOB_ID",""),
  "ros_distro": os.environ.get("ROS_DISTRO","humble"),
  "container": "ros2_humble_franka.sif",
  "simulation_backend": "RViz 2 fake-hardware / joint_state simulation",
  "physical_hardware": False,
}
try:
  meta["ros2_pkg_moveit_panda"] = subprocess.check_output(
    "bash -lc \"source /opt/ros/humble/setup.bash && ros2 pkg prefix moveit_resources_panda_description\"",
    shell=True, text=True).strip()
except Exception as e:
  meta["ros2_pkg_moveit_panda"] = f"error:{e}"
open("/out/environment.json","w").write(json.dumps(meta, indent=2)+"\n")
print(meta)
PY'

# Build workspace
run 'source /opt/ros/humble/setup.bash
set -e
# Prevent Scholar host modules (spack gcc) from leaking into container builds
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH 2>/dev/null || true
cd /repo/ros2_ws
rm -rf build install log || true
colcon build --packages-select franka_teleop_ros2 franka_teleop_bringup 2>&1 | tee /out/build.log
source install/setup.bash
colcon test --packages-select franka_teleop_ros2 --event-handlers console_direct+ 2>&1 | tee /out/test.log || true
colcon test-result --verbose 2>&1 | tee /out/test-result.txt || true
'

# Pure pytest as well
run 'source /opt/ros/humble/setup.bash
cd /repo/ros2_ws/src/franka_teleop_ros2
python3 -m pytest -q test/ 2>&1 | tee /out/pytest.txt
'

# Headless simulation (no RViz) for topic/joint verification
run 'source /opt/ros/humble/setup.bash
source /repo/ros2_ws/install/setup.bash
set -e
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
DIAG=/out/simulation-metrics.json
# Start pipeline without RViz
ros2 launch franka_teleop_bringup panda_rviz_sim.launch.py use_rviz:=false diagnostics_out:=$DIAG &
LPID=$!
sleep 12
# Collect topic evidence
{
  echo "=== nodes ==="
  ros2 node list || true
  echo "=== topics ==="
  ros2 topic list || true
  echo "=== target hz ==="
  timeout 5 ros2 topic hz /teleop/target_pose || true
  echo "=== command hz ==="
  timeout 5 ros2 topic hz /teleop/command_pose || true
  echo "=== joint_states hz ==="
  timeout 5 ros2 topic hz /joint_states || true
  echo "=== joint_states sample ==="
  timeout 3 ros2 topic echo /joint_states --once || true
  echo "=== safety status ==="
  timeout 3 ros2 topic echo /teleop/safety_status --once || true
} | tee /out/topic-evidence.txt

# Capture joint travel over 12s (one demo period)
python3 - <<PY
import json, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

class Cap(Node):
    def __init__(self):
        super().__init__("cap")
        self.js = []
        self.cmd = 0
        self.tgt = 0
        self.safety = []
        self.create_subscription(JointState, "/joint_states", self._j, 10)
        self.create_subscription(PoseStamped, "/teleop/command_pose", self._c, 10)
        self.create_subscription(PoseStamped, "/teleop/target_pose", self._t, 10)
        self.create_subscription(String, "/teleop/safety_status", self._s, 10)
    def _j(self, m):
        if m.position:
            self.js.append(list(m.position[:7]))
    def _c(self, m):
        self.cmd += 1
    def _t(self, m):
        self.tgt += 1
    def _s(self, m):
        self.safety.append(m.data)

rclpy.init()
n = Cap()
end = time.time() + 12.0
while time.time() < end:
    rclpy.spin_once(n, timeout_sec=0.1)
path_travel = 0.0
start_end = 0.0
max_exc = 0.0
if len(n.js) >= 2:
    for a, b in zip(n.js, n.js[1:]):
        path_travel += sum(abs(bi - ai) for ai, bi in zip(a, b))
    start_end = sum(abs(b - a) for a, b in zip(n.js[0], n.js[-1]))
    for j in range(len(n.js[0])):
        series = [s[j] for s in n.js]
        max_exc = max(max_exc, max(series) - min(series))
# Safety: flag unexpected e-stop / timeout strings
bad = [s for s in n.safety if any(k in s for k in ("emergency", "estop", "timeout", "stale", "joint_limit"))]
out = {
  "joint_samples": len(n.js),
  "joint_travel_rad": path_travel,
  "joint_path_travel_rad": path_travel,
  "joint_start_end_travel_rad": start_end,
  "max_single_joint_excursion_rad": max_exc,
  "joints_moved": path_travel > 0.05,
  "command_pose_count": n.cmd,
  "target_pose_count": n.tgt,
  "safety_status_samples": len(n.safety),
  "unexpected_safety_flags": bad[:20],
  "simulation_backend": "RViz 2 fake-hardware / joint_state simulation",
  "physical_hardware": False,
  "depth_used_for_control": False,
  "ik_type": "geometric_approximation",
}
open("/out/joint-verification.json","w").write(json.dumps(out, indent=2)+"\n")
print(out)
n.destroy_node(); rclpy.shutdown()
PY

# Cleanup launch without broad pkill (avoids killing the container bash / SIGTERM 15)
kill $LPID 2>/dev/null || true
wait $LPID 2>/dev/null || true
sleep 1
# Best-effort node teardown by exact executable name only
for p in mock_landmark_publisher human_to_robot_mapper safety_monitor \
         robot_controller target_visualizer diagnostics robot_state_publisher rviz2; do
  pkill -x "$p" 2>/dev/null || true
done
sleep 1
true
'

# Ensure copy runs even if container reported a non-zero from background teardown
set +e

# Copy lightweight results into git tree
cp -f "$OUT"/*.json "$RESULTS/" 2>/dev/null || true
cp -f "$OUT"/topic-evidence.txt "$RESULTS/" 2>/dev/null || true
cp -f "$OUT"/build.log "$RESULTS/build-summary.txt" 2>/dev/null || true
cp -f "$OUT"/pytest.txt "$RESULTS/test-summary.txt" 2>/dev/null || true
cp -f "$OUT"/environment.json "$RESULTS/" 2>/dev/null || true

echo "SIM_DONE out=$OUT results=$RESULTS"
ls -la "$OUT"
