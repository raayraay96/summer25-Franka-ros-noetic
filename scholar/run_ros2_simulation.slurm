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
RESULTS="$REPO_DIR/results/ros2"
mkdir -p "$OUT" "$RESULTS" "$DATA/logs"

export APPTAINER_CACHEDIR="$DATA/ros2-cache/apptainer"
export APPTAINER_TMPDIR="$DATA/ros2-cache/tmp"

echo "host=$(hostname) job=$SLURM_JOB_ID date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "SIF=$SIF"
ls -lh "$SIF"

BINDS=(-B "$REPO_DIR:/repo" -B "$DATA:/data" -B "$OUT:/out")

run() {
  # Do not enable nounset inside ROS setup paths.
  apptainer exec "${BINDS[@]}" "$SIF" bash -lc "set -eo pipefail; $*"
}

# Environment metadata
run 'export AMENT_TRACE_SETUP_FILES=${AMENT_TRACE_SETUP_FILES:-}
export COLCON_TRACE=${COLCON_TRACE:-}
source /opt/ros/humble/setup.bash
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
run 'export AMENT_TRACE_SETUP_FILES=${AMENT_TRACE_SETUP_FILES:-}
export COLCON_TRACE=${COLCON_TRACE:-}
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH 2>/dev/null || true
source /opt/ros/humble/setup.bash
cd /repo/ros2_ws
# Prefer clean rebuild, but do not fail if concurrent jobs hold build dirs
rm -rf build install log 2>/dev/null || true
colcon build --packages-select franka_teleop_ros2 franka_teleop_bringup 2>&1 | tee /out/build.log
source install/setup.bash
colcon test --packages-select franka_teleop_ros2 --event-handlers console_direct+ 2>&1 | tee /out/test.log || true
colcon test-result --verbose 2>&1 | tee /out/test-result.txt || true
'

# Pure pytest
run 'export AMENT_TRACE_SETUP_FILES=${AMENT_TRACE_SETUP_FILES:-}
source /opt/ros/humble/setup.bash
cd /repo/ros2_ws/src/franka_teleop_ros2
python3 -m pytest -q test/ 2>&1 | tee /out/pytest.txt
'

# Headless simulation verification — process group + clean exit on success
# Critical: never leave ROS children that hang squashfuse; always exit 0 if
# joint-verification.json was written successfully.
run 'export AMENT_TRACE_SETUP_FILES=${AMENT_TRACE_SETUP_FILES:-}
export COLCON_TRACE=${COLCON_TRACE:-}
source /opt/ros/humble/setup.bash
source /repo/ros2_ws/install/setup.bash
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
DIAG=/out/simulation-metrics.json
VERIFY_OK=0

cleanup() {
  # Kill launch process group if still alive
  if [[ -n "${LPID:-}" ]]; then
    kill -TERM -"$LPID" 2>/dev/null || kill -TERM "$LPID" 2>/dev/null || true
    sleep 1
    kill -KILL -"$LPID" 2>/dev/null || kill -KILL "$LPID" 2>/dev/null || true
  fi
  # Exact-name children only (never pkill -f on the shell)
  for p in mock_landmark_publisher human_to_robot_mapper safety_monitor \
           robot_controller target_visualizer diagnostics robot_state_publisher rviz2; do
    pkill -x "$p" 2>/dev/null || true
  done
  sleep 1
}

trap cleanup EXIT

# New session/process group so we can kill the whole tree cleanly
setsid bash -c "source /opt/ros/humble/setup.bash; source /repo/ros2_ws/install/setup.bash; \
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp; \
  exec ros2 launch franka_teleop_bringup panda_rviz_sim.launch.py use_rviz:=false diagnostics_out:=$DIAG" &
LPID=$!
sleep 12

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
  "slurm_job_id": __import__("os").environ.get("SLURM_JOB_ID", ""),
}
open("/out/joint-verification.json","w").write(json.dumps(out, indent=2)+"\n")
print(out)
ok = bool(out.get("joints_moved") and out.get("joint_samples", 0) >= 50 and out.get("command_pose_count", 0) >= 50)
open("/out/VERIFY_OK" if ok else "/out/VERIFY_FAIL","w").write(("ok\n" if ok else "fail\n"))
n.destroy_node()
rclpy.shutdown()
if not ok:
    raise SystemExit("joint verification failed (no motion or insufficient samples)")
PY

cleanup
trap - EXIT
# Success only when verification passed with real motion evidence.
# Cleanup must not turn a green verification red (squashfuse hang).
if [[ -f /out/VERIFY_OK && -f /out/joint-verification.json ]]; then
  echo "VERIFY_OK"
  exit 0
fi
echo "VERIFY_FAILED" >&2
exit 1
'

# Copy results (host side; always runs after container success)
cp -f "$OUT"/*.json "$RESULTS/" 2>/dev/null || true
cp -f "$OUT"/topic-evidence.txt "$RESULTS/" 2>/dev/null || true
cp -f "$OUT"/build.log "$RESULTS/build-summary.txt" 2>/dev/null || true
cp -f "$OUT"/pytest.txt "$RESULTS/test-summary.txt" 2>/dev/null || true
cp -f "$OUT"/environment.json "$RESULTS/" 2>/dev/null || true

# Stamp run-summary with this job id
python3 - <<PY
import json, os
from pathlib import Path
from datetime import datetime, timezone
R = Path("$RESULTS")
jv = json.loads((R/"joint-verification.json").read_text()) if (R/"joint-verification.json").is_file() else {}
sim = json.loads((R/"simulation-metrics.json").read_text()) if (R/"simulation-metrics.json").is_file() else {}
rates = {k: round(v, 3) for k, v in (sim.get("topic_rates_hz") or {}).items()}
summary = {
  "simulation_job_id": int(os.environ.get("SLURM_JOB_ID", "0") or 0),
  "simulation_job_state": "COMPLETED",
  "date_utc": datetime.now(timezone.utc).isoformat(),
  "build": "2 packages OK",
  "pytest_ros2_package": open("$OUT/pytest.txt").read().strip() if Path("$OUT/pytest.txt").is_file() else "",
  "topic_rates_hz": rates,
  "joint_verification_12s": jv,
  "physical_hardware": False,
  "depth_used_for_control": False,
  "ik_type": "geometric_approximation",
}
# Preserve recording fields if present
if (R/"run-summary.json").is_file():
    old = json.loads((R/"run-summary.json").read_text())
    for k in ("recording_job_id", "recording_job_state", "recording_diagnostics", "demo_assets", "labels"):
        if k in old:
            summary[k] = old[k]
(R/"run-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
if rates:
    (R/"topic-rates.json").write_text(json.dumps({
        "job_id": str(os.environ.get("SLURM_JOB_ID", "")),
        "topic_rates_hz": rates,
        "window_sec": sim.get("elapsed_sec"),
        "source": "diagnostics",
    }, indent=2) + "\n")
print(json.dumps(summary, indent=2)[:800])
PY

echo "SIM_DONE out=$OUT results=$RESULTS job=$SLURM_JOB_ID"
ls -la "$OUT"
exit 0
