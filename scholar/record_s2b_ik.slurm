#!/bin/bash
#SBATCH --job-name=v11-s2b-ik
#SBATCH --account=scholar
#SBATCH --partition=scholar-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=/scratch/scholar/edraymon/franka-teleop-data/logs/v11-s2b-%j.out
#SBATCH --error=/scratch/scholar/edraymon/franka-teleop-data/logs/v11-s2b-%j.err
#
# Focused S2-B: MoveIt /compute_ik JointState evidence only.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/scholar/edraymon/franka-teleoperation}"
DATA="${RCAC_SCRATCH:-/scratch/scholar/edraymon}/franka-teleop-data"
SIF="${SIF:-$DATA/containers/ros2_humble_franka.sif}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUT="$DATA/outputs/v11_s2b_${JOB_TAG}"
RESULTS="$REPO_DIR/results/v1.1-hardening/ros2"
mkdir -p "$OUT" "$RESULTS" "$DATA/logs"
export APPTAINER_CACHEDIR="$DATA/ros2-cache/apptainer"
export APPTAINER_TMPDIR="$DATA/ros2-cache/tmp"

git -C "$REPO_DIR" rev-parse HEAD | tee "$OUT/git_sha.txt"

cat > "$OUT/run_s2b.sh" << 'INNER'
#!/bin/bash
set -eo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CC=/usr/bin/gcc CXX=/usr/bin/g++
export AMENT_TRACE_SETUP_FILES=${AMENT_TRACE_SETUP_FILES:-}
export COLCON_TRACE=${COLCON_TRACE:-}
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH LIBRARY_PATH 2>/dev/null || true
source /opt/ros/humble/setup.bash
export PYTHONPATH=/repo/src/vision_arm_control/src:${PYTHONPATH:-}
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export ROS_DOMAIN_ID=55
export QT_QPA_PLATFORM=offscreen
export LIBGL_ALWAYS_SOFTWARE=1

cd /repo/ros2_ws
colcon build --packages-select franka_teleop_ros2 franka_teleop_bringup --symlink-install \
  2>&1 | tee /out/build.log
source install/setup.bash

# Headless MoveIt demo (provides /compute_ik)
setsid bash -c '
  source /opt/ros/humble/setup.bash
  source /repo/ros2_ws/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=55
  export QT_QPA_PLATFORM=offscreen LIBGL_ALWAYS_SOFTWARE=1
  exec ros2 launch franka_teleop_bringup panda_moveit_ik.launch.py use_rviz:=false
' &
MPID=$!

echo "waiting for /compute_ik ..."
for i in $(seq 1 90); do
  if ros2 service list 2>/dev/null | grep -qE '(^| )/compute_ik($| )'; then
    echo "compute_ik up after ${i}s"
    break
  fi
  sleep 1
done
ros2 service list 2>/dev/null | grep -i ik | tee /results/s2b_services.txt || true

# Record bag while evidence script publishes verified joints
rm -rf /results/scenario_s2b_moveit_ik_bag
ros2 bag record -o /results/scenario_s2b_moveit_ik_bag /teleop/ik_joint_states &
BPID=$!
sleep 2

python3 /repo/scripts/s2b_moveit_ik_evidence.py \
  --output /results/s2b_joint_evidence.json \
  --jsonl /results/scenario_s2b_moveit_ik.jsonl \
  --repeat 2 \
  --wait-service 30 \
  2>&1 | tee /out/s2b_evidence_run.txt
EC=${PIPESTATUS[0]}

sleep 1
kill -TERM "$BPID" 2>/dev/null || true
sleep 1
kill -KILL "$BPID" 2>/dev/null || true
kill -TERM -"$MPID" 2>/dev/null || kill -TERM "$MPID" 2>/dev/null || true
sleep 2
kill -KILL -"$MPID" 2>/dev/null || kill -KILL "$MPID" 2>/dev/null || true
pkill -f "move_group|ros2_control|spawner|s2b_moveit" 2>/dev/null || true

# Enrich evidence with bag info
python3 - <<'PY'
import json
from pathlib import Path
import subprocess
p = Path("/results/s2b_joint_evidence.json")
ev = json.loads(p.read_text()) if p.exists() else {}
try:
    info = subprocess.check_output(
        ["bash", "-lc", "source /opt/ros/humble/setup.bash && ros2 bag info /results/scenario_s2b_moveit_ik_bag"],
        text=True, stderr=subprocess.STDOUT,
    )
    ev["bag_info"] = info
except Exception as e:
    ev["bag_info_error"] = str(e)
# count jsonl ok lines
jl = Path("/results/scenario_s2b_moveit_ik.jsonl")
if jl.exists():
    lines = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
    ev["jsonl_n"] = len(lines)
    ev["jsonl_ok"] = sum(1 for r in lines if r.get("ok"))
p.write_text(json.dumps(ev, indent=2) + "\n")
print(json.dumps({k: ev.get(k) for k in ("status","n_samples","n_ok","n_fail","jsonl_ok","jsonl_n")}, indent=2))
PY

ls -lh /results/s2b_joint_evidence.json /results/scenario_s2b_moveit_ik.jsonl | tee /out/s2b_listing.txt
exit "$EC"
INNER
chmod +x "$OUT/run_s2b.sh"

apptainer exec \
  -B "$REPO_DIR:/repo" -B "$DATA:/data" -B "$OUT:/out" -B "$RESULTS:/results" \
  --env SLURM_JOB_ID="$JOB_TAG" \
  "$SIF" bash /out/run_s2b.sh

echo "S2-B job done tag=$JOB_TAG"
cat "$RESULTS/s2b_joint_evidence.json" | head -40
