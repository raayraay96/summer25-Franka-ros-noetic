#!/bin/bash
set -euo pipefail
SIF=/scratch/scholar/edraymon/franka-teleop-data/containers/ros2_humble_franka.sif
REPO=/scratch/scholar/edraymon/franka-teleoperation
OUT=/scratch/scholar/edraymon/franka-teleop-data/outputs/fix2_${SLURM_JOB_ID}
mkdir -p "$OUT"
apptainer exec -B "$REPO:/repo" -B "$OUT:/out" "$SIF" bash -lc '
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export CC=/usr/bin/gcc CXX=/usr/bin/g++
source /opt/ros/humble/setup.bash
cd /repo/ros2_ws
rm -rf build install log || true
colcon build --packages-select franka_teleop_ros2 franka_teleop_bringup 2>&1 | tee /out/build.log
source install/setup.bash
ros2 pkg prefix franka_teleop_bringup
ros2 pkg prefix franka_teleop_ros2
ros2 launch franka_teleop_bringup mock_pipeline.launch.py &
LPID=$!
sleep 10
echo NODES; ros2 node list || true
echo TOPICS; ros2 topic list || true
timeout 5 ros2 topic hz /teleop/target_pose 2>&1 | tee /out/hz_target.txt || true
timeout 5 ros2 topic hz /teleop/command_pose 2>&1 | tee /out/hz_cmd.txt || true
timeout 5 ros2 topic hz /joint_states 2>&1 | tee /out/hz_js.txt || true
timeout 3 ros2 topic echo /joint_states --once 2>&1 | tee /out/js_once.txt || true
python3 - <<PY
import json, time
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped
class Cap(Node):
    def __init__(self):
        super().__init__("cap")
        self.js=[]; self.cmd=0; self.tgt=0
        self.create_subscription(JointState, "/joint_states", self._j, 10)
        self.create_subscription(PoseStamped, "/teleop/command_pose", self._c, 10)
        self.create_subscription(PoseStamped, "/teleop/target_pose", self._t, 10)
    def _j(self,m):
        if m.position: self.js.append(list(m.position[:7]))
    def _c(self,m): self.cmd += 1
    def _t(self,m): self.tgt += 1
rclpy.init(); n=Cap(); end=time.time()+8
while time.time()<end: rclpy.spin_once(n, timeout_sec=0.05)
travel=0.0
if len(n.js)>=2: travel=sum(abs(b-a) for a,b in zip(n.js[0], n.js[-1]))
out={"joint_samples":len(n.js),"joint_travel_rad":travel,"joints_moved":travel>0.05,"command_pose_count":n.cmd,"target_pose_count":n.tgt,"physical_hardware":False,"simulation_backend":"RViz 2 fake-hardware / joint_state simulation"}
open("/out/joint-verification.json","w").write(json.dumps(out,indent=2)+"\n")
print(out)
n.destroy_node(); rclpy.shutdown()
PY
kill $LPID 2>/dev/null || true
wait $LPID 2>/dev/null || true
echo FIX2_OK
'
