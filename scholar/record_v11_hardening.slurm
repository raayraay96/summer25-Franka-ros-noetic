#!/bin/bash
#SBATCH --job-name=v11-s2-gates
#SBATCH --account=scholar
#SBATCH --partition=scholar-cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/scholar/edraymon/franka-teleop-data/logs/v11-s2-%j.out
#SBATCH --error=/scratch/scholar/edraymon/franka-teleop-data/logs/v11-s2-%j.err
#
# Completes PR #5 environment-bound gates on Scholar:
#   S2-A  RViz fake-hardware recording of the hardened teleop pipeline
#   S2-B  MoveIt /compute_ik JointState evidence
#
# dry_run only; no physical Franka.
set -euo pipefail

REPO_DIR="${REPO_DIR:-/scratch/scholar/edraymon/franka-teleoperation}"
DATA="${RCAC_SCRATCH:-/scratch/scholar/edraymon}/franka-teleop-data"
SIF="${SIF:-$DATA/containers/ros2_humble_franka.sif}"
JOB_TAG="${SLURM_JOB_ID:-manual}"
OUT_SCRATCH="$DATA/outputs/v11_hardening_${JOB_TAG}"
RESULTS_REPO="$REPO_DIR/results/v1.1-hardening/ros2"
MEDIA_REPO="$REPO_DIR/docs/media/v1.1-hardening"
mkdir -p "$OUT_SCRATCH" "$RESULTS_REPO" "$MEDIA_REPO" "$DATA/logs" \
         "$DATA/ros2-cache/apptainer" "$DATA/ros2-cache/tmp"

export APPTAINER_CACHEDIR="$DATA/ros2-cache/apptainer"
export APPTAINER_TMPDIR="$DATA/ros2-cache/tmp"

WIDTH=1280
HEIGHT=720
FPS=20
SCENARIO_SEC="${SCENARIO_SEC:-16}"
WARMUP_SEC="${WARMUP_SEC:-14}"
RECORD_SEC="${RECORD_SEC:-12}"

echo "host=$(hostname) job=$JOB_TAG date=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
ls -lh "$SIF"
git -C "$REPO_DIR" rev-parse HEAD | tee "$OUT_SCRATCH/git_sha.txt"

# Single container entrypoint script (avoids nested quoting hell)
cat > "$OUT_SCRATCH/run_inside.sh" << 'INNER'
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

cd /repo

echo "=== 1. Pure-Python parity ==="
python3 -m pip install -q -r requirements-ci.txt --user 2>/dev/null || true
python3 -m pytest -q tests/ 2>&1 | tee /out/pytest.txt
python3 scripts/qp_cross_validation.py --generic 200 --cbf 200 2>&1 | tee /out/qp_smoke.txt
python3 scripts/validate_config.py 2>&1 | tee /out/validate_config.txt

echo "=== 2. colcon build ==="
cd /repo/ros2_ws
rm -rf build install log 2>/dev/null || true
colcon build --packages-select franka_teleop_ros2 franka_teleop_bringup --symlink-install \
  2>&1 | tee /out/build.log
source install/setup.bash
echo build_ok > /out/build_ok.txt

# ---------------------------------------------------------------------------
echo "=== 3. S2-A scenarios (bags + diagnostics) ==="
export ROS_DOMAIN_ID=41
mkdir -p /results
DUR="${SCENARIO_SEC:-16}"

run_scenario () {
  local name="$1" strategy="$2" safety="$3" ik="$4" obstacles="$5"
  local jsonl="/results/${name}.jsonl"
  local bag="/results/${name}_bag"
  rm -rf "$bag" "$jsonl"
  echo "=== scenario ${name}: ${strategy}+${safety} ik=${ik} ==="
  ros2 launch franka_teleop_bringup teleop_pipeline_demo.launch.py \
      strategy:="${strategy}" safety:="${safety}" ik_backend:="${ik}" \
      obstacles:="${obstacles}" diagnostics_jsonl:="${jsonl}" &
  local lpid=$!
  sleep 4
  timeout "${DUR}" ros2 bag record -o "$bag" \
      /teleop/target_pose /teleop/command_pose /joint_states \
      /teleop/ik_joint_states /teleop/safety_status /teleop/gate_status \
      /teleop/diagnostics || true
  kill -TERM "$lpid" 2>/dev/null || true
  sleep 2
  pkill -f "teleop_pipeline|mock_landmark|diagnostics" 2>/dev/null || true
  sleep 1
  if [[ -f "$jsonl" ]]; then
    python3 /repo/scripts/extract_run_metrics.py "$jsonl" \
      --output "/results/${name}_metrics.json" || true
  fi
}

OBS_NONE='[]'
OBS_DEMO='[{"id":"demo_sphere","center":[0.48,0.0,0.35],"radius_m":0.10,"margin_m":0.05}]'

run_scenario "scenario_a_shoulder_cbf"    shoulder_relative cbf_qp none "$OBS_NONE"
run_scenario "scenario_b_sew_cbf"         sew_orientation   cbf_qp geometric_unverified "$OBS_NONE"
run_scenario "scenario_c_obstacle_interv" shoulder_relative cbf_qp none "$OBS_DEMO"
echo S2A_scenarios_ok > /out/s2a_scenarios_ok.txt

# ---------------------------------------------------------------------------
echo "=== 4. S2-A RViz recording ==="
export ROS_DOMAIN_ID=42
export LIBGL_ALWAYS_SOFTWARE=1
export QT_QPA_PLATFORM=xcb
export MESA_GL_VERSION_OVERRIDE=3.3
export QT_AUTO_SCREEN_SCALE_FACTOR=0
export QT_SCALE_FACTOR=1
export DISPLAY=:99

W="${WIDTH:-1280}"
H="${HEIGHT:-720}"
F="${FPS:-20}"
WARM="${WARMUP_SEC:-14}"
REC="${RECORD_SEC:-12}"

Xvfb :99 -screen 0 ${W}x${H}x24 -ac +extension GLX +render -noreset &
XPID=$!
sleep 2

setsid bash -c '
  source /opt/ros/humble/setup.bash
  source /repo/ros2_ws/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp LIBGL_ALWAYS_SOFTWARE=1
  export QT_QPA_PLATFORM=xcb MESA_GL_VERSION_OVERRIDE=3.3 DISPLAY=:99
  export QT_AUTO_SCREEN_SCALE_FACTOR=0 QT_SCALE_FACTOR=1
  export PYTHONPATH=/repo/src/vision_arm_control/src:${PYTHONPATH:-}
  export ROS_DOMAIN_ID=42
  exec ros2 launch franka_teleop_bringup teleop_pipeline_rviz.launch.py \
    strategy:=shoulder_relative safety:=cbf_qp ik_backend:=geometric_unverified \
    use_rviz:=true diagnostics_jsonl:=/results/rviz_demo.jsonl
' &
LPID=$!
sleep "$WARM"

ffmpeg -y -video_size ${W}x${H} -framerate ${F} -f x11grab -draw_mouse 0 -i :99.0 \
  -t ${REC} -c:v libx264 -pix_fmt yuv420p -crf 20 /out/raw_rviz.mp4 || true

kill -TERM -"$LPID" 2>/dev/null || kill -TERM "$LPID" 2>/dev/null || true
sleep 2
kill -KILL -"$LPID" 2>/dev/null || kill -KILL "$LPID" 2>/dev/null || true
pkill -f "rviz2|teleop_pipeline|mock_landmark|robot_state_publisher" 2>/dev/null || true
kill -TERM "$XPID" 2>/dev/null || true
sleep 1

JOB="${SLURM_JOB_ID:-local}"
ffmpeg -y -i /out/raw_rviz.mp4 \
  -vf "drawtext=text='ROS 2 SIMULATION | MOCK LANDMARKS | PANDA FAKE HARDWARE | NO PHYSICAL ROBOT':fontcolor=white:fontsize=18:box=1:boxcolor=black@0.6:x=12:y=12,drawtext=text='RETARGETER=shoulder_relative | SAFETY FILTER=cbf_qp | GATE=live on /teleop/diagnostics':fontcolor=white:fontsize=16:box=1:boxcolor=black@0.55:x=12:y=40,drawtext=text='dry_run | Scholar job ${JOB} | no physical Franka':fontcolor=yellow:fontsize=14:box=1:boxcolor=black@0.55:x=12:y=68" \
  -c:v libx264 -pix_fmt yuv420p -crf 20 /media/end-to-end-rviz.mp4 || true

ffmpeg -y -i /media/end-to-end-rviz.mp4 -vf "fps=10,scale=960:-1:flags=lanczos" \
  -t 12 /media/end-to-end-rviz.gif || true
ffmpeg -y -ss 00:00:05 -i /media/end-to-end-rviz.mp4 -frames:v 1 /media/end-to-end-rviz-thumb.png || true

if [[ -f /results/rviz_demo.jsonl ]]; then
  python3 /repo/scripts/extract_run_metrics.py /results/rviz_demo.jsonl \
    --output /results/rviz_demo_metrics.json || true
fi
echo S2A_rviz_ok > /out/s2a_rviz_ok.txt
ls -lh /media/ /results/ | tee /out/s2a_listing.txt

# ---------------------------------------------------------------------------
echo "=== 5. S2-B MoveIt compute_ik ==="
export ROS_DOMAIN_ID=43
export QT_QPA_PLATFORM=offscreen
export LIBGL_ALWAYS_SOFTWARE=1

# Prefer our headless wrapper; fall back to upstream demo with use_rviz:=false
setsid bash -c '
  source /opt/ros/humble/setup.bash
  source /repo/ros2_ws/install/setup.bash
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=43
  export LIBGL_ALWAYS_SOFTWARE=1 QT_QPA_PLATFORM=offscreen
  if ros2 launch franka_teleop_bringup panda_moveit_ik.launch.py use_rviz:=false; then
    exit 0
  fi
  exec ros2 launch moveit_resources_panda_moveit_config demo.launch.py use_rviz:=false
' &
MPID=$!

echo "waiting for /compute_ik ..."
IK_UP=0
for i in $(seq 1 90); do
  if ros2 service list 2>/dev/null | grep -qE '(^| )/compute_ik($| )'; then
    echo "compute_ik up after ${i}s"
    IK_UP=1
    break
  fi
  sleep 1
done
ros2 service list 2>/dev/null | grep -i ik | tee /results/s2b_services.txt || true

if [[ "$IK_UP" -ne 1 ]]; then
  echo "BLOCKER: /compute_ik never appeared" | tee /results/s2b_blocker.txt
  kill -TERM -"$MPID" 2>/dev/null || kill -TERM "$MPID" 2>/dev/null || true
else
  # Pipeline with moveit IK; do not own /joint_states (move_group does).
  setsid bash -c '
    source /opt/ros/humble/setup.bash
    source /repo/ros2_ws/install/setup.bash
    export RMW_IMPLEMENTATION=rmw_fastrtps_cpp ROS_DOMAIN_ID=43
    export PYTHONPATH=/repo/src/vision_arm_control/src:${PYTHONPATH:-}
    ros2 run franka_teleop_ros2 mock_landmark_publisher --ros-args \
      -p topic:=/perception/landmarks &
    exec ros2 run franka_teleop_ros2 teleop_pipeline_node --ros-args \
      -p retargeting_strategy:=shoulder_relative \
      -p safety_filter:=cbf_qp \
      -p ik_backend:=moveit \
      -p publish_joint_states:=false \
      -p diagnostics_jsonl:=/results/scenario_s2b_moveit_ik.jsonl \
      -p obstacles:="[]"
  ' &
  TPID=$!
  sleep 5

  rm -rf /results/scenario_s2b_moveit_ik_bag
  timeout "${DUR}" ros2 bag record -o /results/scenario_s2b_moveit_ik_bag \
    /teleop/target_pose /teleop/command_pose /teleop/ik_joint_states \
    /teleop/safety_status /teleop/gate_status /teleop/diagnostics || true

  kill -TERM -"$TPID" 2>/dev/null || kill -TERM "$TPID" 2>/dev/null || true
  sleep 1
  kill -KILL -"$TPID" 2>/dev/null || kill -KILL "$TPID" 2>/dev/null || true
fi

kill -TERM -"$MPID" 2>/dev/null || kill -TERM "$MPID" 2>/dev/null || true
sleep 2
kill -KILL -"$MPID" 2>/dev/null || kill -KILL "$MPID" 2>/dev/null || true
pkill -f "move_group|ros2_control|spawner|teleop_pipeline|mock_landmark" 2>/dev/null || true

if [[ -f /results/scenario_s2b_moveit_ik.jsonl ]]; then
  python3 /repo/scripts/extract_run_metrics.py /results/scenario_s2b_moveit_ik.jsonl \
    --output /results/scenario_s2b_moveit_ik_metrics.json || true
fi

python3 - <<'PY'
import json, subprocess
from pathlib import Path

evidence = {
    "gate": "S2-B",
    "service": "/compute_ik",
    "ik_backend": "moveit",
    "verified_topic": "/teleop/ik_joint_states",
    "frame_id_expected": "verified_moveit",
    "samples": [],
    "n_samples": 0,
    "status": "no_samples",
}
bag_dir = Path("/results/scenario_s2b_moveit_ik_bag")
jsonl = Path("/results/scenario_s2b_moveit_ik.jsonl")
if jsonl.exists():
    ok = fail = 0
    last = None
    for line in jsonl.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        ok = rec.get("ik_ok_count", ok)
        fail = rec.get("ik_fail_count", fail)
        last = rec
    evidence["diagnostics_ik_ok_count"] = ok
    evidence["diagnostics_ik_fail_count"] = fail
    evidence["last_diagnostics"] = {
        k: last.get(k) for k in (
            "ik_backend", "ik_ok_count", "ik_fail_count", "retargeting_strategy",
            "safety_mode", "gate_state", "solver_status", "accepted"
        ) if last and k in last
    } if last else None

# Try ros2 bag via CLI (rosbags may not be installed)
if bag_dir.exists():
    try:
        info = subprocess.check_output(
            ["bash", "-lc",
             "source /opt/ros/humble/setup.bash && ros2 bag info /results/scenario_s2b_moveit_ik_bag"],
            text=True, stderr=subprocess.STDOUT,
        )
        evidence["bag_info"] = info
    except Exception as e:
        evidence["bag_info_error"] = str(e)
    # Use ros2 bag convert / python reader if available
    try:
        from rosbags.rosbag2 import Reader
        from rosbags.serde import deserialize_cdr
        with Reader(bag_dir) as reader:
            for conn, ts, raw in reader.messages():
                if conn.topic != "/teleop/ik_joint_states":
                    continue
                msg = deserialize_cdr(raw, conn.msgtype)
                evidence["samples"].append({
                    "t_ns": int(ts),
                    "frame_id": getattr(msg.header, "frame_id", ""),
                    "name": list(msg.name),
                    "position": [float(x) for x in msg.position],
                })
                if len(evidence["samples"]) >= 25:
                    break
    except Exception as e:
        evidence["rosbags_error"] = str(e)
        # Fallback: sqlite3 metadata only
        try:
            import sqlite3
            dbs = list(bag_dir.glob("*.db3"))
            if dbs:
                evidence["bag_db"] = dbs[0].name
                con = sqlite3.connect(str(dbs[0]))
                cur = con.cursor()
                # topics table
                try:
                    rows = cur.execute("SELECT name, type FROM topics").fetchall()
                    evidence["bag_topics"] = [{"name": r[0], "type": r[1]} for r in rows]
                    # count messages per topic id
                    counts = cur.execute(
                        "SELECT topic_id, COUNT(*) FROM messages GROUP BY topic_id"
                    ).fetchall()
                    evidence["bag_message_counts"] = {str(t): c for t, c in counts}
                except Exception as e2:
                    evidence["sqlite_error"] = str(e2)
                con.close()
        except Exception as e3:
            evidence["sqlite_fallback_error"] = str(e3)

evidence["n_samples"] = len(evidence["samples"])
if evidence["n_samples"] > 0 and all(
    s.get("frame_id") == "verified_moveit" for s in evidence["samples"]
):
    evidence["status"] = "verified_moveit_jointstate_evidence"
elif evidence.get("diagnostics_ik_ok_count", 0) > 0:
    evidence["status"] = "ik_ok_in_diagnostics"
elif Path("/results/s2b_blocker.txt").exists():
    evidence["status"] = "blocked_compute_ik_missing"
else:
    evidence["status"] = "no_verified_joints_recorded"

Path("/results/s2b_joint_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
print(json.dumps({
    k: evidence.get(k)
    for k in ("status", "n_samples", "diagnostics_ik_ok_count", "diagnostics_ik_fail_count")
}, indent=2))
PY

echo S2B_ok > /out/s2b_ok.txt
ls -lh /results/ | tee /out/s2b_listing.txt

# ---------------------------------------------------------------------------
echo "=== 6. Environment summary ==="
python3 - <<'PY'
import json, platform, os
from pathlib import Path
from datetime import datetime, timezone
meta = {
  "hostname": platform.node(),
  "platform": platform.platform(),
  "python": platform.python_version(),
  "date_utc": datetime.now(timezone.utc).isoformat(),
  "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
  "ros_distro": "humble",
  "container": "ros2_humble_franka.sif",
  "simulation_backend": "RViz 2 fake-hardware + MoveIt compute_ik",
  "physical_hardware": False,
  "git_sha": Path("/out/git_sha.txt").read_text().strip() if Path("/out/git_sha.txt").exists() else "",
  "gates": ["S2-A", "S2-B"],
  "artifacts": sorted(p.name for p in Path("/results").iterdir()) if Path("/results").exists() else [],
}
Path("/results/environment.json").write_text(json.dumps(meta, indent=2) + "\n")
Path("/out/environment.json").write_text(json.dumps(meta, indent=2) + "\n")
print(json.dumps(meta, indent=2))
PY

echo "=== INNER COMPLETE ==="
INNER

chmod +x "$OUT_SCRATCH/run_inside.sh"

apptainer exec \
  -B "$REPO_DIR:/repo" \
  -B "$DATA:/data" \
  -B "$OUT_SCRATCH:/out" \
  -B "$RESULTS_REPO:/results" \
  -B "$MEDIA_REPO:/media" \
  --env SLURM_JOB_ID="$JOB_TAG" \
  --env SCENARIO_SEC="$SCENARIO_SEC" \
  --env WARMUP_SEC="$WARMUP_SEC" \
  --env RECORD_SEC="$RECORD_SEC" \
  --env WIDTH="$WIDTH" \
  --env HEIGHT="$HEIGHT" \
  --env FPS="$FPS" \
  "$SIF" bash /out/run_inside.sh

# Host-side copies
cp -f "$OUT_SCRATCH"/*.txt "$RESULTS_REPO/" 2>/dev/null || true
cp -f "$OUT_SCRATCH"/environment.json "$RESULTS_REPO/" 2>/dev/null || true
cp -f "$OUT_SCRATCH"/git_sha.txt "$RESULTS_REPO/" 2>/dev/null || true

# Media README
cat > "$MEDIA_REPO/README.md" << EOF
# v1.1-hardening end-to-end RViz evidence (S2-A)

Independent post-program engineering by Eric Raymond (2026).

| Item | Value |
|------|--------|
| Job | $JOB_TAG |
| Host | Scholar (container: ros2_humble_franka.sif) |
| Commit | $(cat "$OUT_SCRATCH/git_sha.txt" 2>/dev/null || echo unknown) |
| Backend | RViz 2 fake-hardware / joint_state simulation |
| Physical robot | **No** |
| Labels | ROS 2 SIMULATION · MOCK LANDMARKS · PANDA FAKE HARDWARE · NO PHYSICAL ROBOT · RETARGETER · SAFETY FILTER · live diagnostics topic |

## Files

- \`end-to-end-rviz.mp4\` — labeled screen capture of the real ROS 2 runtime
- \`end-to-end-rviz.gif\` — short GIF derivative
- \`end-to-end-rviz-thumb.png\` — thumbnail

## How produced

\`\`\`bash
sbatch scholar/record_v11_hardening.slurm
# launch: teleop_pipeline_rviz.launch.py
#   mock landmarks → TeleopPipeline (shoulder_relative + cbf_qp) → geometric_unverified IK → RSP → RViz
\`\`\`

Live diagnostics JSONL and metrics: \`results/v1.1-hardening/ros2/\`.
EOF

echo "=== JOB COMPLETE job=$JOB_TAG ==="
ls -la "$RESULTS_REPO" | head -50
ls -la "$MEDIA_REPO" | head -20
