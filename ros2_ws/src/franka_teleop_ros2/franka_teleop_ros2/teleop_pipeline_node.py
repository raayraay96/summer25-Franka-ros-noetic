#!/usr/bin/env python3
"""ROS 2 node wiring the *hardened* selectable teleop pipeline (Phase 5/6).

Runs the pure-Python ``vision_arm_control.TeleopPipeline`` (selectable
retargeter + safety filter + confidence gate, all cross-validated in CI) against
live ROS 2 topics on Panda fake hardware. Defaults keep ``use_robot=false`` /
``dry_run``; no physical output.

Parameters (all overridable from the launch file / YAML):
  retargeting_strategy : shoulder_relative | sew_orientation
  safety_filter        : reject | clamp | cbf_qp
  ik_backend           : none | geometric_unverified | moveit
  dt, alpha, workspace_margin_m, per_axis_velocity_limit_mps
  workspace_{x,y,z}_{min,max}
  obstacles            : JSON string  "[{"id","center":[x,y,z],"radius_m","margin_m"}]"
  landmark_topic       : default /perception/landmarks (std_msgs/String JSON)
  diagnostics_jsonl    : path to append per-step diagnostics (for offline metrics)

Publishes:
  /teleop/target_pose   geometry_msgs/PoseStamped   (retargeted, pre-safety)
  /teleop/command_pose  geometry_msgs/PoseStamped   (safety-accepted command)
  /joint_states         sensor_msgs/JointState      (only if ik_backend != none)
  /teleop/safety_status std_msgs/String  (JSON: solver_status, active, interv.)
  /teleop/gate_status   std_msgs/String  (gate FSM state)
  /teleop/diagnostics   std_msgs/String  (JSON: full structured record)

No silent fallback: every mode change is emitted in the diagnostics record.
The ``moveit`` IK backend is the only one that may claim *verified* joints; the
``geometric_unverified`` backend is for RViz visualization only and is labeled
as such in every diagnostics record and JointState is tagged accordingly.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import String

# Import the hardened pure-Python core from the main package (numpy-only).
_REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_REPO / "src" / "vision_arm_control" / "src"))

from franka_teleop_ros2.ik_utils import panda_geometric_ik  # noqa: E402
from vision_arm_control.landmarks.confidence import ConfidenceGateConfig  # noqa: E402
from vision_arm_control.landmarks.schema import ArmLandmarks, LandmarkPoint  # noqa: E402
from vision_arm_control.pipeline import PipelineConfig, TeleopPipeline  # noqa: E402
from vision_arm_control.pipeline_diagnostics import DiagnosticsBuilder  # noqa: E402
from vision_arm_control.retargeting.base import RetargetingConfig  # noqa: E402
from vision_arm_control.safety_filters.base import SafetyFilterConfig, SphericalObstacle  # noqa: E402
from vision_arm_control.workspace_limits import PANDA_JOINT_LIMITS, AxisAlignedBounds  # noqa: E402

PANDA_JOINTS = [f"panda_joint{i}" for i in range(1, 8)]
# Panda "ready" seed from moveit_resources_panda_moveit_config (radians).
PANDA_READY_SEED = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


class TeleopPipelineNode(Node):
    def __init__(self) -> None:
        super().__init__("teleop_pipeline_node")
        self._cb_group = ReentrantCallbackGroup()
        gp = self._p
        self.declare_parameter("retargeting_strategy", "shoulder_relative")
        self.declare_parameter("safety_filter", "cbf_qp")
        self.declare_parameter("ik_backend", "none")
        self.declare_parameter("dt", 0.05)
        self.declare_parameter("alpha", 4.0)
        self.declare_parameter("workspace_margin_m", 0.03)
        self.declare_parameter("per_axis_velocity_limit_mps", 0.20)
        self.declare_parameter("workspace_x_min", 0.25)
        self.declare_parameter("workspace_x_max", 0.75)
        self.declare_parameter("workspace_y_min", -0.40)
        self.declare_parameter("workspace_y_max", 0.40)
        self.declare_parameter("workspace_z_min", 0.05)
        self.declare_parameter("workspace_z_max", 0.80)
        self.declare_parameter("obstacles", "[]")
        self.declare_parameter("landmark_topic", "/perception/landmarks")
        self.declare_parameter("frame_id", "panda_link0")
        self.declare_parameter("diagnostics_jsonl", "")
        self.declare_parameter("initial_position", [0.45, 0.0, 0.45])
        self.declare_parameter("compute_ik_service", "/compute_ik")
        self.declare_parameter("ik_group_name", "panda_arm")
        self.declare_parameter("ik_link_name", "panda_link8")
        self.declare_parameter("ik_timeout_sec", 0.08)
        self.declare_parameter("publish_joint_states", True)

        self.strategy = str(gp("retargeting_strategy"))
        self.safety_mode = str(gp("safety_filter"))
        self.ik_backend = str(gp("ik_backend"))
        self.frame_id = str(gp("frame_id"))
        self.diag_path = str(gp("diagnostics_jsonl"))
        self.ik_group = str(gp("ik_group_name"))
        self.ik_link = str(gp("ik_link_name"))
        self.ik_timeout = float(gp("ik_timeout_sec"))
        self.publish_joint_states = bool(gp("publish_joint_states"))
        self._last_joints: List[float] = list(PANDA_READY_SEED)
        self._ik_client = None
        self._ik_ready_logged = False
        self._ik_fail_count = 0
        self._ik_ok_count = 0
        self._diag_file = None
        if self.diag_path:
            Path(self.diag_path).parent.mkdir(parents=True, exist_ok=True)
            self._diag_file = open(self.diag_path, "w")  # noqa: SIM115

        bounds = AxisAlignedBounds(
            float(gp("workspace_x_min")),
            float(gp("workspace_x_max")),
            float(gp("workspace_y_min")),
            float(gp("workspace_y_max")),
            float(gp("workspace_z_min")),
            float(gp("workspace_z_max")),
        )
        obstacles = [
            SphericalObstacle(
                id=o["id"],
                center=o["center"],
                radius_m=float(o["radius_m"]),
                margin_m=float(o.get("margin_m", 0.0)),
            )
            for o in json.loads(str(gp("obstacles")) or "[]")
        ]
        safety_cfg = SafetyFilterConfig(
            mode=self.safety_mode,
            workspace=bounds,
            workspace_margin_m=float(gp("workspace_margin_m")),
            dt=float(gp("dt")),
            alpha=float(gp("alpha")),
            per_axis_velocity_limit_mps=float(gp("per_axis_velocity_limit_mps")),
            obstacles=obstacles if self.safety_mode == "cbf_qp" else [],
        )
        init = [float(v) for v in gp("initial_position")]
        self.pipe = TeleopPipeline(
            PipelineConfig(
                retargeting=RetargetingConfig(name=self.strategy, treat_z_as_image_relative=True),
                safety=safety_cfg,
                gate=ConfidenceGateConfig(require_elbow=(self.strategy == "sew_orientation")),
                backend="dry_run",
                initial_position=tuple(init),
            )
        )
        self.diag = DiagnosticsBuilder(strategy=self.strategy, safety_mode=self.safety_mode)

        self.pub_target = self.create_publisher(PoseStamped, "/teleop/target_pose", 10)
        self.pub_cmd = self.create_publisher(PoseStamped, "/teleop/command_pose", 10)
        self.pub_joints = self.create_publisher(JointState, "/joint_states", 10)
        # Evidence-only topic: always receives limit-checked IK when available.
        self.pub_verified = self.create_publisher(JointState, "/teleop/ik_joint_states", 10)
        self.pub_safety = self.create_publisher(String, "/teleop/safety_status", 10)
        self.pub_gate = self.create_publisher(String, "/teleop/gate_status", 10)
        self.pub_diag = self.create_publisher(String, "/teleop/diagnostics", 10)
        self.create_subscription(
            String,
            str(gp("landmark_topic")),
            self._on_landmarks,
            20,
            callback_group=self._cb_group,
        )

        if self.ik_backend == "moveit":
            try:
                from moveit_msgs.srv import GetPositionIK  # noqa: WPS433

                self._ik_client = self.create_client(
                    GetPositionIK,
                    str(gp("compute_ik_service")),
                    callback_group=self._cb_group,
                )
            except ImportError:
                self.get_logger().error(
                    "moveit_msgs not importable; ik_backend=moveit will publish no joints"
                )
                self._ik_client = None

        self.get_logger().info(
            f"teleop_pipeline_node: strategy={self.strategy} safety={self.safety_mode} "
            f"ik={self.ik_backend} (dry_run; no physical output)"
        )

    def _p(self, name: str):
        return self.get_parameter(name).value

    @staticmethod
    def _parse_landmarks(data: str):
        payload = json.loads(data)
        if not payload.get("pose_detected", True):
            return None, float(payload.get("header", {}).get("stamp", 0.0))
        stamp = float(payload.get("header", {}).get("stamp", 0.0))
        by_name = {lm["name"]: lm for lm in payload.get("landmarks", [])}

        def pt(name):
            lm = by_name.get(name)
            if lm is None:
                return None
            return LandmarkPoint(
                float(lm["x"]),
                float(lm["y"]),
                float(lm.get("z", 0.0)),
                confidence=float(lm.get("confidence", 1.0)),
                name=name,
            )

        arm = ArmLandmarks(
            shoulder=pt("right_shoulder"), elbow=pt("right_elbow"), wrist=pt("right_wrist"), timestamp=stamp
        )
        return arm, stamp

    def _on_landmarks(self, msg: String) -> None:
        arm, stamp = self._parse_landmarks(msg.data)
        now = self.get_clock().now().nanoseconds * 1e-9
        step = self.pipe.step(arm, now=stamp or now)
        gate_state = self.pipe.gate.status.value

        if step.retargeted.position is not None:
            self.pub_target.publish(self._pose(step.retargeted.position))
        if step.commanded and step.safety and step.safety.position is not None:
            p = step.safety.position
            self.pub_cmd.publish(self._pose(p))
            self._publish_joints(p)

        if step.safety is not None:
            self.pub_safety.publish(
                self._s(
                    {
                        "solver_status": step.safety.solver_status,
                        "active_constraints": step.safety.active_constraints,
                        "intervention_magnitude": step.safety.intervention_magnitude,
                        "accepted": step.safety.accepted,
                        "reason": step.safety.reason,
                    }
                )
            )
        self.pub_gate.publish(self._s(gate_state))

        rec = self.diag.update(step, gate_state, now=stamp or now)
        rec["ik_backend"] = self.ik_backend
        rec["ik_ok_count"] = self._ik_ok_count
        rec["ik_fail_count"] = self._ik_fail_count
        self.pub_diag.publish(self._s(rec))
        if self._diag_file is not None:
            self._diag_file.write(json.dumps(rec, sort_keys=True) + "\n")
            self._diag_file.flush()

    def _publish_joints(self, p) -> None:
        if self.ik_backend == "none":
            return
        joints: Optional[Sequence[float]] = None
        verified = False
        if self.ik_backend == "geometric_unverified":
            joints = panda_geometric_ik(float(p[0]), float(p[1]), float(p[2]))
        elif self.ik_backend == "moveit":
            joints, verified = self._moveit_ik(p)
        if joints is None:
            return
        # Independent joint-limit check before publishing (never emit OOL joints).
        if not PANDA_JOINT_LIMITS.contains(joints):
            self.get_logger().warn("IK joints out of Panda limits; not publishing")
            self._ik_fail_count += 1
            return
        self._last_joints = [float(v) for v in joints]
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(PANDA_JOINTS)
        js.position = list(self._last_joints)
        # Tag provenance in a way RViz ignores but recordings capture.
        js.header.frame_id = "verified_moveit" if verified else "geometric_unverified"
        # Always publish evidence topic; optional /joint_states for RViz when we own it.
        self.pub_verified.publish(js)
        if self.publish_joint_states:
            self.pub_joints.publish(js)
        if verified:
            self._ik_ok_count += 1

    def _moveit_ik(self, p) -> Tuple[Optional[List[float]], bool]:
        """Verified IK via MoveIt ``/compute_ik`` (moveit_msgs/GetPositionIK).

        Requires a live MoveIt 2 ``move_group`` (e.g. panda MoveItConfig demo) that
        advertises ``/compute_ik``. Returns ``(None, False)`` when the service is
        missing, times out, or returns a non-success error code — never fabricates
        joints or claims verification without a MoveIt solution.
        """
        if self._ik_client is None:
            self.get_logger().warn_once(
                "moveit IK backend selected but GetPositionIK client unavailable; " "publishing no joints"
            )
            self._ik_fail_count += 1
            return None, False
        if not self._ik_client.service_is_ready():
            if not self._ik_ready_logged:
                self.get_logger().warn("waiting for /compute_ik (start panda move_group / MoveIt demo)")
                self._ik_ready_logged = True
            self._ik_fail_count += 1
            return None, False
        if self._ik_ready_logged:
            self.get_logger().info("/compute_ik ready; producing verified joints")
            self._ik_ready_logged = False

        from moveit_msgs.msg import PositionIKRequest  # noqa: WPS433
        from moveit_msgs.srv import GetPositionIK  # noqa: WPS433

        req = GetPositionIK.Request()
        req.ik_request = PositionIKRequest()
        req.ik_request.group_name = self.ik_group
        req.ik_request.ik_link_name = self.ik_link
        req.ik_request.pose_stamped = self._pose(p)
        # Prefer a pointing-down-ish EE for reachable demo poses.
        req.ik_request.pose_stamped.pose.orientation.x = 1.0
        req.ik_request.pose_stamped.pose.orientation.y = 0.0
        req.ik_request.pose_stamped.pose.orientation.z = 0.0
        req.ik_request.pose_stamped.pose.orientation.w = 0.0
        req.ik_request.avoid_collisions = False
        req.ik_request.timeout.sec = 0
        req.ik_request.timeout.nanosec = int(max(self.ik_timeout, 0.01) * 1e9)
        req.ik_request.robot_state.joint_state.name = list(PANDA_JOINTS)
        req.ik_request.robot_state.joint_state.position = list(self._last_joints)

        future = self._ik_client.call_async(req)
        # MultiThreadedExecutor + ReentrantCallbackGroup: wait without deadlocking.
        deadline = time.monotonic() + max(self.ik_timeout, 0.05) + 0.05
        while not future.done() and time.monotonic() < deadline:
            time.sleep(0.001)
        if not future.done():
            self._ik_fail_count += 1
            return None, False
        try:
            resp = future.result()
        except Exception as exc:  # noqa: BLE001 — surface once, keep pipeline alive
            self.get_logger().warn(f"compute_ik call failed: {exc}")
            self._ik_fail_count += 1
            return None, False

        # moveit_msgs/MoveItErrorCodes: SUCCESS == 1
        if resp.error_code.val != 1:
            self._ik_fail_count += 1
            return None, False
        name_to_pos = {
            n: float(v) for n, v in zip(resp.solution.joint_state.name, resp.solution.joint_state.position)
        }
        if not all(j in name_to_pos for j in PANDA_JOINTS):
            self._ik_fail_count += 1
            return None, False
        return [name_to_pos[j] for j in PANDA_JOINTS], True

    def _pose(self, xyz) -> PoseStamped:
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = self.frame_id
        m.pose.position.x = float(xyz[0])
        m.pose.position.y = float(xyz[1])
        m.pose.position.z = float(xyz[2])
        m.pose.orientation.w = 1.0
        return m

    @staticmethod
    def _s(payload) -> String:
        msg = String()
        msg.data = payload if isinstance(payload, str) else json.dumps(payload)
        return msg


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TeleopPipelineNode()
    # Multi-threaded so /compute_ik futures can complete inside the landmark callback.
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        if node._diag_file is not None:
            node._diag_file.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
