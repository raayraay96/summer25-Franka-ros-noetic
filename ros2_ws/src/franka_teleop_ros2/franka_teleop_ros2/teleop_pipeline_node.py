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
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
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


class TeleopPipelineNode(Node):
    def __init__(self) -> None:
        super().__init__("teleop_pipeline_node")
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

        self.strategy = str(gp("retargeting_strategy"))
        self.safety_mode = str(gp("safety_filter"))
        self.ik_backend = str(gp("ik_backend"))
        self.frame_id = str(gp("frame_id"))
        self.diag_path = str(gp("diagnostics_jsonl"))
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
        self.pub_safety = self.create_publisher(String, "/teleop/safety_status", 10)
        self.pub_gate = self.create_publisher(String, "/teleop/gate_status", 10)
        self.pub_diag = self.create_publisher(String, "/teleop/diagnostics", 10)
        self.create_subscription(String, str(gp("landmark_topic")), self._on_landmarks, 20)
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
        self.pub_diag.publish(self._s(rec))
        if self._diag_file is not None:
            self._diag_file.write(json.dumps(rec, sort_keys=True) + "\n")
            self._diag_file.flush()

    def _publish_joints(self, p) -> None:
        if self.ik_backend == "none":
            return
        joints = None
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
            return
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(PANDA_JOINTS)
        js.position = [float(v) for v in joints]
        # Tag provenance in a way RViz ignores but recordings capture.
        js.header.frame_id = "verified_moveit" if verified else "geometric_unverified"
        self.pub_joints.publish(js)

    def _moveit_ik(self, p):
        """Verified IK via MoveIt compute_ik (Scholar only).

        Left as an explicit integration point: on a host with MoveIt 2 + the
        Panda MoveItConfig, call the ``/compute_ik`` service (moveit_msgs/GetPositionIK)
        with the target pose and current seed, enforce joint limits, and return
        (joints, True). Until then this returns (None, False) so no unverified
        joints are ever published as 'verified'.
        """
        self.get_logger().warn_once("moveit IK backend not wired on this host; publishing no joints")
        return None, False

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
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
