#!/usr/bin/env python3
"""Safety monitor: gate target poses before controller execution."""
from __future__ import annotations

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, String

from vision_arm_control.safety import ControlMode, SafetyConfig, SafetyMonitor
from vision_arm_control.workspace_limits import AxisAlignedBounds, JointLimits, PANDA_JOINT_LIMITS


class SafetyMonitorNode:
    def __init__(self) -> None:
        rospy.init_node("safety_monitor_node")
        ws = rospy.get_param(
            "~workspace",
            {"x_min": 0.25, "x_max": 0.75, "y_min": -0.4, "y_max": 0.4, "z_min": 0.05, "z_max": 0.8},
        )
        bounds = AxisAlignedBounds(**{k: float(ws[k]) for k in ws})
        mode_str = rospy.get_param("~control_mode", "dry_run")
        try:
            control_mode = ControlMode(mode_str)
        except ValueError:
            control_mode = ControlMode.DRY_RUN
            rospy.logwarn("Unknown control_mode=%s; defaulting to dry_run", mode_str)

        cfg = SafetyConfig(
            workspace=bounds,
            joint_limits=PANDA_JOINT_LIMITS,
            max_linear_velocity=float(rospy.get_param("~max_linear_velocity", 0.2)),
            max_command_hz=float(rospy.get_param("~max_command_hz", 30.0)),
            pose_timeout_sec=float(rospy.get_param("~pose_timeout_sec", 0.5)),
            workspace_mode=rospy.get_param("~workspace_mode", "reject"),
            require_deadman=bool(rospy.get_param("~require_deadman", True)),
            allow_real_robot=bool(rospy.get_param("~allow_real_robot", False)),
        )
        self.monitor = SafetyMonitor(config=cfg, control_mode=control_mode)
        if control_mode == ControlMode.REAL_ROBOT:
            rospy.logwarn("=" * 60)
            rospy.logwarn("REAL ROBOT MODE REQUESTED")
            rospy.logwarn("Software safeguards do NOT replace Franka E-stop / Desk safety.")
            rospy.logwarn("Confirm clear workspace, reduced speed, and authorized operators.")
            rospy.logwarn("=" * 60)

        in_topic = rospy.get_param("~input_topic", "/human_to_robot_mapper_node/target_pose")
        out_topic = rospy.get_param("~output_topic", "/teleop/command_pose")
        self.sub = rospy.Subscriber(in_topic, PoseStamped, self._cb, queue_size=1)
        self.pub = rospy.Publisher(out_topic, PoseStamped, queue_size=1)
        self.status_pub = rospy.Publisher("~status", String, queue_size=1)
        self.estop_sub = rospy.Subscriber("~emergency_stop", Bool, self._estop_cb, queue_size=1)
        self.deadman_sub = rospy.Subscriber("~deadman", Bool, self._deadman_cb, queue_size=1)
        # Pose detections also refresh timeout when mapper is upstream; additionally
        # treat every accepted input as a pose observation.
        rospy.loginfo("safety_monitor_node mode=%s", self.monitor.control_mode.value)

    def _estop_cb(self, msg: Bool) -> None:
        if msg.data:
            self.monitor.request_emergency_stop()
            rospy.logerr("EMERGENCY STOP asserted")
        else:
            self.monitor.clear_emergency_stop()
            rospy.logwarn("Emergency stop cleared")

    def _deadman_cb(self, msg: Bool) -> None:
        self.monitor.set_deadman(msg.data)

    def _cb(self, msg: PoseStamped) -> None:
        now = rospy.Time.now().to_sec()
        self.monitor.note_pose(now)
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        decision = self.monitor.evaluate_position(pos, now)
        self.status_pub.publish(String(data=decision.reason))
        if not decision.accepted or decision.position is None:
            return
        out = PoseStamped()
        out.header = msg.header
        out.header.stamp = rospy.Time.now()
        out.pose = msg.pose
        out.pose.position.x = float(decision.position[0])
        out.pose.position.y = float(decision.position[1])
        out.pose.position.z = float(decision.position[2])
        self.pub.publish(out)


if __name__ == "__main__":
    try:
        SafetyMonitorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
