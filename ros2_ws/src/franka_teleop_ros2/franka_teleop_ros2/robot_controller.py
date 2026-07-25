#!/usr/bin/env python3
"""Simulation controller: Cartesian command poses to Panda joint trajectories.

Backends selected by parameter:
  - joint_ik: geometric IK + JointTrajectory publication (default)
  - dry_run: validate and report targets without publishing motion commands

Real hardware is never enabled by this node.
"""
from __future__ import annotations

from typing import List

import rclpy
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, String
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from franka_teleop_ros2.ik_utils import panda_geometric_ik
from franka_teleop_ros2.workspace_limits import PANDA_JOINT_LIMITS

# Panda joint names (ros2_control / MoveIt convention)
PANDA_JOINTS = [
    "panda_joint1",
    "panda_joint2",
    "panda_joint3",
    "panda_joint4",
    "panda_joint5",
    "panda_joint6",
    "panda_joint7",
]


class RobotController(Node):
    def __init__(self) -> None:
        super().__init__("robot_controller")
        self.declare_parameter("mode", "joint_ik")  # joint_ik | dry_run
        self.declare_parameter("input_topic", "/teleop/command_pose")
        self.declare_parameter("trajectory_topic", "/panda_arm_controller/joint_trajectory")
        self.declare_parameter("status_topic", "/teleop/controller_status")
        self.declare_parameter("allow_real_robot", False)
        self.declare_parameter("trajectory_time_sec", 0.35)
        self.declare_parameter("hold_time_sec", 0.10)
        self.declare_parameter("blend_alpha", 0.35)
        self.declare_parameter("publish_joint_states_fallback", True)
        self.declare_parameter("joint_states_topic", "/joint_states")

        self.mode = str(self.get_parameter("mode").value)
        allow_real = bool(self.get_parameter("allow_real_robot").value)
        if allow_real:
            self.get_logger().error("allow_real_robot=true is unsupported; forcing simulation")
        self.traj_dt = max(float(self.get_parameter("trajectory_time_sec").value), 0.05)
        self.hold_dt = max(float(self.get_parameter("hold_time_sec").value), 0.02)
        self.blend_alpha = min(max(float(self.get_parameter("blend_alpha").value), 0.0), 1.0)
        self.cancelled = False
        self._last_joints = [0.0, -0.4, 0.0, -2.0, 0.0, 1.8, 0.8]
        self._ee_positions: List[List[float]] = []

        self.traj_pub = self.create_publisher(
            JointTrajectory, str(self.get_parameter("trajectory_topic").value), 10
        )
        self.status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.js_pub = None
        if bool(self.get_parameter("publish_joint_states_fallback").value):
            # Publishes joint states for RViz when full ros2_control is not used.
            self.js_pub = self.create_publisher(
                JointState, str(self.get_parameter("joint_states_topic").value), 10
            )
            self.create_timer(0.05, self._publish_js)

        self.create_subscription(
            PoseStamped, str(self.get_parameter("input_topic").value), self._pose_cb, 10
        )
        self.create_subscription(Bool, "/teleop/cancel_trajectory", self._cancel_cb, 10)
        self.get_logger().info(
            f"robot_controller mode={self.mode} backend=simulation (no physical hardware)"
        )

    @staticmethod
    def _duration(seconds: float) -> Duration:
        sec = int(seconds)
        nanosec = int((seconds - sec) * 1e9)
        return Duration(sec=sec, nanosec=nanosec)

    def _publish_status(self, text: str) -> None:
        msg = String()
        msg.data = text
        self.status_pub.publish(msg)

    def _publish_hold(self) -> None:
        """Publish a short hold trajectory at the last accepted joint state."""
        trajectory = JointTrajectory()
        trajectory.header.stamp = self.get_clock().now().to_msg()
        trajectory.joint_names = list(PANDA_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in self._last_joints]
        point.time_from_start = self._duration(self.hold_dt)
        trajectory.points = [point]
        self.traj_pub.publish(trajectory)

    def _cancel_cb(self, msg: Bool) -> None:
        requested = bool(msg.data)
        if requested:
            if not self.cancelled:
                self.cancelled = True
                self._publish_hold()
                self._publish_status("cancelled_hold")
                self.get_logger().warn("Trajectory cancel received; holding current joints")
            return

        if self.cancelled:
            self.cancelled = False
            self._publish_status("resumed")
            self.get_logger().info("Trajectory cancellation released")

    def _publish_js(self) -> None:
        if self.js_pub is None:
            return
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(PANDA_JOINTS)
        js.position = [float(value) for value in self._last_joints]
        self.js_pub.publish(js)

    def _pose_cb(self, msg: PoseStamped) -> None:
        if self.cancelled:
            self._publish_status("blocked_cancelled")
            return
        if self.mode == "dry_run":
            self._publish_status("dry_run")
            return
        if self.mode != "joint_ik":
            self._publish_status(f"unsupported_mode:{self.mode}")
            return

        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        solution = panda_geometric_ik(x, y, z)
        if solution is None:
            self._publish_status("ik_failed")
            return
        if not PANDA_JOINT_LIMITS.contains(solution):
            self._publish_status("ik_outside_joint_limits")
            return

        blended = [
            (1.0 - self.blend_alpha) * previous + self.blend_alpha * target
            for previous, target in zip(self._last_joints, solution)
        ]
        if not PANDA_JOINT_LIMITS.contains(blended):
            self._publish_status("blended_command_outside_joint_limits")
            return

        self._last_joints = blended
        self._ee_positions.append([x, y, z])
        if len(self._ee_positions) > 500:
            self._ee_positions = self._ee_positions[-500:]

        trajectory = JointTrajectory()
        trajectory.joint_names = list(PANDA_JOINTS)
        point = JointTrajectoryPoint()
        point.positions = [float(value) for value in blended]
        point.time_from_start = self._duration(self.traj_dt)
        trajectory.points = [point]
        trajectory.header.stamp = self.get_clock().now().to_msg()
        self.traj_pub.publish(trajectory)
        self._publish_status(f"commanded:[{x:.3f},{y:.3f},{z:.3f}]")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = RobotController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
