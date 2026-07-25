#!/usr/bin/env python3
"""Simulation controller: Cartesian command poses → Panda joint trajectories.

Backends (selected by parameter):
  - joint_ik: analytic-ish geometric IK + JointTrajectoryController (default, robust)
  - dry_run: echo only

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

# Panda joint names (ros2_control / moveit convention)
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
        self.declare_parameter("publish_joint_states_fallback", True)
        self.declare_parameter("joint_states_topic", "/joint_states")

        self.mode = str(self.get_parameter("mode").value)
        allow_real = bool(self.get_parameter("allow_real_robot").value)
        if allow_real:
            self.get_logger().error("allow_real_robot=true is not supported; forcing simulation")
            allow_real = False
        self.traj_dt = float(self.get_parameter("trajectory_time_sec").value)
        self.cancelled = False
        self._last_joints = [0.0, -0.4, 0.0, -2.0, 0.0, 1.8, 0.8]
        self._ee_positions = []

        self.traj_pub = self.create_publisher(
            JointTrajectory, str(self.get_parameter("trajectory_topic").value), 10
        )
        self.status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.js_pub = None
        if bool(self.get_parameter("publish_joint_states_fallback").value):
            # Publishes joint states for RViz when full ros2_control stack is not used
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

    def _cancel_cb(self, msg: Bool) -> None:
        if msg.data:
            self.cancelled = True
            # Hold current joints (empty trajectory cancel semantics)
            s = String()
            s.data = "cancelled"
            self.status_pub.publish(s)
            self.get_logger().warn("Trajectory cancel received")
        else:
            self.cancelled = False

    def _publish_js(self) -> None:
        if self.js_pub is None:
            return
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name = list(PANDA_JOINTS)
        js.position = [float(v) for v in self._last_joints]
        self.js_pub.publish(js)

    def _pose_cb(self, msg: PoseStamped) -> None:
        if self.cancelled:
            s = String()
            s.data = "blocked_estop"
            self.status_pub.publish(s)
            return
        if self.mode == "dry_run":
            s = String()
            s.data = "dry_run"
            self.status_pub.publish(s)
            return

        x, y, z = msg.pose.position.x, msg.pose.position.y, msg.pose.position.z
        q = panda_geometric_ik(x, y, z)
        if q is None:
            s = String()
            s.data = "ik_failed"
            self.status_pub.publish(s)
            return

        # Smooth blend toward solution
        alpha = 0.35
        blended = [
            (1 - alpha) * a + alpha * b for a, b in zip(self._last_joints, q)
        ]
        self._last_joints = blended
        self._ee_positions.append([x, y, z])
        if len(self._ee_positions) > 500:
            self._ee_positions = self._ee_positions[-500:]

        traj = JointTrajectory()
        traj.joint_names = list(PANDA_JOINTS)
        pt = JointTrajectoryPoint()
        pt.positions = [float(v) for v in blended]
        sec = int(self.traj_dt)
        nsec = int((self.traj_dt - sec) * 1e9)
        pt.time_from_start = Duration(sec=sec, nanosec=nsec)
        traj.points = [pt]
        traj.header.stamp = self.get_clock().now().to_msg()
        self.traj_pub.publish(traj)

        s = String()
        s.data = f"commanded:[{x:.3f},{y:.3f},{z:.3f}]"
        self.status_pub.publish(s)


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
