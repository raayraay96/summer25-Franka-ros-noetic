#!/usr/bin/env python3
"""Safety gate for teleop Cartesian targets (ROS 2)."""
from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Bool, String

from franka_teleop_ros2.safety import ControlMode, SafetyConfig, SafetyMonitor
from franka_teleop_ros2.trajectory_filter import CommandRateLimiter
from franka_teleop_ros2.workspace_limits import AxisAlignedBounds, PANDA_JOINT_LIMITS


class SafetyMonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("safety_monitor")
        self.declare_parameter("input_topic", "/teleop/target_pose")
        self.declare_parameter("output_topic", "/teleop/command_pose")
        self.declare_parameter("status_topic", "/teleop/safety_status")
        self.declare_parameter("control_mode", "simulation")
        self.declare_parameter("allow_real_robot", False)
        self.declare_parameter("require_deadman", False)  # false for pure sim demos
        self.declare_parameter("max_command_hz", 30.0)
        self.declare_parameter("pose_timeout_sec", 0.75)
        self.declare_parameter("workspace_mode", "clamp")
        self.declare_parameter("workspace.x_min", 0.25)
        self.declare_parameter("workspace.x_max", 0.70)
        self.declare_parameter("workspace.y_min", -0.35)
        self.declare_parameter("workspace.y_max", 0.35)
        self.declare_parameter("workspace.z_min", 0.10)
        self.declare_parameter("workspace.z_max", 0.70)
        self.declare_parameter("max_linear_velocity", 0.25)

        bounds = AxisAlignedBounds(
            x_min=float(self.get_parameter("workspace.x_min").value),
            x_max=float(self.get_parameter("workspace.x_max").value),
            y_min=float(self.get_parameter("workspace.y_min").value),
            y_max=float(self.get_parameter("workspace.y_max").value),
            z_min=float(self.get_parameter("workspace.z_min").value),
            z_max=float(self.get_parameter("workspace.z_max").value),
        )
        mode_str = str(self.get_parameter("control_mode").value)
        try:
            control_mode = ControlMode(mode_str)
        except ValueError:
            control_mode = ControlMode.SIMULATION
            self.get_logger().warn(f"Unknown control_mode={mode_str}; using simulation")

        allow_real = bool(self.get_parameter("allow_real_robot").value)
        cfg = SafetyConfig(
            workspace=bounds,
            joint_limits=PANDA_JOINT_LIMITS,
            max_linear_velocity=float(self.get_parameter("max_linear_velocity").value),
            max_command_hz=float(self.get_parameter("max_command_hz").value),
            pose_timeout_sec=float(self.get_parameter("pose_timeout_sec").value),
            workspace_mode=str(self.get_parameter("workspace_mode").value),
            require_deadman=bool(self.get_parameter("require_deadman").value),
            allow_real_robot=allow_real,
        )
        self.monitor = SafetyMonitor(config=cfg, control_mode=control_mode)
        # For simulation, dead-man defaults to enabled if not required
        if not cfg.require_deadman:
            self.monitor.set_deadman(True)
        self.rate = CommandRateLimiter(max_hz=float(self.get_parameter("max_command_hz").value))

        if control_mode == ControlMode.REAL_ROBOT:
            self.get_logger().error(
                "REAL ROBOT MODE — software safeguards do not replace Franka E-stop."
            )

        self.pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("output_topic").value), 10
        )
        self.status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )
        self.cancel_pub = self.create_publisher(Bool, "/teleop/cancel_trajectory", 10)
        self.sub = self.create_subscription(
            PoseStamped, str(self.get_parameter("input_topic").value), self._cb, 10
        )
        self.create_subscription(Bool, "/teleop/emergency_stop", self._estop_cb, 10)
        self.create_subscription(Bool, "/teleop/deadman", self._deadman_cb, 10)
        self.get_logger().info(
            f"safety_monitor mode={self.monitor.control_mode.value} "
            f"allow_real={allow_real}"
        )

    def _estop_cb(self, msg: Bool) -> None:
        if msg.data:
            self.monitor.request_emergency_stop()
            c = Bool()
            c.data = True
            self.cancel_pub.publish(c)
            self.get_logger().error("EMERGENCY STOP — cancel trajectory asserted")
        else:
            self.monitor.clear_emergency_stop()
            self.get_logger().warn("Emergency stop cleared")

    def _deadman_cb(self, msg: Bool) -> None:
        self.monitor.set_deadman(msg.data)

    def _cb(self, msg: PoseStamped) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        self.monitor.note_pose(now)
        if not self.rate.allow(now):
            s = String()
            s.data = "rate_limited"
            self.status_pub.publish(s)
            return
        pos = [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z]
        decision = self.monitor.evaluate_position(pos, now)
        s = String()
        s.data = decision.reason
        self.status_pub.publish(s)
        if not decision.accepted or decision.position is None:
            if decision.emergency_stop:
                c = Bool()
                c.data = True
                self.cancel_pub.publish(c)
            return
        out = PoseStamped()
        out.header = msg.header
        out.header.stamp = self.get_clock().now().to_msg()
        out.pose = msg.pose
        out.pose.position.x = float(decision.position[0])
        out.pose.position.y = float(decision.position[1])
        out.pose.position.z = float(decision.position[2])
        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SafetyMonitorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
