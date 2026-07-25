#!/usr/bin/env python3
"""Publish visualization markers for target / command poses."""
from __future__ import annotations

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from visualization_msgs.msg import Marker


class TargetVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("target_visualizer")
        self.declare_parameter("target_topic", "/teleop/target_pose")
        self.declare_parameter("command_topic", "/teleop/command_pose")
        self.declare_parameter("marker_topic", "/teleop/markers")
        self.pub = self.create_publisher(
            Marker, str(self.get_parameter("marker_topic").value), 10
        )
        self.create_subscription(
            PoseStamped, str(self.get_parameter("target_topic").value), self._target_cb, 10
        )
        self.create_subscription(
            PoseStamped, str(self.get_parameter("command_topic").value), self._cmd_cb, 10
        )
        self.get_logger().info("target_visualizer ready")

    def _sphere(self, pose: PoseStamped, mid: int, r: float, g: float, b: float) -> Marker:
        m = Marker()
        m.header = pose.header
        m.ns = "teleop"
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose = pose.pose
        m.scale.x = m.scale.y = m.scale.z = 0.04
        m.color.a = 0.9
        m.color.r, m.color.g, m.color.b = r, g, b
        return m

    def _target_cb(self, msg: PoseStamped) -> None:
        self.pub.publish(self._sphere(msg, 1, 0.1, 0.8, 1.0))

    def _cmd_cb(self, msg: PoseStamped) -> None:
        self.pub.publish(self._sphere(msg, 2, 0.1, 1.0, 0.2))


def main(args=None) -> None:
    rclpy.init(args=args)
    node = TargetVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
