#!/usr/bin/env python3
"""Publish visualization markers for mock target, command, EE, and trajectories.

Marker legend (distinct, accessible colors):
  Cyan sphere + label  — Mock human target  (/teleop/target_pose)
  Lime sphere + label  — Accepted command   (/teleop/command_pose)
  Orange sphere        — End-effector       (/teleop/ee_pose or command)
  Cyan line strip      — Target trajectory
  Lime line strip      — Command trajectory
  Semi-transparent box — Workspace bounds (optional legend anchors)
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, Pose, PoseStamped
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


# Accessible-ish RGB (0-1)
CYAN = (0.15, 0.75, 0.95)
LIME = (0.20, 0.90, 0.25)
ORANGE = (1.00, 0.55, 0.10)
LEGEND_BG = (0.12, 0.14, 0.18)


class TargetVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("target_visualizer")
        self.declare_parameter("target_topic", "/teleop/target_pose")
        self.declare_parameter("command_topic", "/teleop/command_pose")
        self.declare_parameter("ee_topic", "/teleop/ee_pose")
        self.declare_parameter("marker_topic", "/teleop/markers")
        self.declare_parameter("trail_length", 80)
        self.declare_parameter("base_frame", "panda_link0")
        self.declare_parameter("sphere_scale", 0.05)
        self.declare_parameter("show_workspace", True)
        self.declare_parameter("workspace.x_min", 0.25)
        self.declare_parameter("workspace.x_max", 0.70)
        self.declare_parameter("workspace.y_min", -0.35)
        self.declare_parameter("workspace.y_max", 0.35)
        self.declare_parameter("workspace.z_min", 0.10)
        self.declare_parameter("workspace.z_max", 0.70)

        self.trail_len = int(self.get_parameter("trail_length").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.sphere_scale = float(self.get_parameter("sphere_scale").value)
        self.show_workspace = bool(self.get_parameter("show_workspace").value)

        self.target_trail: Deque[Tuple[float, float, float]] = deque(maxlen=self.trail_len)
        self.cmd_trail: Deque[Tuple[float, float, float]] = deque(maxlen=self.trail_len)
        self._last_target: Optional[PoseStamped] = None
        self._last_cmd: Optional[PoseStamped] = None
        self._last_ee: Optional[PoseStamped] = None

        topic = str(self.get_parameter("marker_topic").value)
        self.pub = self.create_publisher(MarkerArray, topic, 10)
        # Also keep single-Marker topic for older RViz configs
        self.pub_one = self.create_publisher(Marker, topic + "_single", 10)

        self.create_subscription(
            PoseStamped, str(self.get_parameter("target_topic").value), self._target_cb, 10
        )
        self.create_subscription(
            PoseStamped, str(self.get_parameter("command_topic").value), self._cmd_cb, 10
        )
        self.create_subscription(
            PoseStamped, str(self.get_parameter("ee_topic").value), self._ee_cb, 10
        )
        self.create_timer(0.05, self._publish)
        self.get_logger().info("target_visualizer ready (markers + trajectories + legend)")

    def _pos(self, pose: PoseStamped) -> Tuple[float, float, float]:
        p = pose.pose.position
        return float(p.x), float(p.y), float(p.z)

    def _target_cb(self, msg: PoseStamped) -> None:
        self._last_target = msg
        self.target_trail.append(self._pos(msg))

    def _cmd_cb(self, msg: PoseStamped) -> None:
        self._last_cmd = msg
        self.cmd_trail.append(self._pos(msg))
        # Fallback EE if dedicated topic not published yet
        if self._last_ee is None:
            self._last_ee = msg

    def _ee_cb(self, msg: PoseStamped) -> None:
        self._last_ee = msg

    def _header(self, stamp=None):
        from std_msgs.msg import Header

        h = Header()
        h.frame_id = self.base_frame
        h.stamp = stamp if stamp is not None else self.get_clock().now().to_msg()
        return h

    def _sphere(
        self,
        mid: int,
        xyz: Tuple[float, float, float],
        rgb: Tuple[float, float, float],
        scale: Optional[float] = None,
        ns: str = "teleop",
        alpha: float = 0.95,
    ) -> Marker:
        m = Marker()
        m.header = self._header()
        m.ns = ns
        m.id = mid
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x, m.pose.position.y, m.pose.position.z = xyz
        m.pose.orientation.w = 1.0
        s = scale if scale is not None else self.sphere_scale
        m.scale.x = m.scale.y = m.scale.z = s
        m.color.a = alpha
        m.color.r, m.color.g, m.color.b = rgb
        m.lifetime.sec = 0
        return m

    def _text(
        self,
        mid: int,
        xyz: Tuple[float, float, float],
        text: str,
        rgb: Tuple[float, float, float],
        scale: float = 0.045,
    ) -> Marker:
        m = Marker()
        m.header = self._header()
        m.ns = "labels"
        m.id = mid
        m.type = Marker.TEXT_VIEW_FACING
        m.action = Marker.ADD
        m.pose.position.x = xyz[0]
        m.pose.position.y = xyz[1]
        m.pose.position.z = xyz[2] + 0.06
        m.pose.orientation.w = 1.0
        m.scale.z = scale
        m.color.a = 1.0
        m.color.r, m.color.g, m.color.b = rgb
        m.text = text
        return m

    def _line_strip(
        self,
        mid: int,
        points: Deque[Tuple[float, float, float]],
        rgb: Tuple[float, float, float],
        ns: str,
    ) -> Marker:
        m = Marker()
        m.header = self._header()
        m.ns = ns
        m.id = mid
        m.type = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.pose.orientation.w = 1.0
        m.scale.x = 0.012
        m.color.a = 0.85
        m.color.r, m.color.g, m.color.b = rgb
        m.points = []
        for x, y, z in points:
            pt = Point()
            pt.x, pt.y, pt.z = x, y, z
            m.points.append(pt)
        if not m.points:
            # Empty strip — publish delete-safe dummy
            m.action = Marker.DELETE
        return m

    def _workspace_box(self) -> Marker:
        m = Marker()
        m.header = self._header()
        m.ns = "workspace"
        m.id = 50
        m.type = Marker.CUBE
        m.action = Marker.ADD
        xmin = float(self.get_parameter("workspace.x_min").value)
        xmax = float(self.get_parameter("workspace.x_max").value)
        ymin = float(self.get_parameter("workspace.y_min").value)
        ymax = float(self.get_parameter("workspace.y_max").value)
        zmin = float(self.get_parameter("workspace.z_min").value)
        zmax = float(self.get_parameter("workspace.z_max").value)
        m.pose.position.x = 0.5 * (xmin + xmax)
        m.pose.position.y = 0.5 * (ymin + ymax)
        m.pose.position.z = 0.5 * (zmin + zmax)
        m.pose.orientation.w = 1.0
        m.scale.x = max(xmax - xmin, 0.01)
        m.scale.y = max(ymax - ymin, 0.01)
        m.scale.z = max(zmax - zmin, 0.01)
        m.color.a = 0.08
        m.color.r, m.color.g, m.color.b = 0.6, 0.7, 0.9
        return m

    def _legend(self) -> list:
        """Floating legend near the base (unobtrusive, in free space)."""
        # Place legend to the left of the robot base, above the floor
        origin = (-0.15, -0.55, 0.55)
        items = [
            (0, "Mock human target", CYAN),
            (1, "Accepted command", LIME),
            (2, "End-effector", ORANGE),
        ]
        markers = []
        for i, (idx, label, rgb) in enumerate(items):
            y = origin[1] + i * 0.08
            xyz = (origin[0], y, origin[2])
            markers.append(self._sphere(60 + idx, xyz, rgb, scale=0.035, ns="legend"))
            markers.append(
                self._text(70 + idx, (origin[0] + 0.08, y, origin[2] - 0.05), label, rgb, 0.038)
            )
        # Title
        markers.append(
            self._text(
                80,
                (origin[0] + 0.05, origin[1] - 0.06, origin[2] + 0.02),
                "Legend",
                (0.9, 0.9, 0.9),
                0.04,
            )
        )
        return markers

    def _publish(self) -> None:
        arr = MarkerArray()
        markers = []

        if self._last_target is not None:
            xyz = self._pos(self._last_target)
            markers.append(self._sphere(1, xyz, CYAN, scale=self.sphere_scale * 1.15))
            markers.append(self._line_strip(21, self.target_trail, CYAN, "target_trail"))

        if self._last_cmd is not None:
            xyz = self._pos(self._last_cmd)
            markers.append(self._sphere(2, xyz, LIME, scale=self.sphere_scale * 1.05))
            markers.append(self._line_strip(22, self.cmd_trail, LIME, "cmd_trail"))

        if self._last_ee is not None:
            xyz = self._pos(self._last_ee)
            # Slightly larger EE sphere so motion reads at a glance
            markers.append(self._sphere(3, xyz, ORANGE, scale=self.sphere_scale * 0.95))

        if self.show_workspace:
            markers.append(self._workspace_box())

        markers.extend(self._legend())

        arr.markers = markers
        self.pub.publish(arr)
        # Keep last sphere on single-marker topic for compatibility
        if markers:
            self.pub_one.publish(markers[0])


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
