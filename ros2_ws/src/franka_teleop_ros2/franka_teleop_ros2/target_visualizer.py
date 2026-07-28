#!/usr/bin/env python3
"""Publish visualization markers for target, command, EE, and trajectories.

In-scene labels are intentionally omitted — the recruiter-facing legend is a
compact overlay burned into the demo video (post-process) so it stays readable
and does not scatter text across the robot.

Marker colors (match post-process legend):
  Cyan sphere / trail  — Target            (/teleop/target_pose)
  Lime sphere / trail  — Accepted command  (/teleop/command_pose)
  Orange sphere        — End effector      (/teleop/ee_pose)
"""
from __future__ import annotations

from collections import deque
from typing import Deque, Optional, Tuple

import rclpy
from geometry_msgs.msg import Point, PoseStamped
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray


CYAN = (0.15, 0.75, 0.95)
LIME = (0.20, 0.90, 0.25)
ORANGE = (1.00, 0.55, 0.10)


class TargetVisualizer(Node):
    def __init__(self) -> None:
        super().__init__("target_visualizer")
        self.declare_parameter("target_topic", "/teleop/target_pose")
        self.declare_parameter("command_topic", "/teleop/command_pose")
        self.declare_parameter("ee_topic", "/teleop/ee_pose")
        self.declare_parameter("marker_topic", "/teleop/markers")
        self.declare_parameter("trail_length", 90)
        self.declare_parameter("base_frame", "panda_link0")
        self.declare_parameter("sphere_scale", 0.055)
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
        self.pub_one = self.create_publisher(Marker, topic + "_single", 10)

        self.create_subscription(
            PoseStamped, str(self.get_parameter("target_topic").value), self._target_cb, 10
        )
        self.create_subscription(
            PoseStamped, str(self.get_parameter("command_topic").value), self._cmd_cb, 10
        )
        self.create_subscription(PoseStamped, str(self.get_parameter("ee_topic").value), self._ee_cb, 10)
        self.create_timer(0.05, self._publish)
        self.get_logger().info("target_visualizer ready (spheres + trails; legend is video overlay)")

    def _pos(self, pose: PoseStamped) -> Tuple[float, float, float]:
        p = pose.pose.position
        return float(p.x), float(p.y), float(p.z)

    def _target_cb(self, msg: PoseStamped) -> None:
        self._last_target = msg
        self.target_trail.append(self._pos(msg))

    def _cmd_cb(self, msg: PoseStamped) -> None:
        self._last_cmd = msg
        self.cmd_trail.append(self._pos(msg))
        if self._last_ee is None:
            self._last_ee = msg

    def _ee_cb(self, msg: PoseStamped) -> None:
        self._last_ee = msg

    def _header(self):
        from std_msgs.msg import Header

        h = Header()
        h.frame_id = self.base_frame
        h.stamp = self.get_clock().now().to_msg()
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
        m.scale.x = 0.014
        m.color.a = 0.88
        m.color.r, m.color.g, m.color.b = rgb
        m.points = []
        for x, y, z in points:
            pt = Point()
            pt.x, pt.y, pt.z = x, y, z
            m.points.append(pt)
        if not m.points:
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
        m.color.a = 0.06
        m.color.r, m.color.g, m.color.b = 0.6, 0.7, 0.9
        return m

    def _delete_legacy_labels(self) -> list:
        """Delete any leftover TEXT markers from older visualizer versions."""
        out = []
        for mid in list(range(11, 20)) + list(range(60, 90)):
            m = Marker()
            m.header = self._header()
            m.ns = "labels" if mid < 60 else "legend"
            m.id = mid
            m.action = Marker.DELETE
            out.append(m)
        return out

    def _publish(self) -> None:
        arr = MarkerArray()
        markers = self._delete_legacy_labels()

        if self._last_target is not None:
            xyz = self._pos(self._last_target)
            markers.append(self._sphere(1, xyz, CYAN, scale=self.sphere_scale * 1.1))
            markers.append(self._line_strip(21, self.target_trail, CYAN, "target_trail"))

        if self._last_cmd is not None:
            xyz = self._pos(self._last_cmd)
            markers.append(self._sphere(2, xyz, LIME, scale=self.sphere_scale))
            markers.append(self._line_strip(22, self.cmd_trail, LIME, "cmd_trail"))

        if self._last_ee is not None:
            xyz = self._pos(self._last_ee)
            markers.append(self._sphere(3, xyz, ORANGE, scale=self.sphere_scale * 0.95))

        if self.show_workspace:
            markers.append(self._workspace_box())

        arr.markers = markers
        self.pub.publish(arr)
        if markers:
            # Prefer a sphere marker for legacy single-topic consumers
            spheres = [m for m in markers if m.type == Marker.SPHERE and m.action == Marker.ADD]
            self.pub_one.publish(spheres[0] if spheres else markers[0])


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
