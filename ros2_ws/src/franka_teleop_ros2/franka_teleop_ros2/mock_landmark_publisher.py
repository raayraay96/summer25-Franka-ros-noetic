#!/usr/bin/env python3
"""Publish smooth synthetic MediaPipe-like landmarks for simulation demos.

Also publishes an RGB Image panel of the mock arm so RViz / recording can
show "Mock landmark input" driven by the same trajectory as the mapper.
"""
from __future__ import annotations

import json
from collections import deque

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

from franka_teleop_ros2.motion_profile import (
    DEMO_PERIOD_SEC,
    draw_mock_landmark_panel_rgb,
    landmark_payload,
    wrist_xy,
)


class MockLandmarkPublisher(Node):
    def __init__(self) -> None:
        super().__init__("mock_landmark_publisher")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("topic", "/perception/landmarks")
        self.declare_parameter("image_topic", "/perception/mock_landmarks_image")
        self.declare_parameter("motion_scale", 0.18)
        self.declare_parameter("period_sec", DEMO_PERIOD_SEC)
        self.declare_parameter("publish_image", True)
        self.declare_parameter("panel_width", 320)
        self.declare_parameter("panel_height", 240)

        rate = float(self.get_parameter("rate_hz").value)
        topic = str(self.get_parameter("topic").value)
        self.width = int(self.get_parameter("image_width").value)
        self.height = int(self.get_parameter("image_height").value)
        self.scale = float(self.get_parameter("motion_scale").value)
        self.period = float(self.get_parameter("period_sec").value)
        self.publish_image = bool(self.get_parameter("publish_image").value)
        self.panel_w = int(self.get_parameter("panel_width").value)
        self.panel_h = int(self.get_parameter("panel_height").value)

        self.pub = self.create_publisher(String, topic, 10)
        self.img_pub = None
        if self.publish_image:
            self.img_pub = self.create_publisher(
                Image, str(self.get_parameter("image_topic").value), 10
            )
        self.t0 = self.get_clock().now()
        self.trail: deque = deque(maxlen=40)
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._tick)
        self.get_logger().info(
            f"mock_landmark_publisher @ {rate:.1f} Hz → {topic} "
            f"(scale={self.scale}, period={self.period:.1f}s, simulation input)"
        )

    def _tick(self) -> None:
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        stamp = self.get_clock().now().nanoseconds * 1e-9
        payload = landmark_payload(
            t,
            scale=self.scale,
            period=self.period,
            image_width=self.width,
            image_height=self.height,
            stamp=stamp,
        )
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)

        wr = wrist_xy(t, scale=self.scale, period=self.period)
        self.trail.append(wr)

        if self.img_pub is not None:
            rgb = draw_mock_landmark_panel_rgb(
                t,
                width=self.panel_w,
                height=self.panel_h,
                scale=self.scale,
                period=self.period,
                trail=list(self.trail),
            )
            img = Image()
            img.header.stamp = self.get_clock().now().to_msg()
            img.header.frame_id = "camera_optical_frame"
            img.height = self.panel_h
            img.width = self.panel_w
            img.encoding = "rgb8"
            img.is_bigendian = 0
            img.step = self.panel_w * 3
            img.data = rgb
            self.img_pub.publish(img)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MockLandmarkPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
