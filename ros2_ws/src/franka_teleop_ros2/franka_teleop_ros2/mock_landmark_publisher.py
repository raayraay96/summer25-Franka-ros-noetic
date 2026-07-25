#!/usr/bin/env python3
"""Publish smooth synthetic MediaPipe-like landmarks for simulation demos."""
from __future__ import annotations

import json
import math

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


class MockLandmarkPublisher(Node):
    def __init__(self) -> None:
        super().__init__("mock_landmark_publisher")
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)
        self.declare_parameter("topic", "/perception/landmarks")
        self.declare_parameter("motion_scale", 0.12)

        rate = float(self.get_parameter("rate_hz").value)
        topic = str(self.get_parameter("topic").value)
        self.width = int(self.get_parameter("image_width").value)
        self.height = int(self.get_parameter("image_height").value)
        self.scale = float(self.get_parameter("motion_scale").value)
        self.pub = self.create_publisher(String, topic, 10)
        self.t0 = self.get_clock().now()
        self.timer = self.create_timer(1.0 / max(rate, 1.0), self._tick)
        self.get_logger().info(
            f"mock_landmark_publisher @ {rate:.1f} Hz → {topic} (simulation input)"
        )

    def _tick(self) -> None:
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        # Smooth Lissajous-like wrist motion in normalized image coords
        # Pause periodically for pose-timeout demos
        cycle = t % 12.0
        if 10.0 <= cycle < 11.0:
            # hold last region (pause)
            wx, wy = 0.55, 0.45
        else:
            wx = 0.50 + self.scale * math.sin(0.6 * t)
            wy = 0.42 + self.scale * 0.7 * math.cos(0.45 * t)

        payload = {
            "header": {
                "stamp": self.get_clock().now().nanoseconds * 1e-9,
                "frame_id": "camera_optical_frame",
                "image_width": self.width,
                "image_height": self.height,
            },
            "pose_detected": True,
            "landmarks": [
                {
                    "name": "right_shoulder",
                    "x": 0.45,
                    "y": 0.35,
                    "z": 0.0,
                    "visibility": 1.0,
                    "confidence": 1.0,
                },
                {
                    "name": "right_wrist",
                    "x": float(wx),
                    "y": float(wy),
                    "z": 0.0,
                    "visibility": 1.0,
                    "confidence": 1.0,
                },
            ],
            "coordinate_convention": "normalized_image_xy_in_0_1_mediapipe_z_relative",
            "source": "mock",
            "depth_used_for_control": False,
        }
        msg = String()
        msg.data = json.dumps(payload)
        self.pub.publish(msg)


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
