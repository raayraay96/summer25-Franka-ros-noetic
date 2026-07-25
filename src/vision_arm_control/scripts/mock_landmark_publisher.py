#!/usr/bin/env python3
"""Publish synthetic MediaPipe-like landmarks for simulation/mock demos.

Does not require a camera or MediaPipe. Useful on headless CI and Scholar.
"""
from __future__ import annotations

import json
import math

import rospy
from std_msgs.msg import String


def main() -> None:
    rospy.init_node("mock_landmark_publisher")
    pub = rospy.Publisher(
        rospy.get_param("~topic", "/pose_estimator_node/landmarks"),
        String,
        queue_size=1,
    )
    rate_hz = float(rospy.get_param("~rate", 15.0))
    width = int(rospy.get_param("~image_width", 640))
    height = int(rospy.get_param("~image_height", 480))
    rate = rospy.Rate(rate_hz)
    t0 = rospy.Time.now().to_sec()
    rospy.loginfo("mock_landmark_publisher @ %.1f Hz", rate_hz)
    while not rospy.is_shutdown():
        t = rospy.Time.now().to_sec() - t0
        # Oscillate wrist in normalized image coordinates
        wx = 0.5 + 0.15 * math.sin(t)
        wy = 0.45 + 0.10 * math.cos(0.7 * t)
        payload = {
            "header": {
                "stamp": rospy.Time.now().to_sec(),
                "frame_id": "camera_optical_frame",
                "image_width": width,
                "image_height": height,
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
                    "x": wx,
                    "y": wy,
                    "z": 0.0,
                    "visibility": 1.0,
                    "confidence": 1.0,
                },
            ],
            "coordinate_convention": "normalized_image_xy_in_0_1_mediapipe_z_relative",
            "source": "mock",
        }
        pub.publish(String(data=json.dumps(payload)))
        rate.sleep()


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
