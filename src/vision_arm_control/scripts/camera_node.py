#!/usr/bin/env python3
"""Publish webcam frames as sensor_msgs/CompressedImage (or Image).

For rosbag replay or external cameras, prefer perception_only.launch with
an external image topic instead of this node.
"""
from __future__ import annotations

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge


class CameraNode:
    def __init__(self) -> None:
        rospy.init_node("camera_node", anonymous=True)
        self.device_index = int(rospy.get_param("~device_index", 0))
        self.width = int(rospy.get_param("~width", 640))
        self.height = int(rospy.get_param("~height", 480))
        self.fps = float(rospy.get_param("~fps", 15.0))
        self.frame_id = rospy.get_param("~frame_id", "camera_optical_frame")
        self.publish_raw = bool(rospy.get_param("~publish_raw", False))
        self.log_every = int(rospy.get_param("~log_every_n_frames", 30))

        topic = rospy.get_param("~image_topic", "/camera/color/image_raw/compressed")
        self.pub = rospy.Publisher(topic, CompressedImage, queue_size=2)
        self.raw_pub = None
        if self.publish_raw:
            raw_topic = rospy.get_param("~raw_image_topic", "/camera/color/image_raw")
            self.raw_pub = rospy.Publisher(raw_topic, Image, queue_size=2)
            self.bridge = CvBridge()
        else:
            self.bridge = None

        self.cap = cv2.VideoCapture(self.device_index)
        if self.width > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        if self.height > 0:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        if not self.cap.isOpened():
            rospy.logerr("Failed to open camera device %s", self.device_index)
        else:
            rospy.loginfo("Camera node ready on device %s @ %.1f Hz", self.device_index, self.fps)
        self._frame_count = 0

    def run(self) -> None:
        rate = rospy.Rate(max(self.fps, 1.0))
        while not rospy.is_shutdown():
            ok, frame = self.cap.read()
            if not ok:
                rospy.logwarn_throttle(5.0, "Failed to capture frame")
                rate.sleep()
                continue
            stamp = rospy.Time.now()
            try:
                msg = CompressedImage()
                msg.header.stamp = stamp
                msg.header.frame_id = self.frame_id
                msg.format = "jpeg"
                ok_enc, buf = cv2.imencode(".jpg", frame)
                if not ok_enc:
                    raise RuntimeError("JPEG encode failed")
                msg.data = buf.tobytes()
                self.pub.publish(msg)
                if self.raw_pub is not None and self.bridge is not None:
                    raw = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                    raw.header = msg.header
                    self.raw_pub.publish(raw)
            except Exception as exc:  # noqa: BLE001 — surface any capture/publish error
                rospy.logerr_throttle(5.0, "Error publishing image: %s", exc)
            self._frame_count += 1
            if self._frame_count % max(self.log_every, 1) == 0:
                rospy.logdebug("Published %d frames", self._frame_count)
            rate.sleep()
        self.cap.release()


if __name__ == "__main__":
    try:
        CameraNode().run()
    except rospy.ROSInterruptException:
        pass
