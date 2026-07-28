#!/usr/bin/env python3
"""MediaPipe human pose estimation node.

Publishes:
  - /perception/landmarks (std_msgs/String JSON) — documented lightweight format
  - /perception/annotated_image (sensor_msgs/Image) optional
  - /perception/pose_lost (std_msgs/Bool)
"""
from __future__ import annotations

import json
import time

import cv2
import mediapipe as mp
import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import Bool, Header, String

# MediaPipe landmark name subset used by mapping
LANDMARK_NAMES = {
    mp.solutions.pose.PoseLandmark.RIGHT_WRIST: "right_wrist",
    mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER: "right_shoulder",
    mp.solutions.pose.PoseLandmark.LEFT_WRIST: "left_wrist",
    mp.solutions.pose.PoseLandmark.LEFT_SHOULDER: "left_shoulder",
    mp.solutions.pose.PoseLandmark.NOSE: "nose",
}


class PoseEstimatorNode:
    def __init__(self) -> None:
        rospy.init_node("pose_estimator_node")
        self.bridge = CvBridge()
        self.min_det = float(rospy.get_param("~min_detection_confidence", 0.5))
        self.min_trk = float(rospy.get_param("~min_tracking_confidence", 0.5))
        self.headless = bool(rospy.get_param("~headless", True))
        self.publish_viz = bool(rospy.get_param("~publish_visualization", True))
        self.frame_id = rospy.get_param("~frame_id", "camera_optical_frame")
        self.log_every = int(rospy.get_param("~log_every_n_frames", 30))
        self.pose_timeout = float(rospy.get_param("~pose_timeout_sec", 0.5))

        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=self.min_det,
            min_tracking_confidence=self.min_trk,
            model_complexity=int(rospy.get_param("~model_complexity", 1)),
        )
        self.drawer = mp.solutions.drawing_utils

        image_topic = rospy.get_param("~image_topic", "/camera/color/image_raw/compressed")
        transport = rospy.get_param("~transport", "compressed")
        if transport == "compressed" or image_topic.endswith("/compressed"):
            self.sub = rospy.Subscriber(
                image_topic, CompressedImage, self._compressed_cb, queue_size=1, buff_size=2**24
            )
        else:
            self.sub = rospy.Subscriber(image_topic, Image, self._raw_cb, queue_size=1, buff_size=2**24)

        self.lm_pub = rospy.Publisher("~landmarks", String, queue_size=1)
        self.lost_pub = rospy.Publisher("~pose_lost", Bool, queue_size=1)
        self.viz_pub = rospy.Publisher("~annotated_image", Image, queue_size=1) if self.publish_viz else None

        self._frames = 0
        self._last_pose_wall = 0.0
        rospy.loginfo("pose_estimator_node subscribed to %s", image_topic)

    def _compressed_cb(self, msg: CompressedImage) -> None:
        try:
            arr = np.frombuffer(msg.data, dtype=np.uint8)
            frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if frame is None:
                raise ValueError("imdecode returned None")
            self._process(frame, msg.header)
        except Exception as exc:  # noqa: BLE001
            rospy.logerr_throttle(5.0, "compressed image error: %s", exc)

    def _raw_cb(self, msg: Image) -> None:
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
            self._process(frame, msg.header)
        except CvBridgeError as exc:
            rospy.logerr_throttle(5.0, "cv_bridge error: %s", exc)

    def _process(self, frame: np.ndarray, header: Header) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.pose.process(rgb)
        h, w = frame.shape[:2]
        stamp = header.stamp.to_sec() if header.stamp else rospy.Time.now().to_sec()
        now_wall = time.time()

        landmarks_out = []
        pose_ok = False
        if result.pose_landmarks:
            pose_ok = True
            self._last_pose_wall = now_wall
            for enum_lm, name in LANDMARK_NAMES.items():
                lm = result.pose_landmarks.landmark[enum_lm]
                landmarks_out.append(
                    {
                        "name": name,
                        "x": float(lm.x),
                        "y": float(lm.y),
                        "z": float(lm.z),
                        "visibility": float(getattr(lm, "visibility", 1.0)),
                        "confidence": float(getattr(lm, "visibility", 1.0)),
                    }
                )

        payload = {
            "header": {
                "stamp": stamp,
                "frame_id": header.frame_id or self.frame_id,
                "image_width": w,
                "image_height": h,
            },
            "pose_detected": pose_ok,
            "landmarks": landmarks_out,
            # Coordinate convention note for consumers
            "coordinate_convention": "normalized_image_xy_in_0_1_mediapipe_z_relative",
        }
        self.lm_pub.publish(String(data=json.dumps(payload)))

        lost = (not pose_ok) or ((now_wall - self._last_pose_wall) > self.pose_timeout and self._last_pose_wall > 0)
        if not pose_ok and self._last_pose_wall == 0.0:
            lost = True
        self.lost_pub.publish(Bool(data=lost))

        if self.viz_pub is not None and result.pose_landmarks:
            annotated = frame.copy()
            self.drawer.draw_landmarks(annotated, result.pose_landmarks, self.mp_pose.POSE_CONNECTIONS)
            try:
                img_msg = self.bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
                img_msg.header = header
                img_msg.header.frame_id = header.frame_id or self.frame_id
                self.viz_pub.publish(img_msg)
            except CvBridgeError as exc:
                rospy.logerr_throttle(5.0, "viz publish error: %s", exc)
            if not self.headless:
                cv2.imshow("pose", annotated)
                cv2.waitKey(1)

        self._frames += 1
        if self._frames % max(self.log_every, 1) == 0:
            rospy.loginfo_throttle(10.0, "pose frames=%d detected=%s", self._frames, pose_ok)


if __name__ == "__main__":
    try:
        PoseEstimatorNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
