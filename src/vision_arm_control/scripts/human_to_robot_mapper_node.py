#!/usr/bin/env python3
"""Map human landmarks to robot base-frame end-effector position targets.

Publishes geometry_msgs/PoseStamped on ~target_pose.

This implements **end-effector position teleoperation**, not full-body mimicry.
Depth is optional and treated as relative unless use_metric_depth is set with
a calibrated source.
"""
from __future__ import annotations

import json
import math

import numpy as np
import rospy
import tf.transformations as tft
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import String

from vision_arm_control.coordinate_mapping import (
    CameraIntrinsics,
    map_wrist_end_effector_position,
)
from vision_arm_control.pose_timeout import PoseTimeoutMonitor
from vision_arm_control.trajectory_filter import LowPassFilter3D, VelocityLimiter3D
from vision_arm_control.workspace_limits import AxisAlignedBounds, validate_or_clamp_position


def _rpy_to_matrix(rpy):
    return tft.euler_matrix(rpy[0], rpy[1], rpy[2])


def _xyz_rpy_to_T(xyz, rpy):
    T = _rpy_to_matrix(rpy)
    T[0:3, 3] = xyz
    return T


class HumanToRobotMapperNode:
    def __init__(self) -> None:
        rospy.init_node("human_to_robot_mapper_node")

        self.mode = rospy.get_param("~mapping_mode", "shoulder_relative")
        self.min_conf = float(rospy.get_param("~min_landmark_confidence", 0.5))
        self.use_metric_depth = bool(rospy.get_param("~use_metric_depth", False))
        self.rel_scale = float(rospy.get_param("~relative_depth_scale", 1.0))
        self.rel_offset = float(rospy.get_param("~relative_depth_offset", 0.8))
        self.base_frame = rospy.get_param("~base_frame", "panda_link0")
        self.camera_frame = rospy.get_param("~camera_frame", "camera_optical_frame")
        self.workspace_origin = list(rospy.get_param("~workspace_origin", [0.45, 0.0, 0.45]))
        self.workspace_scale = list(rospy.get_param("~workspace_scale", [0.55, 0.55, 0.35]))
        self.workspace_mode = rospy.get_param("~workspace_mode", "reject")

        self.intrinsics = CameraIntrinsics(
            fx=float(rospy.get_param("~fx", 600.0)),
            fy=float(rospy.get_param("~fy", 600.0)),
            cx=float(rospy.get_param("~cx", 320.0)),
            cy=float(rospy.get_param("~cy", 240.0)),
            width=int(rospy.get_param("~image_width", 640)),
            height=int(rospy.get_param("~image_height", 480)),
        )
        cam_xyz = list(rospy.get_param("~camera_in_base_xyz", [0.5, 0.0, 0.8]))
        cam_rpy = list(rospy.get_param("~camera_in_base_rpy", [math.pi, 0.0, 0.0]))
        # T_base_camera: point_base = T * point_camera
        self.T_base_camera = _xyz_rpy_to_T(cam_xyz, cam_rpy)
        rospy.logwarn_once(
            "Camera extrinsics are EXAMPLE values unless calibration status is 'measured'."
        )

        ws = rospy.get_param(
            "~workspace",
            {"x_min": 0.25, "x_max": 0.75, "y_min": -0.4, "y_max": 0.4, "z_min": 0.05, "z_max": 0.8},
        )
        self.bounds = AxisAlignedBounds(**{k: float(ws[k]) for k in ws})
        self.pose_timeout = PoseTimeoutMonitor(float(rospy.get_param("~pose_timeout_sec", 0.5)))
        self.lpf = LowPassFilter3D(alpha=float(rospy.get_param("~filter_alpha", 0.3)))
        self.vlim = VelocityLimiter3D(max_linear_velocity=float(rospy.get_param("~max_linear_velocity", 0.2)))
        self._last_t = None

        fixed_rpy = list(rospy.get_param("~fixed_orientation_rpy", [math.pi, 0.0, 0.0]))
        self.fixed_q = tft.quaternion_from_euler(*fixed_rpy)

        lm_topic = rospy.get_param("~landmarks_topic", "/pose_estimator_node/landmarks")
        self.sub = rospy.Subscriber(lm_topic, String, self._cb, queue_size=1)
        self.pub = rospy.Publisher("~target_pose", PoseStamped, queue_size=1)
        self.reject_pub = rospy.Publisher("~rejected", String, queue_size=1)
        rospy.loginfo("human_to_robot_mapper_node mode=%s", self.mode)

    def _pick(self, landmarks, name):
        for lm in landmarks:
            if lm.get("name") == name:
                return lm
        return None

    def _cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        now = rospy.Time.now().to_sec()
        if not data.get("pose_detected"):
            return
        landmarks = data.get("landmarks") or []
        wrist = self._pick(landmarks, "right_wrist")
        shoulder = self._pick(landmarks, "right_shoulder")
        if wrist is None:
            return
        if float(wrist.get("confidence", wrist.get("visibility", 0.0))) < self.min_conf:
            return
        self.pose_timeout.note_pose(now)
        if not self.pose_timeout.is_valid(now):
            return

        header = data.get("header") or {}
        width = int(header.get("image_width", self.intrinsics.width))
        height = int(header.get("image_height", self.intrinsics.height))
        wrist_xy = (float(wrist["x"]), float(wrist["y"]))
        shoulder_xy = None
        if shoulder is not None:
            shoulder_xy = (float(shoulder["x"]), float(shoulder["y"]))

        try:
            target = map_wrist_end_effector_position(
                wrist_norm_xy=wrist_xy,
                image_size=(width, height),
                intrinsics=self.intrinsics,
                T_base_camera=self.T_base_camera,
                depth_m=None,
                relative_depth=None,
                relative_depth_scale=self.rel_scale,
                relative_depth_offset=self.rel_offset,
                use_metric_depth=self.use_metric_depth,
                shoulder_norm_xy=shoulder_xy,
                workspace_origin=self.workspace_origin,
                workspace_scale=self.workspace_scale,
                mapping_mode=self.mode if shoulder_xy is not None else "image_plane_scaled",
            )
        except Exception as exc:  # noqa: BLE001
            self.reject_pub.publish(String(data=f"mapping_error:{exc}"))
            return

        pos, ok, reason = validate_or_clamp_position(target, self.bounds, mode=self.workspace_mode)
        if not ok or pos is None:
            self.reject_pub.publish(String(data=reason))
            return

        filtered = self.lpf.update(pos)
        dt = 0.05 if self._last_t is None else max(now - self._last_t, 1e-3)
        self._last_t = now
        limited = self.vlim.update(filtered, dt)

        out = PoseStamped()
        out.header.stamp = rospy.Time.now()
        out.header.frame_id = self.base_frame
        out.pose.position.x = float(limited[0])
        out.pose.position.y = float(limited[1])
        out.pose.position.z = float(limited[2])
        out.pose.orientation.x = float(self.fixed_q[0])
        out.pose.orientation.y = float(self.fixed_q[1])
        out.pose.orientation.z = float(self.fixed_q[2])
        out.pose.orientation.w = float(self.fixed_q[3])
        self.pub.publish(out)


if __name__ == "__main__":
    try:
        HumanToRobotMapperNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
