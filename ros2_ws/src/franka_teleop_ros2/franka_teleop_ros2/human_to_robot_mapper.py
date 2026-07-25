#!/usr/bin/env python3
"""Map mock/human landmarks to base-frame end-effector targets (ROS 2).

Depth is visualization-only for this simulation path (shoulder-relative 2D).
"""
from __future__ import annotations

import json
import math
from typing import Optional

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import String

from franka_teleop_ros2.coordinate_mapping import (
    CameraIntrinsics,
    map_wrist_end_effector_position,
)
from franka_teleop_ros2.trajectory_filter import LowPassFilter3D, VelocityLimiter3D
from franka_teleop_ros2.workspace_limits import AxisAlignedBounds, validate_or_clamp_position


def _quat_from_rpy(roll: float, pitch: float, yaw: float):
    """Return (x,y,z,w) quaternion from RPY."""
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    return x, y, z, w


class HumanToRobotMapper(Node):
    def __init__(self) -> None:
        super().__init__("human_to_robot_mapper")
        # Parameters (overridden by YAML under this node namespace)
        self.declare_parameter("landmarks_topic", "/perception/landmarks")
        self.declare_parameter("target_topic", "/teleop/target_pose")
        self.declare_parameter("rejected_topic", "/teleop/mapping_rejected")
        self.declare_parameter("mapping_mode", "shoulder_relative")
        self.declare_parameter("min_landmark_confidence", 0.5)
        self.declare_parameter("use_metric_depth", False)
        self.declare_parameter("base_frame", "panda_link0")
        self.declare_parameter("workspace_origin", [0.40, 0.0, 0.45])
        self.declare_parameter("workspace_scale", [0.45, 0.50, 0.30])
        self.declare_parameter("filter_alpha", 0.35)
        self.declare_parameter("max_linear_velocity", 0.25)
        self.declare_parameter("workspace.x_min", 0.25)
        self.declare_parameter("workspace.x_max", 0.70)
        self.declare_parameter("workspace.y_min", -0.35)
        self.declare_parameter("workspace.y_max", 0.35)
        self.declare_parameter("workspace.z_min", 0.10)
        self.declare_parameter("workspace.z_max", 0.70)
        self.declare_parameter("fixed_orientation_rpy", [math.pi, 0.0, 0.0])
        self.declare_parameter("fx", 600.0)
        self.declare_parameter("fy", 600.0)
        self.declare_parameter("cx", 320.0)
        self.declare_parameter("cy", 240.0)
        # Example extrinsics (not measured)
        self.declare_parameter("camera_in_base_xyz", [0.5, 0.0, 0.8])
        self.declare_parameter("camera_in_base_rpy", [math.pi, 0.0, 0.0])

        self.mode = str(self.get_parameter("mapping_mode").value)
        self.min_conf = float(self.get_parameter("min_landmark_confidence").value)
        self.use_metric = bool(self.get_parameter("use_metric_depth").value)
        self.base_frame = str(self.get_parameter("base_frame").value)
        self.origin = list(self.get_parameter("workspace_origin").value)
        self.scale = list(self.get_parameter("workspace_scale").value)
        self.bounds = AxisAlignedBounds(
            x_min=float(self.get_parameter("workspace.x_min").value),
            x_max=float(self.get_parameter("workspace.x_max").value),
            y_min=float(self.get_parameter("workspace.y_min").value),
            y_max=float(self.get_parameter("workspace.y_max").value),
            z_min=float(self.get_parameter("workspace.z_min").value),
            z_max=float(self.get_parameter("workspace.z_max").value),
        )
        self.lpf = LowPassFilter3D(alpha=float(self.get_parameter("filter_alpha").value))
        self.vlim = VelocityLimiter3D(
            max_linear_velocity=float(self.get_parameter("max_linear_velocity").value)
        )
        rpy = list(self.get_parameter("fixed_orientation_rpy").value)
        self.qx, self.qy, self.qz, self.qw = _quat_from_rpy(*[float(v) for v in rpy])
        self.intrinsics = CameraIntrinsics(
            fx=float(self.get_parameter("fx").value),
            fy=float(self.get_parameter("fy").value),
            cx=float(self.get_parameter("cx").value),
            cy=float(self.get_parameter("cy").value),
            width=640,
            height=480,
        )
        self.T_base_camera = np.eye(4)
        xyz = list(self.get_parameter("camera_in_base_xyz").value)
        self.T_base_camera[0:3, 3] = [float(v) for v in xyz]
        self._last_t: Optional[float] = None

        lm_topic = str(self.get_parameter("landmarks_topic").value)
        self.sub = self.create_subscription(String, lm_topic, self._cb, 10)
        self.pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("target_topic").value), 10
        )
        self.rej = self.create_publisher(
            String, str(self.get_parameter("rejected_topic").value), 10
        )
        self.get_logger().info(
            f"human_to_robot_mapper mode={self.mode} depth_for_control={self.use_metric}"
        )
        self.get_logger().warn(
            "Camera extrinsics/intrinsics are EXAMPLE values unless calibrated."
        )

    def _pick(self, landmarks, name: str):
        for lm in landmarks:
            if lm.get("name") == name:
                return lm
        return None

    def _cb(self, msg: String) -> None:
        try:
            data = json.loads(msg.data)
        except json.JSONDecodeError:
            return
        if not data.get("pose_detected"):
            return
        landmarks = data.get("landmarks") or []
        wrist = self._pick(landmarks, "right_wrist")
        shoulder = self._pick(landmarks, "right_shoulder")
        if wrist is None:
            return
        conf = float(wrist.get("confidence", wrist.get("visibility", 0.0)))
        if conf < self.min_conf:
            return
        header = data.get("header") or {}
        width = int(header.get("image_width", 640))
        height = int(header.get("image_height", 480))
        wrist_xy = (float(wrist["x"]), float(wrist["y"]))
        shoulder_xy = (
            (float(shoulder["x"]), float(shoulder["y"])) if shoulder else None
        )
        try:
            target = map_wrist_end_effector_position(
                wrist_norm_xy=wrist_xy,
                image_size=(width, height),
                intrinsics=self.intrinsics,
                T_base_camera=self.T_base_camera,
                depth_m=None,
                relative_depth=None,
                relative_depth_scale=1.0,
                relative_depth_offset=0.8,
                use_metric_depth=self.use_metric,
                shoulder_norm_xy=shoulder_xy,
                workspace_origin=self.origin,
                workspace_scale=self.scale,
                mapping_mode=self.mode if shoulder_xy is not None else "image_plane_scaled",
            )
        except Exception as exc:  # noqa: BLE001
            r = String()
            r.data = f"mapping_error:{exc}"
            self.rej.publish(r)
            return

        pos, ok, reason = validate_or_clamp_position(target, self.bounds, mode="clamp")
        if not ok or pos is None:
            r = String()
            r.data = reason
            self.rej.publish(r)
            return

        now = self.get_clock().now().nanoseconds * 1e-9
        dt = 0.05 if self._last_t is None else max(now - self._last_t, 1e-3)
        self._last_t = now
        filtered = self.lpf.update(pos)
        limited = self.vlim.update(filtered, dt)

        out = PoseStamped()
        out.header.stamp = self.get_clock().now().to_msg()
        out.header.frame_id = self.base_frame
        out.pose.position.x = float(limited[0])
        out.pose.position.y = float(limited[1])
        out.pose.position.z = float(limited[2])
        out.pose.orientation.x = self.qx
        out.pose.orientation.y = self.qy
        out.pose.orientation.z = self.qz
        out.pose.orientation.w = self.qw
        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HumanToRobotMapper()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
