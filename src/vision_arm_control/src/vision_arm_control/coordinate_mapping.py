"""Coordinate conversion utilities for human-to-robot mapping.

Pipeline (conceptually):
  normalized image coords (MediaPipe) -> pixel coords -> camera-frame 3D
  -> TF to robot base -> workspace scaling / validation

This module is pure Python (numpy-only) so it can be unit-tested without ROS.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera intrinsics.

    Values marked as examples in config must not be treated as measured
    calibration unless explicitly documented.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def as_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass(frozen=True)
class Landmark3D:
    """A single landmark in a named frame."""

    x: float
    y: float
    z: float
    confidence: float = 1.0
    frame_id: str = "camera_optical_frame"


def normalized_to_pixel(
    u_norm: float,
    v_norm: float,
    width: int,
    height: int,
) -> Tuple[float, float]:
    """Convert MediaPipe normalized image coordinates to pixel coordinates.

    MediaPipe returns x,y in [0, 1] relative to image width/height.
    """
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    return u_norm * float(width), v_norm * float(height)


def pixel_to_camera_ray(
    u_px: float,
    v_px: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Return a unit direction vector in the camera optical frame."""
    x = (u_px - intrinsics.cx) / intrinsics.fx
    y = (v_px - intrinsics.cy) / intrinsics.fy
    vec = np.array([x, y, 1.0], dtype=np.float64)
    norm = np.linalg.norm(vec)
    if norm < 1e-12:
        raise ValueError("degenerate ray direction")
    return vec / norm


def pixel_depth_to_camera_point(
    u_px: float,
    v_px: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
) -> np.ndarray:
    """Back-project a pixel + metric depth (meters) to a camera-frame point.

    If depth is relative (monocular), do not call this with relative values
    and claim metric accuracy. Use relative_depth_to_proxy_z instead and
    document the approximation.
    """
    if depth_m <= 0.0:
        raise ValueError("depth_m must be positive for metric back-projection")
    x = (u_px - intrinsics.cx) * depth_m / intrinsics.fx
    y = (v_px - intrinsics.cy) * depth_m / intrinsics.fy
    return np.array([x, y, depth_m], dtype=np.float64)


def relative_depth_to_proxy_z(
    relative_depth: float,
    scale: float,
    offset: float,
) -> float:
    """Map relative monocular depth to a proxy Z using affine parameters.

    This is an approximation for teleoperation demos only. It is NOT metric
    depth unless scale/offset were calibrated against a depth sensor.
    """
    return float(scale * relative_depth + offset)


def transform_point(point_xyz: Sequence[float], T_parent_child: np.ndarray) -> np.ndarray:
    """Apply a 4x4 homogeneous transform to a 3D point."""
    T = np.asarray(T_parent_child, dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError("T_parent_child must be 4x4")
    p = np.array([point_xyz[0], point_xyz[1], point_xyz[2], 1.0], dtype=np.float64)
    out = T @ p
    return out[:3]


def shoulder_relative_wrist(
    shoulder: Sequence[float],
    wrist: Sequence[float],
) -> np.ndarray:
    """Return wrist displacement relative to shoulder (same frame)."""
    s = np.asarray(shoulder, dtype=np.float64)
    w = np.asarray(wrist, dtype=np.float64)
    if s.shape != (3,) or w.shape != (3,):
        raise ValueError("shoulder and wrist must be length-3")
    return w - s


def scale_displacement(
    displacement: Sequence[float],
    scale: Sequence[float],
    origin: Sequence[float],
) -> np.ndarray:
    """Map a human-frame displacement into robot workspace coordinates.

    target = origin + scale * displacement
    """
    d = np.asarray(displacement, dtype=np.float64)
    s = np.asarray(scale, dtype=np.float64)
    o = np.asarray(origin, dtype=np.float64)
    if d.shape != (3,) or s.shape != (3,) or o.shape != (3,):
        raise ValueError("displacement, scale, and origin must be length-3")
    return o + s * d


def confidence_ok(confidence: float, min_confidence: float) -> bool:
    return float(confidence) >= float(min_confidence)


def select_landmark(
    landmarks: Iterable[dict],
    name: str,
) -> Optional[dict]:
    """Find a landmark dict by name from a list of {name,x,y,z,confidence}."""
    for lm in landmarks:
        if lm.get("name") == name:
            return lm
    return None


def map_wrist_end_effector_position(
    wrist_norm_xy: Tuple[float, float],
    image_size: Tuple[int, int],
    intrinsics: CameraIntrinsics,
    T_base_camera: np.ndarray,
    depth_m: Optional[float],
    relative_depth: Optional[float],
    relative_depth_scale: float,
    relative_depth_offset: float,
    use_metric_depth: bool,
    shoulder_norm_xy: Optional[Tuple[float, float]] = None,
    workspace_origin: Sequence[float] = (0.4, 0.0, 0.4),
    workspace_scale: Sequence[float] = (0.5, 0.5, 0.5),
    mapping_mode: str = "image_plane_scaled",
) -> np.ndarray:
    """Map human wrist observation to a robot base-frame end-effector position.

    mapping_mode:
      - image_plane_scaled: project to camera ray, apply proxy/metric depth,
        transform to base (recommended scaffold).
      - shoulder_relative: use shoulder→wrist pixel displacement scaled into
        workspace (no depth required; teleoperation-style).

    Returns a (3,) position in the robot base frame.
    This is **end-effector position teleoperation**, not full pose mimicry.
    """
    width, height = image_size
    u_px, v_px = normalized_to_pixel(wrist_norm_xy[0], wrist_norm_xy[1], width, height)

    if mapping_mode == "shoulder_relative":
        if shoulder_norm_xy is None:
            raise ValueError("shoulder_norm_xy required for shoulder_relative mode")
        su, sv = normalized_to_pixel(shoulder_norm_xy[0], shoulder_norm_xy[1], width, height)
        # Normalize displacement by image size to keep scale roughly body-relative.
        disp = np.array(
            [(u_px - su) / float(width), (v_px - sv) / float(height), 0.0],
            dtype=np.float64,
        )
        return scale_displacement(disp, workspace_scale, workspace_origin)

    if use_metric_depth:
        if depth_m is None or depth_m <= 0.0:
            raise ValueError("metric depth required when use_metric_depth=True")
        cam_pt = pixel_depth_to_camera_point(u_px, v_px, depth_m, intrinsics)
    else:
        # Relative monocular depth is visualization-grade unless calibrated.
        rel = 1.0 if relative_depth is None else float(relative_depth)
        z_proxy = relative_depth_to_proxy_z(rel, relative_depth_scale, relative_depth_offset)
        if z_proxy <= 0.0:
            z_proxy = max(relative_depth_offset, 0.5)
        cam_pt = pixel_depth_to_camera_point(u_px, v_px, z_proxy, intrinsics)

    base_pt = transform_point(cam_pt, T_base_camera)
    return base_pt
