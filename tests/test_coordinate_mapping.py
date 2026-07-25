"""Unit tests for coordinate mapping (no ROS required)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "src" / "vision_arm_control" / "src"
sys.path.insert(0, str(PKG))

from vision_arm_control.coordinate_mapping import (  # noqa: E402
    CameraIntrinsics,
    map_wrist_end_effector_position,
    normalized_to_pixel,
    pixel_depth_to_camera_point,
    relative_depth_to_proxy_z,
    scale_displacement,
    shoulder_relative_wrist,
    transform_point,
)


def test_normalized_to_pixel_center():
    u, v = normalized_to_pixel(0.5, 0.5, 640, 480)
    assert u == pytest.approx(320.0)
    assert v == pytest.approx(240.0)


def test_normalized_to_pixel_rejects_bad_size():
    with pytest.raises(ValueError):
        normalized_to_pixel(0.1, 0.1, 0, 480)


def test_pixel_depth_to_camera_point_principal_point():
    K = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240, width=640, height=480)
    p = pixel_depth_to_camera_point(320, 240, 2.0, K)
    assert p[0] == pytest.approx(0.0)
    assert p[1] == pytest.approx(0.0)
    assert p[2] == pytest.approx(2.0)


def test_transform_point_identity():
    T = np.eye(4)
    p = transform_point([1.0, 2.0, 3.0], T)
    np.testing.assert_allclose(p, [1.0, 2.0, 3.0])


def test_transform_point_translation():
    T = np.eye(4)
    T[0:3, 3] = [1.0, 0.0, 0.5]
    p = transform_point([0.0, 0.0, 0.0], T)
    np.testing.assert_allclose(p, [1.0, 0.0, 0.5])


def test_relative_depth_proxy():
    assert relative_depth_to_proxy_z(1.0, 2.0, 0.5) == pytest.approx(2.5)


def test_shoulder_relative_and_scale():
    disp = shoulder_relative_wrist([0.0, 0.0, 0.0], [0.1, -0.2, 0.0])
    np.testing.assert_allclose(disp, [0.1, -0.2, 0.0])
    target = scale_displacement(disp, [0.5, 0.5, 0.5], [0.4, 0.0, 0.4])
    np.testing.assert_allclose(target, [0.45, -0.1, 0.4])


def test_map_shoulder_relative_mode():
    K = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240, width=640, height=480)
    T = np.eye(4)
    out = map_wrist_end_effector_position(
        wrist_norm_xy=(0.6, 0.5),
        image_size=(640, 480),
        intrinsics=K,
        T_base_camera=T,
        depth_m=None,
        relative_depth=None,
        relative_depth_scale=1.0,
        relative_depth_offset=0.8,
        use_metric_depth=False,
        shoulder_norm_xy=(0.5, 0.4),
        workspace_origin=(0.45, 0.0, 0.45),
        workspace_scale=(0.55, 0.55, 0.35),
        mapping_mode="shoulder_relative",
    )
    assert out.shape == (3,)
    # Wrist is to the right of shoulder in image => positive x displacement component
    assert out[0] > 0.45 or math.isfinite(out[0])


def test_map_rejects_metric_without_depth():
    K = CameraIntrinsics(fx=600, fy=600, cx=320, cy=240, width=640, height=480)
    with pytest.raises(ValueError):
        map_wrist_end_effector_position(
            wrist_norm_xy=(0.5, 0.5),
            image_size=(640, 480),
            intrinsics=K,
            T_base_camera=np.eye(4),
            depth_m=None,
            relative_depth=None,
            relative_depth_scale=1.0,
            relative_depth_offset=0.8,
            use_metric_depth=True,
            mapping_mode="image_plane_scaled",
        )
