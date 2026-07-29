"""Unit tests for SEW-inspired retargeting geometry."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.landmarks.schema import (  # noqa: E402
    ArmLandmarks,
    LandmarkPoint,
    RetargetingStatus,
)
from vision_arm_control.retargeting.base import RetargetingConfig, create_retargeter  # noqa: E402
from vision_arm_control.retargeting.sew_orientation import (  # noqa: E402
    SEWOrientationRetargeter,
    arm_plane_normal,
    elbow_bend_angle,
)
from vision_arm_control.retargeting.shoulder_relative import (  # noqa: E402
    ShoulderRelativeRetargeter,
)


def _pt(x, y, z, c=1.0, name=""):
    return LandmarkPoint(x=x, y=y, z=z, confidence=c, name=name)


def _arm(s, e, w, ts=1.0):
    return ArmLandmarks(
        shoulder=_pt(*s, name="right_shoulder"),
        elbow=_pt(*e, name="right_elbow"),
        wrist=_pt(*w, name="right_wrist"),
        timestamp=ts,
    )


def test_unit_vectors_and_elbow_angle_extension():
    # Fully extended along +x
    lm = _arm((0, 0, 0), (1, 0, 0), (2, 0, 0))
    r = SEWOrientationRetargeter(RetargetingConfig(name="sew_orientation", treat_z_as_image_relative=False))
    out = r.retarget(lm)
    assert out.status == RetargetingStatus.OK
    assert out.upper_arm_unit is not None
    np.testing.assert_allclose(out.upper_arm_unit, [1, 0, 0], atol=1e-9)
    np.testing.assert_allclose(out.lower_arm_unit, [1, 0, 0], atol=1e-9)
    assert out.elbow_bend_rad == pytest.approx(0.0, abs=1e-6)


def test_elbow_angle_right_angle():
    upper = np.array([1.0, 0.0, 0.0])
    lower = np.array([0.0, 1.0, 0.0])
    assert elbow_bend_angle(upper, lower) == pytest.approx(math.pi / 2, rel=1e-6)


def test_arm_plane_normal_orthogonal():
    n, st = arm_plane_normal(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), 1e-6)
    assert st == RetargetingStatus.OK
    np.testing.assert_allclose(n, [0, 0, 1], atol=1e-9)


def test_collinear_arm_plane():
    n, st = arm_plane_normal(np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), 1e-3)
    assert n is None
    assert st == RetargetingStatus.COLLINEAR_ARM_PLANE


def test_missing_keypoints():
    r = SEWOrientationRetargeter()
    base = _arm((0.4, 0.4, 0), (0.5, 0.45, 0), (0.6, 0.5, 0))
    for field, status in (
        ("shoulder", RetargetingStatus.MISSING_SHOULDER),
        ("elbow", RetargetingStatus.MISSING_ELBOW),
        ("wrist", RetargetingStatus.MISSING_WRIST),
    ):
        kwargs = {
            "shoulder": base.shoulder,
            "elbow": base.elbow,
            "wrist": base.wrist,
            "timestamp": 1.0,
        }
        kwargs[field] = None
        out = r.retarget(ArmLandmarks(**kwargs))
        assert out.status == status
        assert out.position is None


def test_near_zero_upper_arm():
    r = SEWOrientationRetargeter(RetargetingConfig(name="sew_orientation", min_segment_norm=1e-3))
    out = r.retarget(_arm((0.5, 0.5, 0), (0.5, 0.5, 0), (0.7, 0.5, 0)))
    assert out.status == RetargetingStatus.UPPER_ARM_NORM_NEAR_ZERO


def test_near_zero_lower_arm():
    r = SEWOrientationRetargeter(RetargetingConfig(name="sew_orientation", min_segment_norm=1e-3))
    out = r.retarget(_arm((0.4, 0.5, 0), (0.6, 0.5, 0), (0.6, 0.5, 0)))
    assert out.status == RetargetingStatus.LOWER_ARM_NORM_NEAR_ZERO


def test_low_confidence():
    r = SEWOrientationRetargeter(RetargetingConfig(min_confidence=0.5))
    lm = ArmLandmarks(
        shoulder=_pt(0.4, 0.4, 0, c=0.9),
        elbow=_pt(0.5, 0.45, 0, c=0.2),
        wrist=_pt(0.6, 0.5, 0, c=0.9),
        timestamp=1.0,
    )
    out = r.retarget(lm)
    assert out.status == RetargetingStatus.LOW_CONFIDENCE


def test_non_finite():
    r = SEWOrientationRetargeter()
    lm = ArmLandmarks(
        shoulder=_pt(0.4, 0.4, 0),
        elbow=_pt(float("nan"), 0.45, 0),
        wrist=_pt(0.6, 0.5, 0),
        timestamp=1.0,
    )
    out = r.retarget(lm)
    assert out.status == RetargetingStatus.NON_FINITE


def test_scale_invariance_direction():
    """Doubling limb lengths should not change unit vectors."""
    r = SEWOrientationRetargeter(RetargetingConfig(name="sew_orientation", treat_z_as_image_relative=False))
    a = r.retarget(_arm((0, 0, 0), (1, 0, 0), (1, 1, 0)))
    b = r.retarget(_arm((0, 0, 0), (2, 0, 0), (2, 2, 0)))
    np.testing.assert_allclose(a.upper_arm_unit, b.upper_arm_unit, atol=1e-9)
    np.testing.assert_allclose(a.lower_arm_unit, b.lower_arm_unit, atol=1e-9)
    assert a.elbow_bend_rad == pytest.approx(b.elbow_bend_rad, abs=1e-9)


def test_no_fake_joint_ik():
    r = SEWOrientationRetargeter(RetargetingConfig(name="sew_orientation", treat_z_as_image_relative=False))
    out = r.retarget(_arm((0, 0, 0), (0.2, 0.1, 0), (0.3, 0.2, 0.05)))
    assert out.is_commandable()
    assert out.joint_positions is None
    assert "missing_ik" in out.ik_status


def test_factory_and_shoulder_relative_preserved():
    sr = create_retargeter(RetargetingConfig(name="shoulder_relative"))
    assert isinstance(sr, ShoulderRelativeRetargeter)
    lm = ArmLandmarks(
        shoulder=_pt(0.45, 0.35, 0),
        elbow=None,
        wrist=_pt(0.55, 0.50, 0),
        timestamp=1.0,
    )
    out = sr.retarget(lm)
    assert out.status == RetargetingStatus.OK
    assert out.position is not None
    assert out.position.shape == (3,)


def test_shoulder_relative_regression_values():
    """Deterministic regression for baseline shoulder-relative mapping."""
    r = ShoulderRelativeRetargeter(
        RetargetingConfig(
            name="shoulder_relative",
            workspace_origin=(0.45, 0.0, 0.45),
            workspace_scale=(0.55, 0.55, 0.35),
            treat_z_as_image_relative=True,
        )
    )
    lm = ArmLandmarks(
        shoulder=_pt(0.5, 0.4, 0),
        elbow=None,
        wrist=_pt(0.6, 0.5, 0),
        timestamp=1.0,
    )
    out = r.retarget(lm)
    # disp = (0.1, 0.1, 0) → origin + scale * disp
    np.testing.assert_allclose(out.position, [0.45 + 0.055, 0.0 + 0.055, 0.45], atol=1e-9)
