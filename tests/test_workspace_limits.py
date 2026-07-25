"""Workspace and joint limit tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.workspace_limits import (  # noqa: E402
    DEFAULT_WORKSPACE,
    PANDA_JOINT_LIMITS,
    AxisAlignedBounds,
    JointLimits,
    validate_or_clamp_position,
)


def test_contains_inside():
    assert DEFAULT_WORKSPACE.contains([0.5, 0.0, 0.4])


def test_contains_outside():
    assert not DEFAULT_WORKSPACE.contains([2.0, 0.0, 0.4])


def test_contains_on_boundary():
    b = DEFAULT_WORKSPACE
    assert b.contains([b.x_min, b.y_min, b.z_min])
    assert b.contains([b.x_max, b.y_max, b.z_max])


def test_reject_mode():
    pos, ok, reason = validate_or_clamp_position([2.0, 0.0, 0.4], DEFAULT_WORKSPACE, mode="reject")
    assert pos is None
    assert not ok
    assert reason == "outside_workspace"


def test_clamp_mode():
    pos, ok, reason = validate_or_clamp_position([2.0, 0.0, 0.4], DEFAULT_WORKSPACE, mode="clamp")
    assert ok
    assert pos is not None
    assert pos[0] == pytest.approx(DEFAULT_WORKSPACE.x_max)
    assert reason == "clamped_to_workspace"


def test_clamp_all_axes():
    pos, ok, reason = validate_or_clamp_position(
        [-10.0, 10.0, 100.0], DEFAULT_WORKSPACE, mode="clamp"
    )
    assert ok
    np.testing.assert_allclose(
        pos,
        [
            DEFAULT_WORKSPACE.x_min,
            DEFAULT_WORKSPACE.y_max,
            DEFAULT_WORKSPACE.z_max,
        ],
    )


def test_validate_rejects_wrong_shape():
    pos, ok, reason = validate_or_clamp_position([1.0, 2.0], DEFAULT_WORKSPACE, mode="reject")
    assert not ok
    assert pos is None
    assert "length-3" in reason


def test_inside_returns_ok():
    pos, ok, reason = validate_or_clamp_position([0.5, 0.0, 0.4], DEFAULT_WORKSPACE, mode="reject")
    assert ok
    assert reason == "ok"
    np.testing.assert_allclose(pos, [0.5, 0.0, 0.4])


def test_joint_limits_home_ish():
    q = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0]
    assert PANDA_JOINT_LIMITS.contains(q)


def test_joint_limits_clamp():
    q = [10.0, 0.0, 0.0, -2.0, 0.0, 1.5, 0.0]
    clamped = PANDA_JOINT_LIMITS.clamp(q)
    assert clamped[0] == pytest.approx(PANDA_JOINT_LIMITS.upper[0])


def test_joint_limits_length_mismatch_contains_false():
    assert not PANDA_JOINT_LIMITS.contains([0.0, 0.0])


def test_joint_limits_construction_requires_equal_length():
    with pytest.raises(ValueError):
        JointLimits(lower=(-1.0,), upper=(-1.0, 1.0))


def test_custom_bounds_clamp():
    b = AxisAlignedBounds(0.0, 1.0, -0.5, 0.5, 0.0, 1.0)
    out = b.clamp([2.0, -2.0, -1.0])
    np.testing.assert_allclose(out, [1.0, -0.5, 0.0])
