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
    validate_or_clamp_position,
)


def test_contains_inside():
    assert DEFAULT_WORKSPACE.contains([0.5, 0.0, 0.4])


def test_contains_outside():
    assert not DEFAULT_WORKSPACE.contains([2.0, 0.0, 0.4])


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


def test_joint_limits_home_ish():
    # A mid-range configuration should pass
    q = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0]
    assert PANDA_JOINT_LIMITS.contains(q)


def test_joint_limits_clamp():
    q = [10.0, 0.0, 0.0, -2.0, 0.0, 1.5, 0.0]
    clamped = PANDA_JOINT_LIMITS.clamp(q)
    assert clamped[0] == pytest.approx(PANDA_JOINT_LIMITS.upper[0])
