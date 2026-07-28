"""Tests for shared demo motion profile (loop closure + panel bytes)."""
from __future__ import annotations

import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from franka_teleop_ros2.motion_profile import (  # noqa: E402
    DEMO_PERIOD_SEC,
    draw_mock_landmark_panel_rgb,
    landmark_payload,
    wrist_xy,
)


def test_loop_closes_after_period():
    a = wrist_xy(0.0, scale=0.18, period=DEMO_PERIOD_SEC)
    b = wrist_xy(DEMO_PERIOD_SEC, scale=0.18, period=DEMO_PERIOD_SEC)
    assert math.isclose(a[0], b[0], abs_tol=1e-9)
    assert math.isclose(a[1], b[1], abs_tol=1e-9)


def test_motion_has_range():
    xs, ys = [], []
    for i in range(48):
        x, y = wrist_xy(i * DEMO_PERIOD_SEC / 48.0, scale=0.18)
        xs.append(x)
        ys.append(y)
    assert max(xs) - min(xs) > 0.2
    assert max(ys) - min(ys) > 0.1


def test_payload_mock_label():
    p = landmark_payload(0.5)
    assert p["source"] == "mock"
    assert p["depth_used_for_control"] is False
    assert p["label"] == "Mock landmark input"
    names = {lm["name"] for lm in p["landmarks"]}
    assert {"right_shoulder", "right_elbow", "right_wrist"} <= names


def test_panel_rgb_size():
    rgb = draw_mock_landmark_panel_rgb(1.0, width=160, height=120)
    assert len(rgb) == 160 * 120 * 3
