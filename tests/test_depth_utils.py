"""Depth utility tests (relative depth helpers)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.depth_utils import (  # noqa: E402
    disparity_to_relative_depth,
    expected_input_size,
    sample_depth_at_pixel,
)


def test_expected_input_size():
    assert expected_input_size() == (640, 192)


def test_disparity_to_relative_depth():
    disp = np.array([[0.5, 1.0], [2.0, 0.0]], dtype=np.float32)
    depth = disparity_to_relative_depth(disp, eps=1e-6)
    assert depth[0, 0] == pytest.approx(2.0)
    assert depth[0, 1] == pytest.approx(1.0)
    assert depth[1, 0] == pytest.approx(0.5)
    # zero disparity clamped by eps
    assert depth[1, 1] == pytest.approx(1e6)


def test_sample_depth_at_pixel_center():
    d = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert sample_depth_at_pixel(d, 1.0, 1.0) == pytest.approx(5.0)


def test_sample_depth_clamps_out_of_bounds():
    d = np.ones((2, 2), dtype=np.float32) * 3.0
    assert sample_depth_at_pixel(d, -5.0, -5.0) == pytest.approx(3.0)
    assert sample_depth_at_pixel(d, 99.0, 99.0) == pytest.approx(3.0)
