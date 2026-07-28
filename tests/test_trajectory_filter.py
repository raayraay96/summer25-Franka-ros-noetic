"""Trajectory filter tests."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.trajectory_filter import (  # noqa: E402
    CommandRateLimiter,
    LowPassFilter3D,
    VelocityLimiter3D,
)


def test_low_pass_first_sample():
    f = LowPassFilter3D(alpha=0.5)
    out = f.update([1.0, 2.0, 3.0])
    np.testing.assert_allclose(out, [1.0, 2.0, 3.0])


def test_low_pass_smoothing():
    f = LowPassFilter3D(alpha=0.5)
    f.update([0.0, 0.0, 0.0])
    out = f.update([1.0, 0.0, 0.0])
    assert out[0] == pytest.approx(0.5)


def test_low_pass_rejects_bad_alpha():
    f = LowPassFilter3D(alpha=0.0)
    with pytest.raises(ValueError):
        f.update([1.0, 0.0, 0.0])


def test_low_pass_rejects_bad_shape():
    f = LowPassFilter3D(alpha=0.3)
    with pytest.raises(ValueError):
        f.update([1.0, 2.0])


def test_low_pass_reset():
    f = LowPassFilter3D(alpha=0.5)
    f.update([1.0, 0.0, 0.0])
    f.reset()
    out = f.update([0.0, 0.0, 0.0])
    np.testing.assert_allclose(out, [0.0, 0.0, 0.0])


def test_velocity_limiter_caps_step():
    lim = VelocityLimiter3D(max_linear_velocity=1.0)
    lim.update([0.0, 0.0, 0.0], dt=0.1)
    out = lim.update([10.0, 0.0, 0.0], dt=0.1)
    assert out[0] == pytest.approx(0.1)


def test_velocity_limiter_allows_small_step():
    lim = VelocityLimiter3D(max_linear_velocity=1.0)
    lim.update([0.0, 0.0, 0.0], dt=0.1)
    out = lim.update([0.05, 0.0, 0.0], dt=0.1)
    assert out[0] == pytest.approx(0.05)


def test_velocity_limiter_rejects_nonpositive_dt():
    lim = VelocityLimiter3D(max_linear_velocity=1.0)
    with pytest.raises(ValueError):
        lim.update([0.0, 0.0, 0.0], dt=0.0)


def test_velocity_limiter_reset():
    lim = VelocityLimiter3D(max_linear_velocity=0.1)
    lim.update([0.0, 0.0, 0.0], dt=0.1)
    lim.update([1.0, 0.0, 0.0], dt=0.1)
    lim.reset()
    out = lim.update([5.0, 0.0, 0.0], dt=0.1)
    # after reset, first sample is accepted as-is
    assert out[0] == pytest.approx(5.0)


def test_rate_limiter():
    r = CommandRateLimiter(max_hz=10.0)
    assert r.allow(0.0)
    assert not r.allow(0.05)
    assert r.allow(0.11)


def test_rate_limiter_disabled_when_nonpositive():
    r = CommandRateLimiter(max_hz=0.0)
    assert r.allow(0.0)
    assert r.allow(0.001)
