"""Pose timeout and staleness tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.pose_timeout import CommandStalenessMonitor, PoseTimeoutMonitor  # noqa: E402


def test_pose_timeout_invalid_until_pose():
    m = PoseTimeoutMonitor(timeout_sec=0.5)
    assert not m.is_valid(0.0)
    assert m.age(0.0) is None


def test_pose_timeout_valid_within_window():
    m = PoseTimeoutMonitor(timeout_sec=0.5)
    m.note_pose(1.0)
    assert m.is_valid(1.2)
    assert not m.is_valid(1.6)


def test_pose_timeout_exact_boundary():
    m = PoseTimeoutMonitor(timeout_sec=0.5)
    m.note_pose(1.0)
    assert m.is_valid(1.5)
    assert not m.is_valid(1.5000001)


def test_pose_timeout_age():
    m = PoseTimeoutMonitor(timeout_sec=1.0)
    m.note_pose(10.0)
    assert m.age(10.25) == pytest.approx(0.25)


def test_pose_timeout_reset():
    m = PoseTimeoutMonitor(timeout_sec=0.5)
    m.note_pose(1.0)
    assert m.is_valid(1.1)
    m.reset()
    assert not m.is_valid(1.2)


def test_command_staleness():
    c = CommandStalenessMonitor(timeout_sec=0.2)
    assert c.is_stale(0.0)
    c.note_command(1.0)
    assert not c.is_stale(1.1)
    assert c.is_stale(1.3)


def test_command_staleness_exact_boundary():
    c = CommandStalenessMonitor(timeout_sec=0.2)
    c.note_command(1.0)
    assert not c.is_stale(1.2)
    assert c.is_stale(1.2000001)
