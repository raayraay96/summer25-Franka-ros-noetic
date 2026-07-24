"""Safety monitor policy tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.safety import ControlMode, SafetyConfig, SafetyMonitor  # noqa: E402
from vision_arm_control.workspace_limits import DEFAULT_WORKSPACE  # noqa: E402


def _monitor(**kwargs):
    cfg = SafetyConfig(workspace=DEFAULT_WORKSPACE, require_deadman=True, allow_real_robot=False)
    return SafetyMonitor(config=cfg, **kwargs)


def test_pose_timeout_blocks():
    m = _monitor()
    d = m.evaluate_position([0.5, 0.0, 0.4], now=1.0)
    assert not d.accepted
    assert d.reason == "pose_timeout"


def test_accepts_in_workspace_after_pose():
    m = _monitor(control_mode=ControlMode.DRY_RUN)
    m.note_pose(1.0)
    d = m.evaluate_position([0.5, 0.0, 0.4], now=1.1)
    assert d.accepted


def test_estop_blocks():
    m = _monitor()
    m.note_pose(1.0)
    m.request_emergency_stop()
    d = m.evaluate_position([0.5, 0.0, 0.4], now=1.1)
    assert not d.accepted
    assert d.emergency_stop


def test_real_robot_requires_allow_flag():
    cfg = SafetyConfig(workspace=DEFAULT_WORKSPACE, allow_real_robot=False, require_deadman=True)
    m = SafetyMonitor(config=cfg, control_mode=ControlMode.REAL_ROBOT)
    # Forced to dry_run in __post_init__ when allow_real_robot is False
    assert m.control_mode == ControlMode.DRY_RUN
