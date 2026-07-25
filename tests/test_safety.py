"""Safety monitor policy tests."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.safety import ControlMode, SafetyConfig, SafetyMonitor  # noqa: E402
from vision_arm_control.workspace_limits import DEFAULT_WORKSPACE  # noqa: E402


def _monitor(**kwargs):
    cfg = SafetyConfig(
        workspace=DEFAULT_WORKSPACE,
        require_deadman=True,
        allow_real_robot=False,
        command_stale_sec=0.2,
    )
    return SafetyMonitor(config=cfg, **kwargs)


def test_pose_timeout_blocks():
    monitor = _monitor()
    decision = monitor.evaluate_position([0.5, 0.0, 0.4], now=1.0)
    assert not decision.accepted
    assert decision.reason == "pose_timeout"


def test_accepts_in_workspace_after_pose():
    monitor = _monitor(control_mode=ControlMode.DRY_RUN)
    monitor.note_pose(1.0)
    decision = monitor.evaluate_position([0.5, 0.0, 0.4], now=1.1)
    assert decision.accepted


def test_stale_timestamp_blocks():
    monitor = _monitor(control_mode=ControlMode.DRY_RUN)
    monitor.note_pose(1.0)
    decision = monitor.evaluate_position(
        [0.5, 0.0, 0.4],
        now=1.0,
        command_timestamp=0.70,
    )
    assert not decision.accepted
    assert decision.reason == "command_stale"


def test_fresh_timestamp_is_accepted():
    monitor = _monitor(control_mode=ControlMode.DRY_RUN)
    monitor.note_pose(1.0)
    decision = monitor.evaluate_position(
        [0.5, 0.0, 0.4],
        now=1.0,
        command_timestamp=0.90,
    )
    assert decision.accepted


def test_estop_blocks():
    monitor = _monitor()
    monitor.note_pose(1.0)
    monitor.request_emergency_stop()
    decision = monitor.evaluate_position([0.5, 0.0, 0.4], now=1.1)
    assert not decision.accepted
    assert decision.emergency_stop


def test_real_robot_requires_allow_flag():
    cfg = SafetyConfig(
        workspace=DEFAULT_WORKSPACE,
        allow_real_robot=False,
        require_deadman=True,
    )
    monitor = SafetyMonitor(config=cfg, control_mode=ControlMode.REAL_ROBOT)
    assert monitor.control_mode == ControlMode.DRY_RUN
