"""Safety monitor policy tests."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src" / "vision_arm_control" / "src"))

from vision_arm_control.safety import ControlMode, SafetyConfig, SafetyMonitor  # noqa: E402
from vision_arm_control.workspace_limits import DEFAULT_WORKSPACE, PANDA_JOINT_LIMITS  # noqa: E402


def _cfg(**kwargs):
    base = dict(
        workspace=DEFAULT_WORKSPACE,
        joint_limits=PANDA_JOINT_LIMITS,
        require_deadman=True,
        allow_real_robot=False,
        workspace_mode="reject",
        pose_timeout_sec=0.5,
    )
    base.update(kwargs)
    return SafetyConfig(**base)


def _monitor(cfg=None, **kwargs):
    return SafetyMonitor(config=cfg or _cfg(), **kwargs)


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
    assert d.position is not None


def test_estop_blocks():
    m = _monitor()
    m.note_pose(1.0)
    m.request_emergency_stop()
    d = m.evaluate_position([0.5, 0.0, 0.4], now=1.1)
    assert not d.accepted
    assert d.emergency_stop


def test_estop_clear_allows_again():
    m = _monitor()
    m.note_pose(1.0)
    m.request_emergency_stop()
    assert not m.evaluate_position([0.5, 0.0, 0.4], now=1.1).accepted
    m.clear_emergency_stop()
    m.note_pose(1.2)
    d = m.evaluate_position([0.5, 0.0, 0.4], now=1.25)
    assert d.accepted


def test_real_robot_requires_allow_flag():
    cfg = _cfg(allow_real_robot=False, require_deadman=True)
    m = SafetyMonitor(config=cfg, control_mode=ControlMode.REAL_ROBOT)
    assert m.control_mode == ControlMode.DRY_RUN


def test_real_robot_deadman_required():
    cfg = _cfg(allow_real_robot=True, require_deadman=True)
    m = SafetyMonitor(config=cfg, control_mode=ControlMode.REAL_ROBOT)
    m.note_pose(1.0)
    d = m.evaluate_position([0.5, 0.0, 0.4], now=1.1)
    assert not d.accepted
    assert d.reason == "deadman_not_enabled"
    m.set_deadman(True)
    d2 = m.evaluate_position([0.5, 0.0, 0.4], now=1.15)
    assert d2.accepted


def test_outside_workspace_reject():
    m = _monitor()
    m.note_pose(1.0)
    d = m.evaluate_position([2.0, 0.0, 0.4], now=1.1)
    assert not d.accepted
    assert d.reason == "outside_workspace"


def test_workspace_clamp_mode():
    m = _monitor(cfg=_cfg(workspace_mode="clamp"))
    m.note_pose(1.0)
    d = m.evaluate_position([2.0, 0.0, 0.4], now=1.1)
    assert d.accepted
    assert d.position is not None
    assert d.position[0] == pytest.approx(DEFAULT_WORKSPACE.x_max)
    assert d.reason == "clamped_to_workspace"


def test_simulation_mode_without_deadman_when_not_required():
    cfg = _cfg(require_deadman=False, allow_real_robot=False)
    m = SafetyMonitor(config=cfg, control_mode=ControlMode.SIMULATION)
    m.note_pose(1.0)
    d = m.evaluate_position([0.5, 0.0, 0.4], now=1.1)
    assert d.accepted


def test_evaluate_joints_ok_and_out_of_range():
    m = _monitor()
    q_ok = [0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.0]
    assert m.evaluate_joints(q_ok).accepted
    q_bad = [10.0, 0.0, 0.0, -2.0, 0.0, 1.5, 0.0]
    d = m.evaluate_joints(q_bad)
    assert not d.accepted
    assert d.reason == "outside_joint_limits"


def test_evaluate_joints_estop():
    m = _monitor()
    m.request_emergency_stop()
    d = m.evaluate_joints([0.0] * 7)
    assert d.emergency_stop
