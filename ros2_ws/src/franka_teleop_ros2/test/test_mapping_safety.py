"""Unit tests runnable with or without a full ROS 2 install (pure logic)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

# Allow running from source tree without installation.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from franka_teleop_ros2.coordinate_mapping import (  # noqa: E402
    CameraIntrinsics,
    map_wrist_end_effector_position,
)
from franka_teleop_ros2.ik_utils import panda_geometric_ik  # noqa: E402
from franka_teleop_ros2.safety import ControlMode, SafetyConfig, SafetyMonitor  # noqa: E402
from franka_teleop_ros2.trajectory_filter import CommandRateLimiter  # noqa: E402
from franka_teleop_ros2.workspace_limits import (  # noqa: E402
    DEFAULT_WORKSPACE,
    PANDA_JOINT_LIMITS,
)


def test_yaml_defaults_exist():
    cfg = ROOT / "config" / "teleop.yaml"
    assert cfg.is_file()
    text = cfg.read_text()
    assert "allow_real_robot: false" in text
    assert "use_metric_depth: false" in text
    assert "command_stale_sec:" in text
    assert "watchdog_hz:" in text
    assert "control_mode: \"simulation\"" in text or "control_mode: simulation" in text


def test_mapping_stays_in_workspace_region():
    intrinsics = CameraIntrinsics(600, 600, 320, 240, 640, 480)
    out = map_wrist_end_effector_position(
        wrist_norm_xy=(0.55, 0.45),
        image_size=(640, 480),
        intrinsics=intrinsics,
        T_base_camera=np.eye(4),
        depth_m=None,
        relative_depth=None,
        relative_depth_scale=1.0,
        relative_depth_offset=0.8,
        use_metric_depth=False,
        shoulder_norm_xy=(0.45, 0.35),
        workspace_origin=(0.40, 0.0, 0.45),
        workspace_scale=(0.45, 0.50, 0.30),
        mapping_mode="shoulder_relative",
    )
    assert out.shape == (3,)
    assert 0.2 < out[0] < 0.8


def test_safety_blocks_without_pose():
    cfg = SafetyConfig(
        workspace=DEFAULT_WORKSPACE,
        allow_real_robot=False,
        require_deadman=False,
    )
    monitor = SafetyMonitor(config=cfg, control_mode=ControlMode.SIMULATION)
    monitor.set_deadman(True)
    decision = monitor.evaluate_position([0.4, 0.0, 0.4], now=1.0)
    assert not decision.accepted
    assert decision.reason == "pose_timeout"


def test_safety_estop():
    cfg = SafetyConfig(
        workspace=DEFAULT_WORKSPACE,
        allow_real_robot=False,
        require_deadman=False,
    )
    monitor = SafetyMonitor(config=cfg, control_mode=ControlMode.SIMULATION)
    monitor.set_deadman(True)
    monitor.note_pose(1.0)
    monitor.request_emergency_stop()
    decision = monitor.evaluate_position([0.4, 0.0, 0.4], now=1.1)
    assert decision.emergency_stop
    assert decision.reason == "emergency_stop"


def test_stale_timestamp_is_rejected():
    cfg = SafetyConfig(
        workspace=DEFAULT_WORKSPACE,
        allow_real_robot=False,
        require_deadman=False,
        command_stale_sec=0.20,
    )
    monitor = SafetyMonitor(config=cfg, control_mode=ControlMode.SIMULATION)
    monitor.note_pose(1.0)
    stale = monitor.evaluate_position(
        [0.4, 0.0, 0.4],
        now=1.0,
        command_timestamp=0.70,
    )
    fresh = monitor.evaluate_position(
        [0.4, 0.0, 0.4],
        now=1.0,
        command_timestamp=0.90,
    )
    assert not stale.accepted
    assert stale.reason == "command_stale"
    assert fresh.accepted


def test_unstamped_mock_command_is_not_marked_stale():
    cfg = SafetyConfig(
        workspace=DEFAULT_WORKSPACE,
        allow_real_robot=False,
        require_deadman=False,
        command_stale_sec=0.20,
    )
    monitor = SafetyMonitor(config=cfg, control_mode=ControlMode.SIMULATION)
    monitor.note_pose(1.0)
    decision = monitor.evaluate_position(
        [0.4, 0.0, 0.4],
        now=1.0,
        command_timestamp=None,
    )
    assert decision.accepted


def test_command_rate_limiter():
    limiter = CommandRateLimiter(max_hz=20.0)
    assert limiter.allow(1.00)
    assert not limiter.allow(1.02)
    assert limiter.allow(1.05)


def test_ik_returns_seven_joints_inside_limits():
    joints = panda_geometric_ik(0.4, 0.1, 0.45)
    assert joints is not None
    assert len(joints) == 7
    assert all(math.isfinite(value) for value in joints)
    assert PANDA_JOINT_LIMITS.contains(joints)


def test_joint_limit_validator_rejects_invalid_command():
    invalid = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 4.0]
    assert not PANDA_JOINT_LIMITS.contains(invalid)


def test_real_robot_forced_off_in_config():
    cfg = SafetyConfig(
        workspace=DEFAULT_WORKSPACE,
        allow_real_robot=False,
        require_deadman=True,
    )
    monitor = SafetyMonitor(config=cfg, control_mode=ControlMode.REAL_ROBOT)
    assert monitor.control_mode == ControlMode.DRY_RUN
