"""Unit tests runnable with or without a full ROS 2 install (pure logic)."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

# Allow running from source tree without install
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from franka_teleop_ros2.coordinate_mapping import (  # noqa: E402
    CameraIntrinsics,
    map_wrist_end_effector_position,
)
from franka_teleop_ros2.safety import ControlMode, SafetyConfig, SafetyMonitor  # noqa: E402
from franka_teleop_ros2.workspace_limits import DEFAULT_WORKSPACE, AxisAlignedBounds  # noqa: E402
from franka_teleop_ros2.ik_utils import panda_geometric_ik  # noqa: E402


def test_yaml_defaults_exist():
    cfg = ROOT / "config" / "teleop.yaml"
    assert cfg.is_file()
    text = cfg.read_text()
    assert "allow_real_robot: false" in text
    assert "use_metric_depth: false" in text
    assert "control_mode: \"simulation\"" in text or "control_mode: simulation" in text


def test_mapping_stays_in_workspace_region():
    K = CameraIntrinsics(600, 600, 320, 240, 640, 480)
    out = map_wrist_end_effector_position(
        wrist_norm_xy=(0.55, 0.45),
        image_size=(640, 480),
        intrinsics=K,
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
    cfg = SafetyConfig(workspace=DEFAULT_WORKSPACE, allow_real_robot=False, require_deadman=False)
    m = SafetyMonitor(config=cfg, control_mode=ControlMode.SIMULATION)
    m.set_deadman(True)
    d = m.evaluate_position([0.4, 0.0, 0.4], now=1.0)
    assert not d.accepted


def test_safety_estop():
    cfg = SafetyConfig(workspace=DEFAULT_WORKSPACE, allow_real_robot=False, require_deadman=False)
    m = SafetyMonitor(config=cfg, control_mode=ControlMode.SIMULATION)
    m.set_deadman(True)
    m.note_pose(1.0)
    m.request_emergency_stop()
    d = m.evaluate_position([0.4, 0.0, 0.4], now=1.1)
    assert d.emergency_stop


def test_ik_returns_seven_joints():
    q = panda_geometric_ik(0.4, 0.1, 0.45)
    assert q is not None
    assert len(q) == 7
    assert all(math.isfinite(v) for v in q)


def test_real_robot_forced_off_in_config():
    cfg = SafetyConfig(workspace=DEFAULT_WORKSPACE, allow_real_robot=False, require_deadman=True)
    m = SafetyMonitor(config=cfg, control_mode=ControlMode.REAL_ROBOT)
    assert m.control_mode == ControlMode.DRY_RUN
