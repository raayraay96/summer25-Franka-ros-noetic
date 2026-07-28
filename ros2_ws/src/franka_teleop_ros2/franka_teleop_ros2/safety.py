"""Safety policy helpers for teleoperation command gating."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

import numpy as np

from .pose_timeout import PoseTimeoutMonitor
from .workspace_limits import AxisAlignedBounds, JointLimits, validate_or_clamp_position


class ControlMode(str, Enum):
    DRY_RUN = "dry_run"
    SIMULATION = "simulation"
    REAL_ROBOT = "real_robot"


@dataclass
class SafetyConfig:
    workspace: AxisAlignedBounds
    joint_limits: Optional[JointLimits] = None
    max_linear_velocity: float = 0.25
    max_joint_velocity: float = 1.0
    max_command_hz: float = 50.0
    pose_timeout_sec: float = 0.5
    command_stale_sec: float = 0.2
    workspace_mode: str = "reject"  # reject | clamp
    require_deadman: bool = True
    allow_real_robot: bool = False


@dataclass
class SafetyDecision:
    accepted: bool
    reason: str
    position: Optional[np.ndarray] = None
    emergency_stop: bool = False


@dataclass
class SafetyMonitor:
    """Gate Cartesian targets before they reach a controller."""

    config: SafetyConfig
    control_mode: ControlMode = ControlMode.DRY_RUN
    deadman_enabled: bool = False
    emergency_stop: bool = False
    pose_monitor: PoseTimeoutMonitor = field(init=False)

    def __post_init__(self) -> None:
        self.pose_monitor = PoseTimeoutMonitor(timeout_sec=self.config.pose_timeout_sec)
        if self.control_mode == ControlMode.REAL_ROBOT and not self.config.allow_real_robot:
            # Hard policy: real robot requires explicit allow flag in config/launch.
            self.control_mode = ControlMode.DRY_RUN

    def request_emergency_stop(self) -> None:
        self.emergency_stop = True

    def clear_emergency_stop(self) -> None:
        self.emergency_stop = False

    def set_deadman(self, enabled: bool) -> None:
        self.deadman_enabled = bool(enabled)

    def note_pose(self, timestamp: float) -> None:
        self.pose_monitor.note_pose(timestamp)

    def evaluate_position(
        self,
        position: Sequence[float],
        now: float,
    ) -> SafetyDecision:
        if self.emergency_stop:
            return SafetyDecision(False, "emergency_stop", emergency_stop=True)

        if self.control_mode == ControlMode.REAL_ROBOT and not self.config.allow_real_robot:
            return SafetyDecision(False, "real_robot_not_allowed")

        if self.config.require_deadman and self.control_mode == ControlMode.REAL_ROBOT:
            if not self.deadman_enabled:
                return SafetyDecision(False, "deadman_not_enabled")

        if not self.pose_monitor.is_valid(now):
            return SafetyDecision(False, "pose_timeout")

        pos, ok, reason = validate_or_clamp_position(
            position,
            self.config.workspace,
            mode=self.config.workspace_mode,
        )
        if not ok or pos is None:
            return SafetyDecision(False, reason)
        return SafetyDecision(True, reason, position=pos)

    def evaluate_joints(self, joints: Sequence[float]) -> SafetyDecision:
        if self.emergency_stop:
            return SafetyDecision(False, "emergency_stop", emergency_stop=True)
        limits = self.config.joint_limits
        if limits is None:
            return SafetyDecision(True, "no_joint_limits_configured", position=None)
        if limits.contains(joints):
            return SafetyDecision(True, "ok")
        return SafetyDecision(False, "outside_joint_limits")
