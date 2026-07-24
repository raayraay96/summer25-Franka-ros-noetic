"""Pose-loss and command-staleness detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class PoseTimeoutMonitor:
    """Tracks last successful pose observation time.

    When (now - last_pose_time) exceeds timeout_sec, the pipeline should
    hold or stop motion rather than tracking a stale target.
    """

    timeout_sec: float = 0.5
    _last_pose_time: Optional[float] = None
    _has_pose: bool = False

    def reset(self) -> None:
        self._last_pose_time = None
        self._has_pose = False

    def note_pose(self, timestamp: float) -> None:
        self._last_pose_time = float(timestamp)
        self._has_pose = True

    def is_valid(self, now: float) -> bool:
        if not self._has_pose or self._last_pose_time is None:
            return False
        return (float(now) - self._last_pose_time) <= self.timeout_sec

    def age(self, now: float) -> Optional[float]:
        if self._last_pose_time is None:
            return None
        return float(now) - self._last_pose_time


@dataclass
class CommandStalenessMonitor:
    """Detect when outgoing robot commands become stale."""

    timeout_sec: float = 0.2
    _last_command_time: Optional[float] = None

    def note_command(self, timestamp: float) -> None:
        self._last_command_time = float(timestamp)

    def is_stale(self, now: float) -> bool:
        if self._last_command_time is None:
            return True
        return (float(now) - self._last_command_time) > self.timeout_sec
