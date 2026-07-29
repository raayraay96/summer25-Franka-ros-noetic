"""Robot backend protocol — pure Python, no ROS required for dry_run."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence, runtime_checkable

import numpy as np


@dataclass
class BackendResult:
    accepted: bool
    reason: str
    mode: str
    position: Optional[np.ndarray] = None
    joint_positions: Optional[np.ndarray] = None
    metadata: dict = field(default_factory=dict)


@runtime_checkable
class RobotBackend(Protocol):
    name: str

    def send_cartesian_target(
        self,
        position: Sequence[float],
        orientation_xyzw: Optional[Sequence[float]] = None,
    ) -> BackendResult:
        ...

    def reset(self) -> None:
        ...


class DryRunBackend:
    """Default backend: accept and record commands without hardware I/O."""

    def __init__(self) -> None:
        self.name = "dry_run"
        self.history: List[np.ndarray] = []

    def reset(self) -> None:
        self.history.clear()

    def send_cartesian_target(
        self,
        position: Sequence[float],
        orientation_xyzw: Optional[Sequence[float]] = None,
    ) -> BackendResult:
        p = np.asarray(position, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(p)):
            return BackendResult(False, "non_finite", self.name)
        self.history.append(p.copy())
        return BackendResult(True, "dry_run_logged", self.name, position=p)


class Ros2FakeHardwareBackend:
    """Placeholder backend for ROS 2 fake-hardware integration.

    Actual ROS 2 publishers live in the ROS 2 package nodes. This class
    documents the interface and records targets for pure-Python tests.
    """

    def __init__(self) -> None:
        self.name = "ros2_fake_hardware"
        self.history: List[np.ndarray] = []

    def reset(self) -> None:
        self.history.clear()

    def send_cartesian_target(
        self,
        position: Sequence[float],
        orientation_xyzw: Optional[Sequence[float]] = None,
    ) -> BackendResult:
        p = np.asarray(position, dtype=np.float64).reshape(3)
        if not np.all(np.isfinite(p)):
            return BackendResult(False, "non_finite", self.name)
        self.history.append(p.copy())
        return BackendResult(
            True,
            "recorded_for_ros2_fake_hardware_bridge",
            self.name,
            position=p,
            metadata={"ros2_bridge": "external_node"},
        )


class OptionalMoveItBackend:
    """MoveIt backend stub — only when a verified MoveIt session exists.

    Not enabled by default. Calling without a verified session returns reject.
    """

    def __init__(self, verified: bool = False) -> None:
        self.name = "optional_moveit"
        self.verified = bool(verified)

    def reset(self) -> None:
        return None

    def send_cartesian_target(
        self,
        position: Sequence[float],
        orientation_xyzw: Optional[Sequence[float]] = None,
    ) -> BackendResult:
        if not self.verified:
            return BackendResult(
                False,
                "moveit_not_verified_in_this_environment",
                self.name,
            )
        return BackendResult(
            False,
            "moveit_verified_flag_set_but_no_live_session",
            self.name,
        )


def create_backend(name: str = "dry_run", **kwargs) -> RobotBackend:
    key = str(name).lower().strip()
    if key in ("dry_run", "dry-run"):
        return DryRunBackend()
    if key in ("ros2_fake_hardware", "ros2", "fake_hardware"):
        return Ros2FakeHardwareBackend()
    if key in ("optional_moveit", "moveit"):
        return OptionalMoveItBackend(verified=bool(kwargs.get("verified", False)))
    raise ValueError(f"Unknown backend '{name}'. Supported: dry_run, ros2_fake_hardware, optional_moveit")
