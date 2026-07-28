"""Shared landmark and retargeting schemas (pure Python, no ROS)."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence, Tuple

import numpy as np


class RetargetingStatus(str, Enum):
    """Status / reason codes for retargeting and gating."""

    OK = "ok"
    MISSING_SHOULDER = "missing_shoulder"
    MISSING_ELBOW = "missing_elbow"
    MISSING_WRIST = "missing_wrist"
    LOW_CONFIDENCE = "low_confidence"
    UPPER_ARM_NORM_NEAR_ZERO = "upper_arm_norm_near_zero"
    LOWER_ARM_NORM_NEAR_ZERO = "lower_arm_norm_near_zero"
    COLLINEAR_ARM_PLANE = "collinear_arm_plane"
    STALE_TIMESTAMP = "stale_timestamp"
    NON_FINITE = "non_finite"
    HOLDING_LAST_SAFE = "holding_last_safe"
    NO_COMMAND = "no_command"
    DEGENERATE_POSE = "degenerate_pose"
    MAPPING_ERROR = "mapping_error"


@dataclass(frozen=True)
class LandmarkPoint:
    """A single 3D or 2.5D keypoint with confidence."""

    x: float
    y: float
    z: float
    confidence: float = 1.0
    name: str = ""
    frame_id: str = "human_body"

    def as_array(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=np.float64)

    def is_finite(self) -> bool:
        return bool(np.isfinite([self.x, self.y, self.z, self.confidence]).all())


@dataclass(frozen=True)
class ArmLandmarks:
    """Shoulder–elbow–wrist observation for one arm."""

    shoulder: Optional[LandmarkPoint]
    elbow: Optional[LandmarkPoint]
    wrist: Optional[LandmarkPoint]
    timestamp: float
    frame_id: str = "human_body"
    source: str = "unknown"

    def points(self) -> Tuple[Optional[LandmarkPoint], ...]:
        return (self.shoulder, self.elbow, self.wrist)

    def min_confidence(self) -> float:
        vals = [p.confidence for p in self.points() if p is not None]
        if not vals:
            return 0.0
        return float(min(vals))

    def all_present(self) -> bool:
        return all(p is not None for p in self.points())


@dataclass
class RetargetingTarget:
    """Documented retargeting output suitable for Panda EE teleoperation.

    Joint targets are optional and only populated when a verified solver runs.
    Missing IK is represented by ``joint_positions is None`` and
    ``ik_status`` explaining the gap — never fabricated joint solutions.
    """

    position: Optional[np.ndarray]
    orientation_xyzw: Optional[np.ndarray] = None  # quaternion xyzw
    direction: Optional[np.ndarray] = None  # preferred EE approach / lower-arm dir
    elbow_plane_normal: Optional[np.ndarray] = None
    upper_arm_unit: Optional[np.ndarray] = None
    lower_arm_unit: Optional[np.ndarray] = None
    elbow_bend_rad: Optional[float] = None
    confidence: float = 0.0
    status: RetargetingStatus = RetargetingStatus.NO_COMMAND
    reason: str = ""
    timestamp: float = 0.0
    joint_positions: Optional[np.ndarray] = None
    ik_status: str = "not_implemented"
    metadata: dict = field(default_factory=dict)

    def is_commandable(self) -> bool:
        return (
            self.status
            in (
                RetargetingStatus.OK,
                RetargetingStatus.HOLDING_LAST_SAFE,
            )
            and self.position is not None
            and np.all(np.isfinite(self.position))
        )


def as_vec3(seq: Sequence[float]) -> np.ndarray:
    v = np.asarray(seq, dtype=np.float64).reshape(-1)
    if v.shape[0] != 3:
        raise ValueError("expected length-3 vector")
    return v
